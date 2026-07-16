#!/usr/bin/env python3
"""
오케스트레이션 — 한 번의 명령으로 cfg 스윕 전 과정 자동 실행.

실행 순서 (cfg 리스트를 재설정하기 전까지 전부 자동):
  1. 심판(judge) 분류기 준비 (없으면 원본 60k로 학습 후 동결)
  2. 원본 부분집합(크기 매칭) 준비 → TRTR 천장 학습 (clf_seed별)
  3. cfg마다: 데이터 생성 → TSTR 분류기 학습 (clf_seed별) → 충실도/다양성 진단
  4. results/results_{stage}.csv 집계 (cfg, seed, accuracy, gap, fidelity, diversity)
  5. plot.py 호출 → 발표용 그림 세트(png) 저장

이미 존재하는 데이터셋/결과는 자동 스킵(재실행 안전). 처음부터 다시 하려면 --force.

사용 예 (src/src 에서):
  # 1차 coarse: s ∈ {0,1,2,3,4,6,8}, cfg당 2만 장, 1 seed
  python experiments/run_sweep.py --stage coarse

  # 2차 fine: coarse 곡선 보고 관심 구간 추가
  python experiments/run_sweep.py --stage fine --cfgs 0 0.25 0.5 0.75 1 --n-per-class 2000

  # 최종(공정 비교): 생성 6만 장 vs 원본 train 60k 전체 × 3 seed
  #   --real-n-per-class 0 = TRTR을 확산 모델과 같은 60k 전체로 학습
  python experiments/run_sweep.py --stage full60k --cfgs -0.5 -0.3 -0.25 0 \
      --n-per-class 6000 --real-n-per-class 0 --clf-seeds 0 1 2

  # 소규모 데이터 실험 (main.py train --n-per-class 100 으로 학습한 ckpt 사용):
  #   TRTR이 확산 모델 학습 서브셋(real_n100_seed0)과 동일 데이터로 학습됨
  python experiments/run_sweep.py --stage n1000 --n-per-class 100 \
      --sd-ckpt checkpoints/mnist_dit_n100.pth
"""
import argparse
import csv
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from accelerate import Accelerator

import config as C
import generate as G
import make_real_subset as R
import train_classifier as TC
import judge_and_diversity as JD
import plot as P

COARSE_CFGS = [-0.5, -0.4, -0.3, -0.25, -0.2, -0.1, 0]

CSV_FIELDS = ['stage', 'kind', 'cfg_scale', 'n_per_class', 'gen_seed', 'clf_seed',
              'accuracy', 'gap', 'fidelity', 'diversity', 'data_dir', 'result_json']


def run(stage, cfgs, n_per_class, clf_seeds, gen_seed, sd_ckpt, batch_size,
        force=False, skip_plot=False, real_n_per_class=None):
    C.ensure_dirs()
    t0 = time.time()

    # TRTR용 원본 크기: 기본은 생성과 동일(크기 매칭), 0이면 train 60k 전체.
    # 60k 실험은 MNIST 클래스 불균형(최소 5421장) 때문에 균등 6000/class 추출이 불가능하므로
    # --real-n-per-class 0 으로 전체를 쓴다 (확산 모델이 학습한 것과 동일 데이터).
    real_npc = n_per_class if real_n_per_class is None else real_n_per_class
    real_total = 60000 if real_npc == 0 else real_npc * 10
    print(f'=== sweep [{stage}] cfgs={cfgs}  gen n/class={n_per_class}  '
          f'real={real_total}장  clf_seeds={clf_seeds} ===')

    # 1. 심판 준비 (동결, 1회)
    JD.train_judge()
    judge = JD.load_judge()

    rows = []

    # 2. TRTR 천장
    rdir = C.real_dir(real_npc, seed=0)
    if not C.dataset_exists(rdir) or force:
        R.make_real_subset(real_npc, seed=0, out_dir=rdir)
    else:
        print(f'[skip] 원본 부분집합 존재: {rdir}')
    rdiag = JD.diagnose(rdir, judge=judge, force=force)   # 참고용 baseline

    trtr_accs = []
    for s in clf_seeds:
        res = TC.train_and_eval(rdir, s, force=force)
        trtr_accs.append(res['accuracy'])
        rows.append(dict(stage=stage, kind='TRTR', cfg_scale='', n_per_class=real_npc,
                         gen_seed=0, clf_seed=s, accuracy=res['accuracy'], gap='',
                         fidelity=rdiag['fidelity'], diversity=rdiag['diversity'],
                         data_dir=rdir.name, result_json=str(TC.result_path(rdir, s))))
    trtr_mean = sum(trtr_accs) / len(trtr_accs)
    print(f'[TRTR] 천장선(원본 {real_total}장, {len(clf_seeds)} seed 평균) = {trtr_mean:.4f}')

    # 3. cfg 스윕: 생성 → TSTR → 진단
    accel = Accelerator()
    sd_model = None   # 첫 생성 필요 시에만 로드 (전 cfg 재사용)
    for cfg in cfgs:
        gdir = C.gen_dir(cfg, n_per_class, gen_seed)
        if not C.dataset_exists(gdir) or force:
            if sd_model is None:
                sd_model = G.load_sd_model(sd_ckpt, accel.device)
            G.generate(cfg, n_per_class, seed=gen_seed, out_dir=gdir,
                       batch_size=batch_size, model=sd_model, accel=accel)
        else:
            print(f'[skip] 생성 데이터 존재: {gdir}')

        diag = JD.diagnose(gdir, judge=judge, force=force)
        for s in clf_seeds:
            res = TC.train_and_eval(gdir, s, force=force)
            rows.append(dict(stage=stage, kind='TSTR', cfg_scale=C.fmt(cfg),
                             n_per_class=n_per_class, gen_seed=gen_seed, clf_seed=s,
                             accuracy=res['accuracy'], gap=trtr_mean - res['accuracy'],
                             fidelity=diag['fidelity'], diversity=diag['diversity'],
                             data_dir=gdir.name, result_json=str(TC.result_path(gdir, s))))

    # 4. CSV 집계
    csv_path = C.RESULTS_DIR / f'results_{stage}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'[sweep] 결과 저장: {csv_path} ({len(rows)} rows)')

    # 5. 그림 생성
    if not skip_plot:
        P.make_all(csv_path, stage)

    dt = time.time() - t0
    print(f'=== sweep [{stage}] 완료 — {dt/60:.1f}분 소요 ===')
    print(f'    TRTR 천장 = {trtr_mean:.4f} | 그림: {C.FIGS_DIR}')
    return csv_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stage', default='coarse',
                   help='결과 파일/그림 태그 (coarse|fine|final|자유 문자열)')
    p.add_argument('--cfgs', type=float, nargs='+', default=COARSE_CFGS,
                   help='cfg_scale(s) 그리드')
    p.add_argument('--n-per-class', type=int, default=2000,
                   help='클래스당 생성 수 (coarse=2000 → 2만 장, final=6000 → 6만 장)')
    p.add_argument('--real-n-per-class', type=int, default=None,
                   help='TRTR용 원본 클래스당 개수. 미지정=--n-per-class와 동일(크기 매칭), '
                        '0=train 60k 전체(6만장 실험용; MNIST는 불균형이라 6000/class 균등 추출 불가)')
    p.add_argument('--clf-seeds', type=int, nargs='+', default=[0],
                   help='분류기 seed 목록 (coarse=[0], final=[0,1,2])')
    p.add_argument('--gen-seed', type=int, default=C.GEN_SEED,
                   help='생성 seed (전 cfg 동일 고정)')
    p.add_argument('--sd-ckpt', default=str(C.SD_CKPT))
    p.add_argument('--batch-size', type=int, default=500, help='생성 배치 크기')
    p.add_argument('--force', action='store_true',
                   help='기존 데이터셋/결과 무시하고 전부 재실행')
    p.add_argument('--skip-plot', action='store_true')
    a = p.parse_args()
    run(a.stage, a.cfgs, a.n_per_class, a.clf_seeds, a.gen_seed,
        a.sd_ckpt, a.batch_size, a.force, a.skip_plot,
        real_n_per_class=a.real_n_per_class)


if __name__ == '__main__':
    main()
