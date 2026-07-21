#!/usr/bin/env python3
"""
STEP 4 — 오케스트레이션: (n, r, seed) 그리드 실행 → CSV.

각 (n, seed)마다:
  - r=0        : baseline  = n개 원본만으로 분류기 학습
  - r>0        : augmented = n개 원본 + s(=r*n)개 합성  (합성 풀에서 subsample)
  - r>0 & 가능 : oracle    = n개 원본 + s개 '추가 원본' (pool에서, n-서브셋과 disjoint)

모두 real val 조기종료 분류기(train_clf_es)로 학습하고 진짜 test 10k로 평가한다.
조건별 결과 json을 캐시하므로 재실행 시 이미 끝난 조건은 스킵(--force로 무시).

선행: splits.py → train_diffusion_ds.py → gen_synth.py 완료돼 있어야 함.

사용 (src/src 에서):
    python experiments/datasize/run_datasize.py                 # 전체 그리드
    python experiments/datasize/run_datasize.py --keys n1000 n5000
    python experiments/datasize/run_datasize.py --seeds 0       # 빠른 확인
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import torch

import ds_config as D
import config as C
import splits as S
import train_clf_es as TE

CSV_FIELDS = ['n_key', 'total', 'r', 's', 'kind', 'seed',
              'n_train', 'accuracy', 'val_acc', 'best_epoch']


def synth_subset(key, s_per_class, seed):
    """합성 풀(라벨 균등, 클래스 순서 저장)에서 클래스당 s_per_class장 무작위 추출."""
    imgs01, labels, meta = C.load_dataset(D.synth_dir(key))
    ppc = int(meta['n_per_class'])
    if s_per_class > ppc:
        raise SystemExit(f'[{key}] 합성 부족: 필요 {s_per_class}/class > 풀 {ppc}/class. '
                         f'gen_synth.py 의 --max-r 를 키우세요.')
    g = torch.Generator().manual_seed(2000 + seed)
    pi, pl = [], []
    for c in range(C.NUM_CLASSES):
        rel = torch.randperm(ppc, generator=g)[:s_per_class]
        sel = c * ppc + rel
        pi.append(imgs01[sel]); pl.append(labels[sel])
    return torch.cat(pi), torch.cat(pl)


def _cond_path(key, r, kind, seed):
    return D.DS_CLF_DIR / f'{key}_r{C.fmt(r)}_{kind}_seed{seed}.json'


def _run_cond(key, total, r, s, kind, seed, imgs, labels, force):
    cp = _cond_path(key, r, kind, seed)
    if cp.exists() and not force:
        res = json.loads(cp.read_text())
        print(f'[skip] {cp.name} → acc={res["accuracy"]:.4f}')
    else:
        t0 = time.time()
        res = TE.train_es(imgs, labels, seed, desc=f'{key}_r{C.fmt(r)}_{kind}')
        res.pop('confusion', None)   # CSV/요약엔 불필요
        res.update(dict(n_key=key, total=total, r=r, s=s, kind=kind, seed=seed,
                        n_train=int(len(labels))))
        cp.write_text(json.dumps(res, indent=2))
        print(f'[{kind}] {key} r={C.fmt(r)} seed={seed} '
              f'n_train={len(labels)} → test acc={res["accuracy"]:.4f} '
              f'(val={res["val_acc"]:.4f}, ep={res["best_epoch"]}, {time.time()-t0:.0f}s)')
    return {k: res[k] for k in CSV_FIELDS}


def run(keys, r_list, seeds, force=False):
    D.ensure_dirs()
    if not C.dataset_exists(D.POOL_DIR):
        raise SystemExit('splits 미생성. 먼저: python experiments/datasize/splits.py')

    rows = []
    for key in keys:
        spec = D.N_BY_KEY[key]
        total = spec['total']
        if not C.dataset_exists(D.subset_dir(key)):
            raise SystemExit(f'{key} 서브셋 없음 → splits.py 실행')
        if not C.dataset_exists(D.synth_dir(key)):
            raise SystemExit(f'{key} 합성 풀 없음 → gen_synth.py --key {key} 실행')
        base_i, base_l, _ = C.load_dataset(D.subset_dir(key))

        for seed in seeds:
            for r in r_list:
                if r == 0:
                    rows.append(_run_cond(key, total, 0.0, 0, 'baseline', seed,
                                          base_i, base_l, force))
                    continue
                s = D.s_count(total, r)
                spc = s // C.NUM_CLASSES
                # augmented
                si, sl = synth_subset(key, spc, seed)
                rows.append(_run_cond(key, total, r, s, 'aug', seed,
                                      torch.cat([base_i, si]), torch.cat([base_l, sl]), force))
                # oracle (가능할 때만; 가장 적은 클래스 기준으로 판정)
                if D.oracle_feasible(key, r):
                    oi, ol = S.oracle_real_subset(key, spc, seed)
                    rows.append(_run_cond(key, total, r, s, 'oracle', seed,
                                          torch.cat([base_i, oi]), torch.cat([base_l, ol]), force))
                else:
                    print(f'[oracle] {key} r={C.fmt(r)}: pool 부족으로 생략')

    out = D.DS_RESULTS / 'results_datasize.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'\n[run] 결과 저장: {out} ({len(rows)} rows)')
    print('[run] 요약: python experiments/datasize/aggregate.py')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--keys', nargs='+', default=[s['key'] for s in D.N_SPECS])
    p.add_argument('--r-list', type=float, nargs='+', default=D.R_LIST)
    p.add_argument('--seeds', type=int, nargs='+', default=D.SEEDS)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    run(a.keys, a.r_list, a.seeds, force=a.force)


if __name__ == '__main__':
    main()
