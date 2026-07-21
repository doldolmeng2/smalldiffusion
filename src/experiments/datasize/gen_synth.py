#!/usr/bin/env python3
"""
STEP 3 — n별 합성 데이터 풀 생성 (CFG_SCALE = -0.25 고정).

각 n의 확산모델로 라벨 균등 합성 풀을 미리 크게(=max(r)*n) 한 번 생성해 둔다.
이후 run_datasize.py 가 r별로 필요한 만큼 이 풀에서 부분추출(subsample)한다.
→ r을 바꿔도 재생성 불필요, 재현성 있음.

기존 generate.generate() 를 그대로 재사용한다(샘플러/step/스케줄 = config 고정).

사용 (src/src 에서):
    python experiments/datasize/gen_synth.py --key n1000
    python experiments/datasize/gen_synth.py --all
    # 가장 큰 n의 합성량을 줄이고 싶으면(메모리/시간) max-r 축소:
    python experiments/datasize/gen_synth.py --key n60000 --max-r 4
"""
import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from accelerate import Accelerator

import ds_config as D
import config as C
import generate as G


def gen_for(key, batch_size=500, max_r=None, force=False):
    D.ensure_dirs()
    spec = D.N_BY_KEY[key]
    per_class = D.synth_pool_per_class(spec['total'], max_r)   # 풀의 클래스당 개수
    out = D.synth_dir(key)
    if C.dataset_exists(out) and not force:
        print(f'[gen_synth] {key}: 이미 존재 → 스킵 ({out})')
        return out

    sd = D.diffusion_ckpt(key)
    if not Path(sd).exists():
        raise SystemExit(f'{key} 확산 체크포인트 없음: {sd}\n'
                         f'  먼저: python experiments/datasize/train_diffusion_ds.py --key {key}')

    accel = Accelerator()
    model = G.load_sd_model(sd, accel.device)
    print(f'[gen_synth] {key}: cfg={D.CFG_SCALE}, 클래스당 {per_class}장 '
          f'(총 {per_class * C.NUM_CLASSES}장) 생성 시작')
    G.generate(D.CFG_SCALE, per_class, seed=D.MASTER_SEED, out_dir=out,
               batch_size=batch_size, sd_ckpt=sd, model=model, accel=accel)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--key', default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--max-r', type=float, default=None,
                   help='풀 크기 = max-r * n (기본=max(R_LIST)=8)')
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()

    keys = [s['key'] for s in D.N_SPECS] if a.all else ([a.key] if a.key else None)
    if not keys:
        p.error('--key 또는 --all 을 지정하세요')
    for k in keys:
        gen_for(k, batch_size=a.batch_size, max_r=a.max_r, force=a.force)


if __name__ == '__main__':
    main()
