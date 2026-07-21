#!/usr/bin/env python3
"""
STEP 2 — n별 조건부 DiT 확산모델 학습.

각 n(1k/5k/10k)에 대해 '그 n개 원본 서브셋'(splits.py가 만든 것)으로 DiT를 학습한다.
→ 합성 데이터가 '정확히 그 n개에서 배운' 것이 되도록 보장한다(분류기 baseline과 동일 데이터).
n60000 은 이미 학습된 full-60k 체크포인트(config.SD_CKPT)를 재사용하므로 학습하지 않는다.

공정성: 모든 n을 '동일 총 gradient step 수(--steps)'로 학습한다(에폭이 아니라 step 기준).
데이터가 적은 n은 epoch가 많아지지만 총 업데이트 수는 같아진다.

체크포인트 포맷은 generate.load_sd_model 이 로드하는 {'model','ema'} 규약을 따른다.

사용 (src/src 에서):
    python experiments/datasize/train_diffusion_ds.py --key n1000  --steps 40000
    python experiments/datasize/train_diffusion_ds.py --all        --steps 40000
"""
import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import math

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from torch_ema import ExponentialMovingAverage as EMA

from smalldiffusion import ScheduleDDPM, training_loop

import ds_config as D
import config as C
import generate as G   # build_dit() 재사용 → load_sd_model 과 구조 일치 보장


def train_one(key, steps=40000, lr=1e-3, batch_size=256, num_workers=2):
    if key == 'n60000':
        print(f'[train_diff] {key}: full-60k 체크포인트({C.SD_CKPT}) 재사용 → 학습 생략')
        return
    D.ensure_dirs()
    sdir = D.subset_dir(key)
    if not C.dataset_exists(sdir):
        raise SystemExit(f'서브셋 없음: {sdir}\n  먼저: python experiments/datasize/splits.py')

    imgs01, labels, _ = C.load_dataset(sdir)
    ds = TensorDataset(imgs01 * 2 - 1, labels)          # [0,1]→[-1,1], (x, label) 유지
    accel = Accelerator()
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)),
                        shuffle=True, num_workers=num_workers, drop_last=False)

    schedule = ScheduleDDPM(beta_start=C.BETA_START, beta_end=C.BETA_END, N=C.SCHEDULE_N)
    model = G.build_dit()
    ema = EMA(model.parameters(), decay=C.EMA_DECAY)
    ema.to(accel.device)

    # step 기준 학습: 목표 step 을 채우도록 epoch 수를 환산하고, 도달 시 중단.
    steps_per_epoch = math.ceil(len(ds) / min(batch_size, len(ds)))
    epochs = steps // steps_per_epoch + 2
    step = 0
    for ns in training_loop(loader, model, schedule, epochs=epochs,
                            lr=lr, accelerator=accel, conditional=True):
        ema.update()
        step += 1
        ns.pbar.set_description(f'[{key}] step {step}/{steps} Loss={ns.loss.item():.5f}')
        if step >= steps:
            break

    ckpt = D.diffusion_ckpt(key)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'ema': ema.state_dict(),
                'steps': step, 'key': key}, ckpt)
    print(f'[train_diff] {key}: {step} step 학습 완료 → {ckpt}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--key', default=None, help='n1000 | n5000 | n10000')
    p.add_argument('--all', action='store_true', help='학습 대상 n 전체(n60000 제외) 순차 학습')
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=256)
    a = p.parse_args()

    if a.all:
        for spec in D.N_SPECS:
            if spec['key'] != 'n60000':
                train_one(spec['key'], a.steps, a.lr, a.batch_size)
    elif a.key:
        train_one(a.key, a.steps, a.lr, a.batch_size)
    else:
        p.error('--key 또는 --all 을 지정하세요')


if __name__ == '__main__':
    main()
