#!/usr/bin/env python3
"""
STEP 4/6/7 — 생성 데이터셋 만들기.

학습된 DiT(EMA 가중치)로 라벨 균등(숫자당 n_per_class장) 생성 데이터셋을 저장한다.
cfg_scale만 바꾸고, step 수/샘플러/스케줄/seed 정책은 config.py 상수로 고정.

사용 예 (src/src 에서):
    python experiments/generate.py --cfg-scale 4 --n-per-class 2000
"""
import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from smalldiffusion import ScheduleDDPM, samples, img_normalize, DiT, CondEmbedderLabel

import config as C


def build_dit():
    return DiT(**C.DIT_KW,
               cond_embed=CondEmbedderLabel(C.COND_DIM, C.NUM_CLASSES, C.COND_DROPOUT))


def load_sd_model(ckpt_path, device):
    """체크포인트를 로드하고 EMA 가중치를 모델에 적용(추론 전용)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'DiT 체크포인트가 없습니다: {ckpt_path}')
    model = build_dit()
    ckpt = C.torch_load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    ema = EMA(model.parameters(), decay=C.EMA_DECAY)
    ema.load_state_dict(ckpt['ema'])
    model.to(device)
    ema.to(device)
    ema.copy_to(model.parameters())   # 이후 model 파라미터 = EMA 가중치
    model.eval()
    return model


def generate(cfg_scale: float, n_per_class: int, seed: int = C.GEN_SEED,
             out_dir=None, batch_size: int = 500, sd_ckpt=C.SD_CKPT,
             model=None, accel=None) -> Path:
    """생성 데이터셋을 out_dir에 저장하고 경로 반환. model/accel 재사용 가능(스윕용)."""
    C.ensure_dirs()
    out_dir = Path(out_dir) if out_dir else C.gen_dir(cfg_scale, n_per_class, seed)

    accel = accel or Accelerator()
    if model is None:
        model = load_sd_model(sd_ckpt, accel.device)

    schedule = ScheduleDDPM(beta_start=C.BETA_START, beta_end=C.BETA_END, N=C.SCHEDULE_N)
    sigmas = schedule.sample_sigmas(C.SAMPLE_STEPS)

    # 라벨 균등: [0]*n, [1]*n, ... 순서로 생성 (분류기 학습 시 shuffle하므로 순서 무관)
    all_labels = torch.arange(C.NUM_CLASSES).repeat_interleave(n_per_class)

    C.set_seed(seed)
    chunks = []
    for i in tqdm(range(0, len(all_labels), batch_size),
                  desc=f'[generate] cfg={C.fmt(cfg_scale)} n/class={n_per_class}'):
        cond = all_labels[i:i + batch_size]
        *_, x0 = samples(model, sigmas, gam=C.GAM, mu=C.MU,
                         batchsize=len(cond), cond=cond,
                         cfg_scale=cfg_scale, accelerator=accel)
        chunks.append(img_normalize(x0).cpu())   # [-1,1] → [0,1]

    images = torch.cat(chunks, dim=0)
    meta = dict(kind='gen', cfg_scale=float(cfg_scale), n_per_class=n_per_class,
                total=len(all_labels), seed=seed, steps=C.SAMPLE_STEPS,
                gam=C.GAM, mu=C.MU, beta_start=C.BETA_START, beta_end=C.BETA_END,
                schedule_N=C.SCHEDULE_N, sd_ckpt=str(sd_ckpt))
    C.save_dataset(out_dir, images, all_labels, meta)
    print(f'[generate] 저장 완료: {out_dir} ({len(all_labels)}장)')
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cfg-scale', type=float, required=True)
    p.add_argument('--n-per-class', type=int, default=2000)
    p.add_argument('--seed', type=int, default=C.GEN_SEED)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--sd-ckpt', default=str(C.SD_CKPT))
    a = p.parse_args()
    generate(a.cfg_scale, a.n_per_class, a.seed, a.out_dir, a.batch_size, a.sd_ckpt)


if __name__ == '__main__':
    main()
