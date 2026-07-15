#!/usr/bin/env python3
"""
STEP 3/9 — 원본 MNIST 라벨 균등 부분집합 만들기 (TRTR 천장선용).

generate.py와 완전히 동일한 포맷(images.pt/labels.pt/meta.json)으로 저장해서
TRTR·TSTR을 같은 로더(train_classifier.py)로 학습한다.
크기 매칭 원칙: 생성 2만과 비교할 천장은 원본 2만, 생성 6만은 원본 6만.

사용 예 (src/src 에서):
    python experiments/make_real_subset.py --n-per-class 2000
"""
import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torchvision.datasets import MNIST

import config as C


def make_real_subset(n_per_class: int, seed: int = 0, out_dir=None) -> Path:
    C.ensure_dirs()
    out_dir = Path(out_dir) if out_dir else C.real_dir(n_per_class, seed)

    ds = MNIST(str(C.MNIST_ROOT), train=True, download=True)
    data = ds.data.unsqueeze(1)          # (60000, 1, 28, 28) uint8, flip 없음
    targets = ds.targets                 # (60000,) int64

    g = torch.Generator().manual_seed(seed)
    img_chunks, lbl_chunks = [], []
    for c in range(C.NUM_CLASSES):
        idx = (targets == c).nonzero(as_tuple=True)[0]
        if len(idx) < n_per_class:
            raise ValueError(f'클래스 {c}의 원본 개수({len(idx)}) < n_per_class({n_per_class})')
        pick = idx[torch.randperm(len(idx), generator=g)[:n_per_class]]
        img_chunks.append(data[pick])
        lbl_chunks.append(targets[pick])

    images01 = torch.cat(img_chunks).float() / 255.0
    labels = torch.cat(lbl_chunks)
    meta = dict(kind='real', cfg_scale=None, n_per_class=n_per_class,
                total=len(labels), seed=seed, source='MNIST train (no flip)')
    C.save_dataset(out_dir, images01, labels, meta)
    print(f'[real_subset] 저장 완료: {out_dir} ({len(labels)}장)')
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n-per-class', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=None)
    a = p.parse_args()
    make_real_subset(a.n_per_class, a.seed, a.out_dir)


if __name__ == '__main__':
    main()
