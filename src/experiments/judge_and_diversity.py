#!/usr/bin/env python3
"""
보조 지표 — 충실도(fidelity)·다양성(diversity) 진단.

- 심판(judge) 분류기: 원본 train 60,000장 전체로 1회 학습 후 동결(ckpts/judge.pt).
  TSTR 분류기와 역할 분리(설계서 4장 5번). 구조·옵티마이저는 동일, 에폭만 JUDGE_EPOCHS.
- 충실도 = 라벨 y로 생성한 이미지를 동결 심판이 y로 분류하는 비율 (cfg↑ → 상승 예상)
- 다양성 = intra-class 픽셀 분산 평균: 같은 라벨 생성물의 픽셀별 분산을 평균 (cfg↑ → 하락 예상)

사용 예 (src/src 에서):
    python experiments/judge_and_diversity.py --data-dir experiments/data/gen_cfg4_n2000_seed0
"""
import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torchvision.datasets import MNIST

import config as C
import train_classifier as TC

JUDGE_SEED = 12345   # TSTR/TRTR seed와 겹치지 않게 별도 고정


# ---------------------------------------------------------------------------
# 심판 분류기
# ---------------------------------------------------------------------------
def train_judge(force=False) -> Path:
    """원본 60k 전체로 심판 학습 → 동결 저장. 이미 있으면 스킵."""
    C.ensure_dirs()
    if C.JUDGE_CKPT.exists() and not force:
        return C.JUDGE_CKPT

    print(f'[judge] 심판 분류기 학습 (원본 60k, {C.JUDGE_EPOCHS} epochs)...')
    ds = MNIST(str(C.MNIST_ROOT), train=True, download=True)
    images01 = ds.data.unsqueeze(1).float() / 255.0
    device = C.get_device()
    model = TC.fit(images01, ds.targets, seed=JUDGE_SEED,
                   epochs=C.JUDGE_EPOCHS, device=device, desc='judge')
    top1, _, _ = TC.evaluate(model, device)
    torch.save(dict(model=model.state_dict(), test_acc=top1,
                    epochs=C.JUDGE_EPOCHS, seed=JUDGE_SEED), C.JUDGE_CKPT)
    print(f'[judge] 저장: {C.JUDGE_CKPT} (test top-1 = {top1:.4f})')
    return C.JUDGE_CKPT


def load_judge(device=None):
    device = device or C.get_device()
    if not C.JUDGE_CKPT.exists():
        train_judge()
    ckpt = C.torch_load(C.JUDGE_CKPT)
    model = C.MnistCNN().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 진단
# ---------------------------------------------------------------------------
@torch.no_grad()
def diagnose(data_dir, judge=None, force=False) -> dict:
    """data_dir의 (이미지,라벨)에 대해 fidelity/diversity 계산 → diag.json 저장."""
    data_dir = Path(data_dir)
    dpath = data_dir / 'diag.json'
    if dpath.exists() and not force:
        return json.loads(dpath.read_text())

    device = C.get_device()
    judge = judge or load_judge(device)
    images01, labels, meta = C.load_dataset(data_dir)

    # (a) 충실도: 심판이 각 이미지를 그 라벨로 맞히는 비율
    correct = 0
    for i in range(0, len(labels), C.CLF_TEST_BATCH):
        xb = C.normalize_for_clf(images01[i:i + C.CLF_TEST_BATCH]).to(device)
        pred = judge(xb).argmax(dim=1).cpu()
        correct += (pred == labels[i:i + C.CLF_TEST_BATCH]).sum().item()
    fidelity = correct / len(labels)

    # (b) 다양성: 클래스별 픽셀 분산의 평균 (같은 라벨 내 생성물의 퍼짐 정도)
    per_class_var = []
    for c in range(C.NUM_CLASSES):
        xc = images01[labels == c]              # (Nc, 1, 28, 28), [0,1]
        per_class_var.append(xc.var(dim=0, unbiased=False).mean().item())
    diversity = sum(per_class_var) / len(per_class_var)

    diag = dict(data_dir=str(data_dir), dataset=data_dir.name,
                kind=meta.get('kind'), cfg_scale=meta.get('cfg_scale'),
                fidelity=fidelity, diversity=diversity,
                per_class_diversity=per_class_var)
    dpath.write_text(json.dumps(diag, indent=2))
    print(f'[diag] {data_dir.name}: fidelity={fidelity:.4f}, diversity={diversity:.5f}')
    return diag


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--retrain-judge', action='store_true')
    a = p.parse_args()
    if a.retrain_judge:
        train_judge(force=True)
    diagnose(a.data_dir, force=a.force)


if __name__ == '__main__':
    main()
