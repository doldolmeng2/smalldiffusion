#!/usr/bin/env python3
"""
분류기 학습·평가 (early stopping 버전) — 이 실험 전용.

기존 train_classifier.py 는 '고정 5 epoch'라 데이터 크기가 크게 달라지는 이 실험엔
부적절하다(작은 n은 덜 수렴, 큰 n은 과도). 그래서:
  - real val 5,000장(splits.py의 VAL)으로 매 epoch 검증
  - val 정확도가 CLF_PATIENCE epoch 동안 개선 없으면 중단, 최적 가중치 복원
  - 그 최적 모델을 진짜 test 10k(TC.evaluate)로 1회 평가

구조(MnistCNN)·정규화(normalize_for_clf)·test 로더는 기존 config/train_classifier 재사용.
baseline / augmented / oracle 전 조건에 이 함수를 동일하게 적용한다(단일 절차 원칙).
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import ds_config as D
import config as C
import train_classifier as TC   # get_test_tensors / evaluate 재사용


@torch.no_grad()
def _accuracy(model, x, y, device, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        pred = model(x[i:i + bs].to(device)).argmax(dim=1).cpu()
        correct += (pred == y[i:i + bs]).sum().item()
    return correct / len(y)


def train_es(train_imgs01, train_labels, seed, device=None, desc='clf'):
    """train_imgs01: float [0,1] (N,1,28,28). 반환: dict(accuracy, val_acc, best_epoch, ...)."""
    device = device or C.get_device()
    C.set_seed(seed)

    model = C.MnistCNN().to(device)
    opt = Adam(model.parameters(), lr=D.CLF_LR)

    ds = TensorDataset(C.normalize_for_clf(train_imgs01), train_labels)
    loader = DataLoader(ds, batch_size=D.CLF_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    # real 검증셋 (early stopping 전용)
    vx01, vy, _ = C.load_dataset(D.VAL_DIR)
    vx = C.normalize_for_clf(vx01)

    max_ep = D.CLF_MAX_EPOCH if D.EARLY_STOP else D.CLF_FIXED_EPOCHS
    best_acc, best_state, best_ep, since = -1.0, None, 0, 0

    for ep in range(max_ep):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.nll_loss(model(xb), yb)
            loss.backward()
            opt.step()

        if D.EARLY_STOP:
            va = _accuracy(model, vx, vy, device)
            if va > best_acc:
                best_acc = va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_ep = ep + 1
                since = 0
            else:
                since += 1
                if since >= D.CLF_PATIENCE:
                    break

    if D.EARLY_STOP and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = max_ep
        best_acc = _accuracy(model, vx, vy, device)

    top1, per_class, confusion = TC.evaluate(model, device)
    return dict(accuracy=top1, val_acc=best_acc, best_epoch=best_ep,
                per_class=per_class, confusion=confusion)
