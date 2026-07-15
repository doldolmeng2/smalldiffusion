#!/usr/bin/env python3
"""
STEP 3/5/8/9 — 분류기 학습·평가 (TRTR/TSTR 공용).

config.py의 CNN·하이퍼파라미터를 그대로 쓰고 seed만 바꾼다(단일 변수 원칙).
data_dir(생성 또는 원본)로 학습 → 진짜 MNIST test 10,000장으로 평가 →
results/clf/{dataset}_seed{K}.json 에 top-1/per-class/confusion 저장.

사용 예 (src/src 에서):
    python experiments/train_classifier.py --data-dir experiments/data/gen_cfg4_n2000_seed0 --seed 0
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
import torch.nn.functional as F
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from tqdm import tqdm

import config as C


# ---------------------------------------------------------------------------
# 진짜 test 10k (평가 전용 — 어떤 학습에도 사용 금지)
# ---------------------------------------------------------------------------
def get_test_tensors():
    ds = MNIST(str(C.MNIST_ROOT), train=False, download=True)
    x = C.normalize_for_clf(ds.data.unsqueeze(1).float() / 255.0)
    return x, ds.targets


# ---------------------------------------------------------------------------
# 학습/평가 코어 (judge 학습에서도 재사용)
# ---------------------------------------------------------------------------
def fit(images01, labels, seed, epochs=C.CLF_EPOCHS, device=None, desc='clf'):
    """images01: float [0,1] (N,1,28,28). 반환: 학습된 MnistCNN."""
    device = device or C.get_device()
    C.set_seed(seed)
    model = C.MnistCNN().to(device)
    opt = Adadelta(model.parameters(), lr=C.CLF_LR)
    sched = StepLR(opt, step_size=1, gamma=C.CLF_LR_GAMMA)

    ds = TensorDataset(C.normalize_for_clf(images01), labels)
    loader = DataLoader(ds, batch_size=C.CLF_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f'[{desc}] seed={seed} epoch {ep+1}/{epochs}', leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.nll_loss(model(xb), yb)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        sched.step()
    return model


@torch.no_grad()
def evaluate(model, device=None):
    """진짜 test 10k 평가 → (top1, per_class 리스트, confusion 10x10 리스트)."""
    device = device or next(model.parameters()).device
    x, y = get_test_tensors()
    model.eval()
    confusion = torch.zeros(C.NUM_CLASSES, C.NUM_CLASSES, dtype=torch.long)
    correct = 0
    for i in range(0, len(y), C.CLF_TEST_BATCH):
        xb = x[i:i + C.CLF_TEST_BATCH].to(device)
        yb = y[i:i + C.CLF_TEST_BATCH]
        pred = model(xb).argmax(dim=1).cpu()
        correct += (pred == yb).sum().item()
        for t, p in zip(yb.tolist(), pred.tolist()):
            confusion[t, p] += 1
    top1 = correct / len(y)
    per_class = (confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()).tolist()
    return top1, per_class, confusion.tolist()


# ---------------------------------------------------------------------------
# 단일 (data_dir, seed) 실행
# ---------------------------------------------------------------------------
def result_path(data_dir, seed) -> Path:
    return C.CLF_DIR / f'{Path(data_dir).name}_seed{seed}.json'


def train_and_eval(data_dir, seed, save_ckpt=True, force=False) -> dict:
    """이미 result.json이 있으면(force=False) 재학습 없이 그대로 반환."""
    C.ensure_dirs()
    data_dir = Path(data_dir)
    rpath = result_path(data_dir, seed)
    if rpath.exists() and not force:
        print(f'[clf] 기존 결과 재사용: {rpath}')
        return json.loads(rpath.read_text())

    device = C.get_device()
    images01, labels, meta = C.load_dataset(data_dir)
    model = fit(images01, labels, seed, device=device, desc=data_dir.name)
    top1, per_class, confusion = evaluate(model, device)

    result = dict(data_dir=str(data_dir), dataset=data_dir.name, seed=seed,
                  kind=meta.get('kind'), cfg_scale=meta.get('cfg_scale'),
                  n_per_class=meta.get('n_per_class'), gen_seed=meta.get('seed'),
                  epochs=C.CLF_EPOCHS, accuracy=top1,
                  per_class=per_class, confusion=confusion)
    rpath.write_text(json.dumps(result, indent=2))
    if save_ckpt:
        torch.save(model.state_dict(), C.CKPT_DIR / f'clf_{data_dir.name}_seed{seed}.pt')
    print(f'[clf] {data_dir.name} seed={seed} → test top-1 = {top1:.4f}  ({rpath})')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force', action='store_true', help='기존 result.json 무시하고 재학습')
    p.add_argument('--no-save-ckpt', action='store_true')
    a = p.parse_args()
    train_and_eval(a.data_dir, a.seed, save_ckpt=not a.no_save_ckpt, force=a.force)


if __name__ == '__main__':
    main()
