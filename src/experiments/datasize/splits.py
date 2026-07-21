#!/usr/bin/env python3
"""
STEP 1 — 고정 데이터 분할 만들기 (한 번만 실행, 재현성 보장).

MNIST train 60,000장을 다음으로 쪼갠다 (master seed 고정):
  - val  : 5,000장 (클래스당 500) — 분류기 early stopping 전용. 학습/평가에 안 씀.
  - pool : 나머지 55,000장 — 모든 n-서브셋과 oracle용 추가 원본을 여기서 뽑음.
  - n별 서브셋: pool의 각 클래스 '앞쪽 npc개'. (확산모델 학습 = 분류기 baseline 원본,
    문자 그대로 같은 데이터를 쓰도록 디스크에 저장)

저장 포맷은 기존 config.save_dataset 규약(images.pt/labels.pt/meta.json)과 동일하다.
test(진짜 10k)는 절대 손대지 않는다 — 최종 평가 전용.

주의: n60000 은 val 예약 때문에 실제로 pool 전체(55,000장)다.

사용:
    python experiments/datasize/splits.py            # 전부 생성
    python experiments/datasize/splits.py --force    # 다시 생성
"""
import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent        # .../experiments/datasize
_SRC = _EXP.parents[1]                         # .../src/src
for _p in (str(_SRC), str(_SRC / 'experiments'), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torchvision.datasets import MNIST

import ds_config as D
import config as C


def _mnist_train():
    ds = MNIST(str(C.MNIST_ROOT), train=True, download=True)
    return ds.data.unsqueeze(1), ds.targets   # uint8 (60000,1,28,28), int64 (60000,)


def _save(out_dir, data_uint8, labels, meta):
    imgs01 = data_uint8.float() / 255.0        # [0,1]
    C.save_dataset(out_dir, imgs01, labels, meta)


def _pool_offsets(class_counts):
    """pool 배열에서 각 클래스 슬라이스 시작 오프셋 (len = num_classes+1)."""
    offs = [0]
    for c in class_counts:
        offs.append(offs[-1] + c)
    return offs


def build(force=False):
    D.ensure_dirs()
    if C.dataset_exists(D.POOL_DIR) and not force:
        print(f'[splits] 이미 존재 → 스킵 (다시 만들려면 --force): {D.SPLIT_DIR}')
        return

    data, targets = _mnist_train()
    g = torch.Generator().manual_seed(D.MASTER_SEED)
    val_per_class = D.VAL_SIZE // C.NUM_CLASSES     # 500

    val_idx, pool_by_class = [], []
    for c in range(C.NUM_CLASSES):
        idx = (targets == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=g)]
        val_idx.append(perm[:val_per_class])
        pool_by_class.append(perm[val_per_class:])   # 클래스별 pool 인덱스
    val_idx = torch.cat(val_idx)
    pool_idx = torch.cat(pool_by_class)              # 클래스 순서로 concat
    class_counts = [int(len(x)) for x in pool_by_class]

    # 1) val
    _save(D.VAL_DIR, data[val_idx], targets[val_idx],
          dict(kind='val', total=int(len(val_idx)), per_class=val_per_class,
               source='MNIST train (no flip)'))
    print(f'[splits] val: {len(val_idx)}장 → {D.VAL_DIR}')

    # 2) pool (클래스 순서 정렬 + 클래스별 개수 기록)
    _save(D.POOL_DIR, data[pool_idx], targets[pool_idx],
          dict(kind='pool', total=int(len(pool_idx)), class_counts=class_counts,
               source='MNIST train minus val (no flip, unbalanced)'))
    print(f'[splits] pool: {len(pool_idx)}장 (클래스별 {class_counts}) → {D.POOL_DIR}')

    # 3) n별 서브셋 (pool의 각 클래스 앞쪽 npc개; full은 pool 전체)
    offs = _pool_offsets(class_counts)
    for spec in D.N_SPECS:
        key, npc = spec['key'], spec['npc']
        sdir = D.subset_dir(key)
        if npc is None:
            _save(sdir, data[pool_idx], targets[pool_idx],
                  dict(kind='real_subset', key=key, total=int(len(pool_idx)),
                       npc=None, class_counts=class_counts))
            print(f'[splits] {key}: pool 전체 {len(pool_idx)}장 → {sdir}')
        else:
            sub = []
            for c in range(C.NUM_CLASSES):
                cslice = pool_idx[offs[c]:offs[c + 1]]
                if len(cslice) < npc:
                    raise ValueError(f'class {c}: pool {len(cslice)} < npc {npc}')
                sub.append(cslice[:npc])
            sub = torch.cat(sub)
            _save(sdir, data[sub], targets[sub],
                  dict(kind='real_subset', key=key, total=int(len(sub)), npc=npc))
            print(f'[splits] {key}: {len(sub)}장 (클래스당 {npc}) → {sdir}')
    print('[splits] 완료')


def oracle_real_subset(key, s_per_class, seed):
    """n-서브셋(각 클래스 앞쪽 npc개)과 겹치지 않는 '추가 원본' s_per_class/class 을 반환.

    pool은 클래스 순서로 저장돼 있고 n-서브셋은 각 클래스 앞쪽 npc개이므로,
    각 클래스의 [npc:] 구간(여분)에서만 뽑으면 n-서브셋과 확실히 disjoint.
    반환: (imgs01 float[0,1] (10*s_per_class,1,28,28), labels int64)
    """
    spec = D.N_BY_KEY[key]
    npc = spec['npc']
    if npc is None:
        raise ValueError('full pool(n60000)은 oracle용 여분 원본이 없다')
    imgs01, labels, meta = C.load_dataset(D.POOL_DIR)
    counts = meta['class_counts']
    offs = _pool_offsets(counts)
    g = torch.Generator().manual_seed(1000 + seed)
    pi, pl = [], []
    for c in range(C.NUM_CLASSES):
        start, end = offs[c] + npc, offs[c + 1]     # n-서브셋 이후 여분 구간
        avail = end - start
        if avail < s_per_class:
            raise ValueError(f'class {c}: oracle 여분 부족 avail={avail} < need={s_per_class}')
        rel = torch.randperm(avail, generator=g)[:s_per_class]
        sel = start + rel
        pi.append(imgs01[sel]); pl.append(labels[sel])
    return torch.cat(pi), torch.cat(pl)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    build(force=a.force)


if __name__ == '__main__':
    main()
