#!/usr/bin/env python3
"""
"합성이 정말 빈 자리를 채우나?"를 눈으로 확인하는 2D 임베딩 시각화.

같은 (n) 확산모델이 만든 합성 점군을, baseline 진짜 점군과 '같은 2D 좌표계'에
겹쳐 찍는다. t-SNE는 진짜+합성(+oracle)을 union으로 한 번에 학습하므로 좌표가 공유돼
"합성 점이 진짜 클러스터 위에 얹히는가(=커버), 중심으로 뭉치는가(=mode collapse),
바깥/사이로 새는가(=off-manifold)"를 직접 볼 수 있다.

특징(feature) 공간 2가지:
  --feat judge : 심판 CNN의 penultimate 128차원 (의미공간, 권장)
  --feat pixel : 원본 784픽셀 (모델 무관, 심판 없어도 됨)
그 다음 PCA(50) → t-SNE(2).

사용 (src/src 에서, splits.py + gen_synth.py 선행 필요):
    python experiments/datasize/embed_baseline_vs_synth.py --key n1000 --r 8
    python experiments/datasize/embed_baseline_vs_synth.py --key n1000 --r 8 --include-oracle
    python experiments/datasize/embed_baseline_vs_synth.py --key n5000 --r 8 --feat pixel

출력: results/figs/embed_<key>_r<r>_<feat>.png
"""
import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent        # .../experiments/datasize
_SRC = _EXP.parents[1]                         # .../src/src
for _p in (str(_SRC), str(_SRC / 'experiments'), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config as C
import ds_config as D
import splits


# ---------------------------------------------------------------------------
# 특징 추출
# ---------------------------------------------------------------------------
@torch.no_grad()
def judge_features(imgs01, device, batch=1000):
    """심판 CNN의 fc1 relu 직후 128차원 벡터. (dropout은 eval에서 항등)"""
    from judge_and_diversity import load_judge
    m = load_judge(device)
    feats = []
    for i in range(0, len(imgs01), batch):
        x = C.normalize_for_clf(imgs01[i:i + batch]).to(device)
        x = F.relu(m.conv1(x)); x = F.relu(m.conv2(x)); x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1); x = F.relu(m.fc1(x))
        feats.append(x.cpu())
    return torch.cat(feats).numpy()


def pixel_features(imgs01):
    return imgs01.reshape(len(imgs01), -1).numpy()


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
def balanced_take(imgs, labels, per_class, seed=0):
    """클래스별 per_class개 균등 추출."""
    g = torch.Generator().manual_seed(seed)
    out_i, out_l = [], []
    for c in range(C.NUM_CLASSES):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if len(idx) > per_class:
            idx = idx[torch.randperm(len(idx), generator=g)[:per_class]]
        out_i.append(imgs[idx]); out_l.append(labels[idx])
    return torch.cat(out_i), torch.cat(out_l)


def load_groups(key, r, include_oracle, plot_per_class):
    npc = D.N_BY_KEY[key]['npc']            # baseline 클래스당 개수 (n60000은 None)
    real_i, real_l, _ = C.load_dataset(D.subset_dir(key))
    synth_i, synth_l, _ = C.load_dataset(D.synth_dir(key))

    # r*n 만큼 합성 부분추출 (클래스 균등). npc None(=full)이면 진짜 클래스당 5500 기준.
    npc_eff = npc if npc is not None else (len(real_l) // C.NUM_CLASSES)
    s_per_class = int(round(r * npc_eff))
    synth_i, synth_l = balanced_take(synth_i, synth_l, s_per_class, seed=0)

    groups = {'real (baseline)': (real_i, real_l),
              f'synth (r={r:g})': (synth_i, synth_l)}

    if include_oracle and npc is not None:
        orc_i, orc_l = splits.oracle_real_subset(key, s_per_class, seed=0)
        groups['oracle (real +)'] = (orc_i, orc_l)

    # 플롯용 다운샘플 (t-SNE 속도)
    for k in groups:
        gi, gl = groups[k]
        groups[k] = balanced_take(gi, gl, plot_per_class, seed=1)
    return groups


# ---------------------------------------------------------------------------
# 임베딩 + 플롯
# ---------------------------------------------------------------------------
def embed(groups, feat, device):
    imgs = torch.cat([v[0] for v in groups.values()])
    X = judge_features(imgs, device) if feat == 'judge' else pixel_features(imgs)
    X = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
    Z = TSNE(n_components=2, init='pca', perplexity=30,
             random_state=0).fit_transform(X)     # union을 한 번에 → 좌표 공유
    # 그룹별로 다시 슬라이스
    out, off = {}, 0
    for k, (gi, gl) in groups.items():
        n = len(gi)
        out[k] = (Z[off:off + n], gl.numpy())
        off += n
    return out


def plot(emb, key, r, feat, out_path):
    names = list(emb.keys())
    real_name = names[0]
    overlays = names[1:]                     # synth (, oracle)
    n_panels = len(overlays)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.6),
                             squeeze=False)
    cmap = plt.get_cmap('tab10')
    Zr, yr = emb[real_name]

    for ax, ov in zip(axes[0], overlays):
        Zo, yo = emb[ov]
        # 진짜 = 채운 점(연하게), 오버레이 = 테두리만(진하게) — 겹침을 봐야 하므로
        ax.scatter(Zr[:, 0], Zr[:, 1], s=14, c=[cmap(c) for c in yr],
                   alpha=0.35, linewidths=0, label=f'{real_name}')
        ax.scatter(Zo[:, 0], Zo[:, 1], s=22, facecolors='none',
                   edgecolors=[cmap(c) for c in yo], linewidths=0.9,
                   label=ov)
        ax.set_title(f'{real_name}  +  {ov}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    handles = [plt.Line2D([0], [0], marker='o', ls='', mfc=cmap(c), mec=cmap(c),
                          ms=6, label=str(c)) for c in range(C.NUM_CLASSES)]
    fig.legend(handles=handles, title='digit', loc='center left',
               bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f'{key}  |  feat={feat}  |  채움=진짜, 테두리=합성/oracle',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.97, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print('saved:', out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--key', default='n1000')
    p.add_argument('--r', type=float, default=8.0)
    p.add_argument('--feat', choices=['judge', 'pixel'], default='judge')
    p.add_argument('--include-oracle', action='store_true')
    p.add_argument('--plot-per-class', type=int, default=150,
                   help='그룹별 클래스당 플롯 점 수 (t-SNE 속도용)')
    a = p.parse_args()

    device = C.get_device()
    groups = load_groups(a.key, a.r, a.include_oracle, a.plot_per_class)
    for k, (gi, _) in groups.items():
        print(f'  {k}: {len(gi)}장')
    emb = embed(groups, a.feat, device)
    out = D.DS_RESULTS / 'figs' / f'embed_{a.key}_r{a.r:g}_{a.feat}.png'
    plot(emb, a.key, a.r, a.feat, out)


if __name__ == '__main__':
    main()
