#!/usr/bin/env python3
"""
그림 생성 — results_{stage}.csv 로부터 발표용 png 세트 생성.

  (1) tstr_vs_cfg_{stage}.png        : 헤드라인 곡선 (TSTR 평균±std 밴드 + TRTR 천장선)
  (2) gap_{stage}.png                : 정확도 갭 (TRTR − TSTR)
  (3) fidelity_diversity_{stage}.png : 충실도·다양성 이중축 곡선
  (4) confusion_{stage}_cfg{S}.png   : 대표 cfg(최소/최적/최대) confusion matrix

축 라벨은 오해 방지를 위해 'cfg_scale (s)' 명시 (논문 w = s + 1).

사용 예 (src/src 에서):
    python experiments/plot.py --csv experiments/results/results_coarse.csv --stage coarse
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as C

# 그림 안 텍스트는 서버에 한글 폰트가 없어도 깨지지 않게 영문으로 통일
XLABEL = 'cfg_scale (s)   [paper notation: w = s + 1]'


def load_rows(csv_path):
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def split_rows(rows):
    """→ (trtr_accs, tstr: {cfg: {'accs':[], 'fid':f, 'div':d, 'jsons':[]}})"""
    trtr_accs = [float(r['accuracy']) for r in rows if r['kind'] == 'TRTR']
    tstr = defaultdict(lambda: dict(accs=[], fid=None, div=None, jsons=[]))
    for r in rows:
        if r['kind'] != 'TSTR':
            continue
        d = tstr[float(r['cfg_scale'])]
        d['accs'].append(float(r['accuracy']))
        d['fid'] = float(r['fidelity'])
        d['div'] = float(r['diversity'])
        d['jsons'].append(r['result_json'])
    return trtr_accs, dict(sorted(tstr.items()))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'[plot] 저장: {path}')


def plot_tstr(trtr_accs, tstr, stage):
    cfgs = np.array(list(tstr.keys()))
    means = np.array([np.mean(v['accs']) for v in tstr.values()])
    stds = np.array([np.std(v['accs']) for v in tstr.values()])
    ceil_m, ceil_s = np.mean(trtr_accs), np.std(trtr_accs)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cfgs, means, 'o-', color='tab:blue', label='TSTR (Train Synthetic, Test Real)')
    ax.fill_between(cfgs, means - stds, means + stds, color='tab:blue', alpha=0.2,
                    label='±1 std' if len(trtr_accs) > 1 or stds.any() else None)
    ax.axhline(ceil_m, ls='--', color='tab:red', label=f'TRTR ceiling = {ceil_m:.4f}')
    if ceil_s > 0:
        ax.axhspan(ceil_m - ceil_s, ceil_m + ceil_s, color='tab:red', alpha=0.1)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel('Test accuracy (real 10k)')
    ax.set_title(f'TSTR accuracy vs CFG scale  [{stage}]')
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, C.FIGS_DIR / f'tstr_vs_cfg_{stage}.png')


def plot_gap(trtr_accs, tstr, stage):
    cfgs = np.array(list(tstr.keys()))
    ceil = np.mean(trtr_accs)
    gaps = np.array([ceil - np.mean(v['accs']) for v in tstr.values()])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cfgs, gaps, 's-', color='tab:purple')
    ax.axhline(0, ls=':', color='gray')
    ax.set_xlabel(XLABEL)
    ax.set_ylabel('Accuracy gap  (TRTR − TSTR)')
    ax.set_title(f'Synthetic data deficit — lower is better  [{stage}]')
    ax.grid(alpha=0.3)
    _save(fig, C.FIGS_DIR / f'gap_{stage}.png')


def plot_fid_div(tstr, stage):
    cfgs = np.array(list(tstr.keys()))
    fid = np.array([v['fid'] for v in tstr.values()], dtype=float)
    div = np.array([v['div'] for v in tstr.values()], dtype=float)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    l1, = ax1.plot(cfgs, fid, 'o-', color='tab:green', label='Fidelity (judge accuracy)')
    ax1.set_xlabel(XLABEL)
    ax1.set_ylabel('Fidelity', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax2 = ax1.twinx()
    l2, = ax2.plot(cfgs, div, '^--', color='tab:orange',
                   label='Diversity (intra-class pixel var)')
    ax2.set_ylabel('Diversity', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax1.set_title(f'Fidelity vs Diversity trade-off  [{stage}]')
    ax1.legend(handles=[l1, l2], loc='center right')
    ax1.grid(alpha=0.3)
    _save(fig, C.FIGS_DIR / f'fidelity_diversity_{stage}.png')


def plot_confusions(tstr, stage):
    """대표 cfg 3개(최소 / TSTR 최고 / 최대)의 confusion matrix."""
    if not tstr:
        return
    cfgs = list(tstr.keys())
    best = max(cfgs, key=lambda c: np.mean(tstr[c]['accs']))
    for cfg in sorted({cfgs[0], best, cfgs[-1]}):
        jpath = Path(tstr[cfg]['jsons'][0])   # 첫 seed 결과 사용
        if not jpath.exists():
            continue
        res = json.loads(jpath.read_text())
        cm = np.array(res['confusion'], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] > 0:
                    ax.text(j, i, int(cm[i, j]), ha='center', va='center', fontsize=6,
                            color='white' if cm_norm[i, j] > 0.5 else 'black')
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        acc = res['accuracy']
        ax.set_title(f'Confusion (TSTR, cfg={C.fmt(cfg)}, acc={acc:.4f})')
        fig.colorbar(im, fraction=0.046)
        _save(fig, C.FIGS_DIR / f'confusion_{stage}_cfg{C.fmt(cfg)}.png')


def make_all(csv_path, stage):
    C.ensure_dirs()
    rows = load_rows(csv_path)
    trtr_accs, tstr = split_rows(rows)
    if not trtr_accs or not tstr:
        print('[plot] TRTR/TSTR 행이 부족해 그림을 생략합니다.')
        return
    plot_tstr(trtr_accs, tstr, stage)
    plot_gap(trtr_accs, tstr, stage)
    plot_fid_div(tstr, stage)
    plot_confusions(tstr, stage)
    print(f'[plot] 그림 세트 완료: {C.FIGS_DIR}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv', required=True)
    p.add_argument('--stage', default='coarse')
    a = p.parse_args()
    make_all(a.csv, a.stage)


if __name__ == '__main__':
    main()
