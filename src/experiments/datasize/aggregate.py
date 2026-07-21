#!/usr/bin/env python3
"""
STEP 5 — seed 집계. results_datasize.csv → seed 평균±표준편차 요약 CSV.

(그림은 결과 확정 후 따로 그린다 — 여기선 숫자 요약만.)

출력: results/datasize/summary_datasize.csv
  컬럼: n_key, total, r, kind, n_seeds, acc_mean, acc_std
  + 각 (n, r) 에서 aug - baseline(증강 이득), oracle - aug(합성 vs 실제 격차) 파생 컬럼.

사용 (src/src 에서):
    python experiments/datasize/aggregate.py
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_EXP = Path(__file__).resolve().parent        # .../experiments/datasize
_SRC = _EXP.parents[1]                         # .../src/src
for _p in (str(_SRC), str(_SRC / 'experiments'), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ds_config as D


def _mean_std(xs):
    m = statistics.mean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, sd


def aggregate(csv_path=None):
    csv_path = Path(csv_path) if csv_path else (D.DS_RESULTS / 'results_datasize.csv')
    if not csv_path.exists():
        raise SystemExit(f'결과 CSV 없음: {csv_path}\n  먼저 run_datasize.py 실행')

    rows = list(csv.DictReader(open(csv_path)))
    # (n_key, total, r, kind) -> [acc...]
    acc = defaultdict(list)
    for r in rows:
        acc[(r['n_key'], r['total'], r['r'], r['kind'])].append(float(r['accuracy']))

    # 요약 + 파생지표(증강 이득, 합성-실제 격차) 계산용 조회 테이블
    stat = {k: _mean_std(v) for k, v in acc.items()}

    out = D.DS_RESULTS / 'summary_datasize.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_key', 'total', 'r', 'kind', 'n_seeds',
                    'acc_mean', 'acc_std', 'gain_vs_baseline', 'oracle_minus_aug'])
        for (nk, total, r, kind), accs in sorted(
                acc.items(), key=lambda kv: (kv[0][0], float(kv[0][2]), kv[0][3])):
            m, sd = stat[(nk, total, r, kind)]
            base = stat.get((nk, total, '0.0', 'baseline'))
            gain = (m - base[0]) if (base and kind in ('aug', 'oracle')) else ''
            oa = ''
            if kind == 'oracle':
                a = stat.get((nk, total, r, 'aug'))
                if a:
                    oa = m - a[0]   # oracle - aug : 같은 총량에서 '실제 - 합성' 품질 격차(+면 실제 우세)
            w.writerow([nk, total, r, kind, len(accs),
                        f'{m:.4f}', f'{sd:.4f}',
                        (f'{gain:.4f}' if gain != '' else ''),
                        (f'{oa:.4f}' if oa != '' else '')])
    print(f'[aggregate] 저장: {out}')
    # 콘솔 요약
    print('\n n_key   r    kind      acc(mean±std)   gain_vs_base')
    for (nk, total, r, kind), accs in sorted(
            acc.items(), key=lambda kv: (kv[0][0], float(kv[0][2]), kv[0][3])):
        m, sd = stat[(nk, total, r, kind)]
        base = stat.get((nk, total, '0.0', 'baseline'))
        g = f'{m-base[0]:+.4f}' if (base and kind != 'baseline') else '   -   '
        print(f' {nk:7s} {r:>4s} {kind:9s} {m:.4f}±{sd:.4f}   {g}')
    return out


if __name__ == '__main__':
    aggregate()
