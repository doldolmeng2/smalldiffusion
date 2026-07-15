# CFG–TSTR 실험 (자료/실험설계_CFG_TSTR.md 구현)

한 번의 명령으로 **cfg별 데이터 생성 → 분류기 학습(TSTR·TRTR) → 진단 → results.csv → 그래프**까지 자동 실행된다.

## 빠른 시작 (src/src 에서 실행)

```bash
# 1차 coarse: s ∈ {0,1,2,3,4,6,8} × 2만 장 × 1 seed  → 곡선 개형 파악
python experiments/run_sweep.py --stage coarse

# 2차 fine: coarse 결과 보고 관심 구간 cfg만 추가 지정
python experiments/run_sweep.py --stage fine --cfgs 0 0.25 0.5 0.75 1

# 최종: 확정 cfg × 6만 장 × 3 seed → 저분산 발표 곡선
python experiments/run_sweep.py --stage final --cfgs 0 1 2 3 4 --n-per-class 6000 --clf-seeds 0 1 2
```

- 이미 만든 데이터셋/분류기 결과는 **자동 스킵**되므로 중단 후 재실행해도 이어서 진행된다. 전부 다시 하려면 `--force`.
- DiT 체크포인트 기본 경로: `src/checkpoints/mnist_dit_e900.pth` (다르면 `--sd-ckpt`로 지정).

## 파일 구성

| 파일 | 책임 (설계서 9장) |
|---|---|
| `config.py` | 공통 상수(단일 변수 원칙 강제): CNN 구조·학습 설정, 샘플링(step=20, gam=1.6), 경로, 데이터 포맷 |
| `generate.py` | cfg별 라벨 균등 생성 데이터셋 (STEP 4/6/7) |
| `make_real_subset.py` | 원본 라벨 균등 부분집합 — TRTR 천장용, 생성과 동일 포맷 (STEP 3/9) |
| `train_classifier.py` | 분류기 학습·평가(TRTR/TSTR 공용), result.json (STEP 3/5/8/9) |
| `judge_and_diversity.py` | 동결 심판 학습 + 충실도·다양성 진단, diag.json |
| `run_sweep.py` | 전 과정 오케스트레이션 → results.csv → plot 호출 |
| `plot.py` | 헤드라인 곡선/갭/충실도·다양성/confusion png 세트 |

## 산출물

```
experiments/
├── data/    real_n{N}_seed{K}/, gen_cfg{S}_n{N}_seed{K}/  (images.pt + labels.pt + meta.json + diag.json)
├── ckpts/   judge.pt, clf_{dataset}_seed{K}.pt
├── results/
│   ├── results_{stage}.csv     # stage,kind,cfg_scale,…,accuracy,gap,fidelity,diversity
│   ├── clf/*.json              # 분류기별 top-1/per-class/confusion
│   └── figs/*.png              # 발표용 그림
└── logs/
```

## 개별 스크립트 단독 실행

```bash
python experiments/generate.py --cfg-scale 4 --n-per-class 2000
python experiments/make_real_subset.py --n-per-class 2000
python experiments/train_classifier.py --data-dir experiments/data/gen_cfg4_n2000_seed0 --seed 0
python experiments/judge_and_diversity.py --data-dir experiments/data/gen_cfg4_n2000_seed0
python experiments/plot.py --csv experiments/results/results_coarse.csv --stage coarse
```

## 설계서의 함정 체크 반영 사항

- `RandomHorizontalFlip` 미사용 — 원본은 `MNIST.data`에서 직접 추출, 분류기는 [0,1]→(0.1307, 0.3081) 정규화만.
- `cfg=0` = 순수 조건부(무조건 아님). `cond=None`은 TSTR 축에서 제외.
- TRTR·TSTR 학습 크기 매칭(같은 `--n-per-class`), CNN 구조/에폭/옵티마이저 전 구간 동일(seed만 변경).
- 심판 분류기는 원본 60k로 1회 학습 후 동결(`ckpts/judge.pt`), TSTR 분류기와 분리.
- 생성은 EMA 가중치 사용, 생성 seed는 전 cfg 동일 고정.
- test 10k는 평가 전용(어떤 학습에도 미사용).

추가 필요 패키지: `matplotlib` (`pip install matplotlib`)
