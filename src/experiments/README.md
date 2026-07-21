# CFG–TSTR 실험 (자료/실험설계_CFG_TSTR.md 구현)

한 번의 명령으로 **cfg별 데이터 생성 → 분류기 학습(TSTR·TRTR) → 진단 → results.csv → 그래프**까지 자동 실행된다.

## 빠른 시작 (src/src 에서 실행)

```bash
# 1차 coarse: s ∈ {0,1,2,3,4,6,8} × 2만 장 × 1 seed  → 곡선 개형 파악
python experiments/run_sweep.py --stage coarse

# 2차 fine: coarse 결과 보고 관심 구간 cfg만 추가 지정
python experiments/run_sweep.py --stage fine --cfgs 0 0.25 0.5 0.75 1
```

- 이미 만든 데이터셋/분류기 결과는 **자동 스킵**되므로 중단 후 재실행해도 이어서 진행된다. 전부 다시 하려면 `--force`.
- DiT 체크포인트 기본 경로: `src/checkpoints/mnist_dit_e900.pth` (다르면 `--sd-ckpt`로 지정).

## 6만장 공정 비교 실험 (생성 60k vs 원본 60k)

기존 coarse(2만장)에서 TSTR이 TRTR 천장을 넘은 것은, 확산 모델은 60k로 학습했는데
천장 분류기는 20k로만 학습했기 때문일 수 있다. 공정 비교를 위해 **생성도 60k,
원본 분류기도 60k(확산 모델이 학습한 것과 동일한 train 전체)** 로 맞춘다.

주의: MNIST train은 클래스당 5421(숫자 5)~6742(숫자 1)장으로 **불균형**이라
"클래스당 6000장 균등 추출"이 불가능하다. 그래서 원본 쪽은 `--real-n-per-class 0`
(= train 60k 전체, `data/real_full`)을 쓰고, 생성 쪽만 6000/class 균등으로 만든다.

```bash
# 관심 구간(음수 cfg 포함) × 생성 6만 장 × 원본 60k 전체
python experiments/run_sweep.py --stage full60k \
    --cfgs -0.5 -0.4 -0.3 -0.25 -0.2 -0.1 0 \
    --n-per-class 6000 --real-n-per-class 0

# 발표용 저분산 버전 (분류기 seed 3개 평균)
python experiments/run_sweep.py --stage full60k \
    --cfgs -0.5 -0.4 -0.3 -0.25 -0.2 -0.1 0 \
    --n-per-class 6000 --real-n-per-class 0 --clf-seeds 0 1 2
```

- 생성 6만 장은 2만 장의 3배 시간이 걸린다(모델 로드는 1회, cfg당 약 3배).
- `--real-n-per-class`를 생략하면 기존처럼 생성과 같은 크기의 균등 서브셋을 쓴다(coarse 재현용).

## 데이터 개수 제한 실험 (예: 1000장만으로 전체 파이프라인)

`main.py train --n-per-class N` 을 주면 확산 모델이 클래스당 N장(총 10N장)의
클래스 균등 서브셋으로만 학습된다. 이 서브셋은 `experiments/data/real_n{N}_seed{K}`에
저장·재사용되며, run_sweep의 TRTR 분류기가 **문자 그대로 같은 파일**에서 학습하므로
"확산 모델이 본 데이터 = 원본 분류기가 본 데이터"가 보장된다.

```bash
# 예: 총 1000장(클래스당 100장) 실험 — 3단계
# 1) 확산 모델 학습 (서브셋 자동 생성, ckpt는 mnist_dit_n100.pth로 자동 저장)
#    데이터가 60분의 1이므로 epoch을 늘리는 것을 권장 (스텝 수 유지 목적)
python main.py train --n-per-class 100 --epochs 900 --batch-size 256 --snapshot-every 300

# 2) 스윕: 생성 1000장(100/class) + TRTR(같은 1000장) + TSTR + 그림
python experiments/run_sweep.py --stage n1000 \
    --cfgs -0.5 -0.4 -0.3 -0.25 -0.1 0 \
    --n-per-class 100 --sd-ckpt checkpoints/mnist_dit_n100.pth

# 다른 크기도 동일 패턴: --n-per-class 1000 → 총 1만 장, ckpt mnist_dit_n1000.pth
```

- `--subset-seed`(기본 0)를 바꾸면 다른 서브셋으로 실험 가능. run_sweep의 real subset seed(0 고정)와
  일치해야 데이터가 공유되므로, 특별한 이유가 없으면 기본값을 쓴다.
- `--ckpt-name`을 직접 주지 않으면 `mnist_dit_n{N}.pth`로 자동 명명되어 기존 60k ckpt를 덮어쓰지 않는다.
- 생성 개수를 학습 데이터보다 늘려보고 싶으면(예: 1000장 학습 → 6만 장 생성) run_sweep에서
  `--n-per-class 6000 --real-n-per-class 100` 처럼 두 인자를 분리 지정하면 된다.

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
python experiments/make_real_subset.py --n-per-class 0        # train 60k 전체 → data/real_full
python experiments/train_classifier.py --data-dir experiments/data/gen_cfg4_n2000_seed0 --seed 0
python experiments/judge_and_diversity.py --data-dir experiments/data/gen_cfg4_n2000_seed0
python experiments/plot.py --csv experiments/results/results_coarse.csv --stage coarse
```

---

# 데이터 크기(n) × 합성 증강(r) 실험 — `datasize/`

> 설계서: `자료/실험설계_데이터크기_합성증강.md`

CFG 스윕과 **별개의 실험**이다. CFG를 **-0.25로 고정**하고, 변수를 **원본 데이터 크기 n**과
**합성 비율 r=s/n**으로 둔다. 질문: *"원본이 n개일 때 그 n개로 학습한 확산이 만든 합성을
더하면 분류기가 좋아지는가? 이득은 n이 작을수록 큰가?"*

이 실험은 기존 `experiments/` 모듈(`config.py`·`generate.py`·`train_classifier.py`)을
**재사용**하며, 기존 파일은 건드리지 않는다. 코드는 `experiments/datasize/`에 있다.

## 변수·조건

- **n** ∈ {1k, 5k, 10k, 60k} — 확산 학습 = 분류기 baseline 원본(같은 데이터).
  ⚠️ n=60k는 val 5k 예약으로 **실제 학습 55k**, 합성은 기존 `mnist_dit_e900.pth` 재사용.
- **r** ∈ {0, 0.5, 1, 2, 4, 8} — r=0은 baseline.
- **조건**: baseline(원본 n) / augmented(원본 n + 합성 s) / oracle(원본 n + 추가 진짜 s).
- **seed** 3개(0·1·2) 평균±표준편차.
- **통제**: real val 5k로 **early stopping**(Adam·batch128·patience8), test 10k는 평가 전용,
  진짜·합성 동일 정규화, 라벨 균등.

## 실행 (src/src 에서, 순서대로)

```bash
python experiments/datasize/splits.py                              # 1) 분할(val/pool/n서브셋)
python experiments/datasize/train_diffusion_ds.py --all --steps 40000  # 2) 작은 n 확산 학습
python experiments/datasize/gen_synth.py --all                     # 3) n별 합성 풀 생성(cfg=-0.25)
python experiments/datasize/run_datasize.py                        # 4) (n,r,seed) 그리드→CSV
python experiments/datasize/aggregate.py                           # 5) seed 집계→요약 CSV
```

- 조건별 결과 캐시 → 중단 후 재실행 시 이어서 진행(`--force`로 무시).
- 빠른 점검: `run_datasize.py --seeds 0 --keys n1000`.
- 큰 n 축소(메모리/시간): `gen_synth.py --key n60000 --max-r 4`.

## oracle 가용성 (MNIST 클래스 불균형 반영, 최소 클래스 4,921/class 기준)

| n | oracle 가능 r |
|---|---|
| 1,000 / 5,000 | 0.5·1·2·4·8 (전부) |
| 10,000 | 0.5·1·2 (4·8은 pool 여분 부족) |
| 60,000(=55k) | 없음 (pool 소진) |

불가 지점은 코드가 자동 생략한다.

## 파일 구성 (`datasize/`)

| 파일 | 책임 |
|---|---|
| `ds_config.py` | 그리드·경로·early stopping 분류기 설정, oracle 가용성 판정(불균형 반영) |
| `splits.py` | val/pool/n서브셋 고정 분할 + oracle 추가원본 추출(n서브셋과 disjoint) |
| `train_diffusion_ds.py` | n별 DiT 학습(동일 step 수). n60000은 기존 e900 재사용 |
| `gen_synth.py` | n별 합성 풀(=8n) 생성 — `generate.py` 재사용 |
| `train_clf_es.py` | real val 조기종료 분류기 — `MnistCNN`/`evaluate` 재사용 |
| `run_datasize.py` | baseline/aug/oracle 그리드 실행 → `results_datasize.csv` |
| `aggregate.py` | seed 평균±std + gain_vs_baseline + oracle_minus_aug → `summary_datasize.csv` |

## 산출물 (`results/datasize/`)

```
clf/{key}_r{r}_{kind}_seed{seed}.json   # 조건별 top-1 / val_acc / best_epoch
results_datasize.csv                    # n_key,total,r,s,kind,seed,n_train,accuracy,val_acc,best_epoch
summary_datasize.csv                    # 평균±std + gain_vs_baseline + oracle_minus_aug
```

---

## 설계서의 함정 체크 반영 사항

- `RandomHorizontalFlip` 미사용 — 원본은 `MNIST.data`에서 직접 추출, 분류기는 [0,1]→(0.1307, 0.3081) 정규화만.
- `cfg=0` = 순수 조건부(무조건 아님). `cond=None`은 TSTR 축에서 제외.
- TRTR·TSTR 학습 크기 매칭(같은 `--n-per-class`), CNN 구조/에폭/옵티마이저 전 구간 동일(seed만 변경).
- 심판 분류기는 원본 60k로 1회 학습 후 동결(`ckpts/judge.pt`), TSTR 분류기와 분리.
- 생성은 EMA 가중치 사용, 생성 seed는 전 cfg 동일 고정.
- test 10k는 평가 전용(어떤 학습에도 미사용).

추가 필요 패키지: `matplotlib` (`pip install matplotlib`)
