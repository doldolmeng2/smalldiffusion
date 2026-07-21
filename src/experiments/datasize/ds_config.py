"""
데이터 크기(n) × 합성 증강(r) 실험 — 공통 설정.

핵심 질문:
  "원본 데이터가 n개일 때, 그 n개로 학습한 확산모델이 만든 합성 데이터 s개를 더해
   분류기를 학습하면 정확도가 오르는가? 그 이득은 n에 따라 어떻게 변하는가?"

설계 요약:
  - CFG_SCALE = -0.25 로 고정 (데이터 크기별 CFG 경향성은 유사하다고 확인 → 상수로 둠)
  - 변수: n(원본 총량), r(=s/n, 합성 비율)
  - 조건: baseline(n real) / augmented(n real + s synth) / oracle(n real + s real)
  - 지표: 진짜 test 10k top-1 정확도, seed 3개 평균±표준편차

기존 CFG-TSTR 실험의 experiments/config.py(=C)를 재사용한다:
  MnistCNN, normalize_for_clf, save/load_dataset, DIT_KW, 샘플링 상수, 경로 등.
이 파일은 '데이터 크기 실험' 전용 상수/경로/그리드만 추가한다.
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent      # .../experiments/datasize
_SRC = _EXP.parents[1]                       # .../src/src (main.py 위치)
for _p in (str(_SRC), str(_SRC / 'experiments'), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C   # 기존 실험 공통 모듈 (experiments/config.py)


# ---------------------------------------------------------------------------
# 그리드
# ---------------------------------------------------------------------------
CFG_SCALE = -0.25                            # 고정
R_LIST    = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]   # r = s / n  (0 = baseline)
SEEDS     = [0, 1, 2]                        # 분류기/샘플 추출 seed

# early stopping용 real 검증셋(60k에서 예약) → 나머지가 학습 풀
VAL_SIZE    = 5000
POOL_SIZE   = 60000 - VAL_SIZE               # 55000: 모든 학습 서브셋/oracle 추출 풀
MASTER_SEED = 0                              # split 인덱스 고정 seed (재현성)

# MNIST train 클래스별 개수(고정 사실). oracle 가용성은 '평균'이 아니라 '가장 적은 클래스'가
# 결정하므로 정확히 판정하려면 이 값이 필요하다 (합=60000).
MNIST_TRAIN_COUNTS = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

# n(원본 총량) 정의.
#   npc = 클래스당 개수 (균등 추출). npc=None 은 'pool 전체'(라벨 불균형 그대로).
#   ⚠️ 'n60000' 은 val 5k 예약 때문에 실제 학습에 쓰는 원본은 pool 전체 = 55000장.
#      (문서 '한계' 참고. x축엔 실제 학습량 55000으로 표기 권장)
N_SPECS = [
    dict(key='n1000',  total=1000,       npc=100),
    dict(key='n5000',  total=5000,       npc=500),
    dict(key='n10000', total=10000,      npc=1000),
    dict(key='n60000', total=POOL_SIZE,  npc=None),   # full pool(≈60k, 실제 55k)
]
N_BY_KEY = {s['key']: s for s in N_SPECS}


# ---------------------------------------------------------------------------
# 분류기 (early stopping) — 이 실험 전체(baseline/aug/oracle)에 동일 적용
# ---------------------------------------------------------------------------
EARLY_STOP       = True     # False면 고정 epoch(CLF_FIXED_EPOCHS) 사용
CLF_LR           = 1e-3     # Adam
CLF_BATCH        = 128
CLF_MAX_EPOCH    = 60       # early stopping 상한
CLF_PATIENCE     = 8        # val acc가 이만큼 epoch 개선 없으면 중단(최적 가중치 복원)
CLF_FIXED_EPOCHS = 15       # EARLY_STOP=False일 때 사용


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
SPLIT_DIR   = C.DATA_DIR / 'ds_splits'         # val / pool / n별 서브셋
SYNTH_DIR   = C.DATA_DIR / 'ds_synth'          # n별 합성 풀
DS_CKPT_DIR = C.CKPT_DIR / 'ds_diffusion'      # n별 DiT 체크포인트
DS_RESULTS  = C.RESULTS_DIR / 'datasize'       # CSV/요약
DS_CLF_DIR  = DS_RESULTS / 'clf'               # 조건별 result.json (재실행 스킵용)

VAL_DIR  = SPLIT_DIR / 'val'
POOL_DIR = SPLIT_DIR / 'pool'


def ensure_dirs():
    C.ensure_dirs()
    for d in (SPLIT_DIR, SYNTH_DIR, DS_CKPT_DIR, DS_RESULTS, DS_CLF_DIR):
        d.mkdir(parents=True, exist_ok=True)


def subset_dir(key) -> Path:
    return SPLIT_DIR / key


def synth_dir(key) -> Path:
    return SYNTH_DIR / key


def diffusion_ckpt(key) -> Path:
    """n별 확산 체크포인트 경로.
    가장 큰 n(n60000)은 이미 학습된 full-60k 모델(config.SD_CKPT)을 재사용한다.
    (그 모델은 val 5k를 포함한 60k로 학습됐지만, val은 분류기 조기종료에만 쓰고
     학습셋엔 안 들어가므로 학습 누수는 아님 — 문서의 '한계' 참고.)
    """
    if key == 'n60000':
        return C.SD_CKPT
    return DS_CKPT_DIR / f'{key}.pth'


def oracle_feasible(key: str, r: float) -> bool:
    """oracle(원본+추가 원본) 가능 여부.

    각 클래스에서 균등하게 뽑으므로 병목은 '가장 적은 클래스'의 pool 여유다:
      클래스 c의 pool 개수 = MNIST_TRAIN_COUNTS[c] - (VAL_SIZE/10)
      n-서브셋이 클래스당 npc개를 이미 차지 → 여분 = pool_c - npc
    모든 클래스에서 여분 >= s_per_class 여야 가능. (full pool(npc=None)은 여분 없음)
    """
    spec = N_BY_KEY[key]
    npc = spec['npc']
    if r <= 0 or npc is None:
        return False
    spc = s_count(spec['total'], r) // C.NUM_CLASSES
    val_per_class = VAL_SIZE // C.NUM_CLASSES
    min_pool_per_class = min(cnt - val_per_class for cnt in MNIST_TRAIN_COUNTS)
    return npc + spc <= min_pool_per_class


def s_count(total: int, r: float) -> int:
    """추가 장수 s = round(r*n)을 클래스 균등(10의 배수)으로 내림."""
    s = int(round(r * total))
    return (s // C.NUM_CLASSES) * C.NUM_CLASSES


def synth_pool_per_class(total: int, max_r: float = None) -> int:
    """합성 풀을 max(r)*n 크기(클래스 균등)로 미리 생성해두기 위한 클래스당 개수."""
    max_r = max(R_LIST) if max_r is None else max_r
    return int(round(max_r * total)) // C.NUM_CLASSES
