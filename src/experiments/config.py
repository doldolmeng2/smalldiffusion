"""
CFG–TSTR 실험 공통 상수/유틸 모듈.

실험설계_CFG_TSTR.md 4장 "단일 변수 원칙"을 코드 레벨에서 강제한다:
변수는 cfg_scale 하나뿐이고, 그 외 모든 것(샘플러/step 수/분류기 구조·학습 설정/전처리)은
전부 이 파일의 상수를 참조한다. TRTR·TSTR·심판 분류기가 모두 같은 상수를 쓴다.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# 경로 규약
# ---------------------------------------------------------------------------
EXP_DIR     = Path(__file__).resolve().parent        # .../src/src/experiments
SRC_DIR     = EXP_DIR.parent                         # .../src/src (main.py 위치)
DATA_DIR    = EXP_DIR / 'data'                       # 생성/원본 데이터셋들
CKPT_DIR    = EXP_DIR / 'ckpts'                      # judge.pt, clf_*.pt
RESULTS_DIR = EXP_DIR / 'results'
CLF_DIR     = RESULTS_DIR / 'clf'                    # 분류기별 result.json
FIGS_DIR    = RESULTS_DIR / 'figs'
LOGS_DIR    = EXP_DIR / 'logs'

MNIST_ROOT = SRC_DIR / 'datasets'                    # main.py와 동일 위치 재사용
# 학습 완료된 DiT (3090Ti 서버 기준: <repo>/src/checkpoints/mnist_dit_e900.pth)
SD_CKPT    = SRC_DIR / 'checkpoints' / 'mnist_dit_e900.pth'
JUDGE_CKPT = CKPT_DIR / 'judge.pt'                   # 동결 심판 분류기


def ensure_dirs():
    for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, CLF_DIR, FIGS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 데이터 / 확산 모델(DiT) / 샘플링 상수
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
IMG_SIZE    = 28

# DiT 구조 — main.py 학습 기본값과 반드시 동일해야 체크포인트가 로드된다.
DIT_KW = dict(in_dim=IMG_SIZE, channels=1, patch_size=2, depth=6,
              head_dim=32, num_heads=6, mlp_ratio=4.0)
COND_DIM     = DIT_KW['head_dim'] * DIT_KW['num_heads']   # 192
COND_DROPOUT = 0.1
EMA_DECAY    = 0.99

# 노이즈 스케줄 / 샘플링 — 전 구간 고정 (cfg_scale만 변수)
BETA_START, BETA_END, SCHEDULE_N = 1e-4, 0.02, 1000
SAMPLE_STEPS = 20
GAM          = 1.6
MU           = 0.0
GEN_SEED     = 0      # 생성 seed 정책: 전 cfg 동일한 seed 1개 사용

# MNIST 분류기 정규화 상수 (PyTorch 공식 예제와 동일).
# 주의: 확산 모델 학습/생성 쪽 전처리에는 RandomHorizontalFlip 금지(숫자 오염).
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# ---------------------------------------------------------------------------
# 분류기 — PyTorch 공식 예제(pytorch/examples → mnist/main.py) 구조 그대로.
# TRTR·모든 TSTR·심판이 전부 이 구조/설정을 쓰고, seed만 바꾼다.
# ---------------------------------------------------------------------------
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)


CLF_BATCH      = 128
CLF_TEST_BATCH = 1000
CLF_EPOCHS     = 5      # 에폭 고정 — TRTR/TSTR 전 구간 동일
CLF_LR         = 1.0    # Adadelta
CLF_LR_GAMMA   = 0.7    # StepLR(step_size=1)
JUDGE_EPOCHS   = 8      # 심판만 에폭을 늘려 상한 확보(구조·옵티마이저는 동일)


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def torch_load(path):
    """체크포인트 로더.

    - torch 버전별 weights_only 기본값 차이 호환.
    - main.py가 저장한 체크포인트에는 'args'에 func(=cmd_train 함수 객체)가 pickle로
      들어있어, main.py 밖에서 로드하면 "Can't get attribute 'cmd_train'" 에러가 난다.
      해석 불가능한 참조는 placeholder로 대체해서 무시한다 (model/ema 가중치만 쓰면 됨).
    """
    import io
    import pickle

    class _Missing:  # 해석 실패한 pickle 참조 자리표시자
        def __init__(self, *a, **k):
            pass

        def __repr__(self):
            return '<unresolvable pickled object>'

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                return _Missing

    class _SafePickle:
        Unpickler = _SafeUnpickler

        @staticmethod
        def load(f, **kw):
            return _SafeUnpickler(f, **kw).load()

        @staticmethod
        def loads(b, **kw):
            return _SafeUnpickler(io.BytesIO(b), **kw).load()

    try:
        return torch.load(path, map_location='cpu', weights_only=False,
                          pickle_module=_SafePickle)
    except TypeError:  # 구버전 torch: weights_only 인자 없음
        return torch.load(path, map_location='cpu', pickle_module=_SafePickle)


def fmt(v: float) -> str:
    """폴더명용 숫자 포맷 (4.0 -> '4', 0.5 -> '0.5')."""
    return f'{float(v):g}'


def gen_dir(cfg_scale, n_per_class, seed) -> Path:
    return DATA_DIR / f'gen_cfg{fmt(cfg_scale)}_n{n_per_class}_seed{seed}'


def real_dir(n_per_class, seed) -> Path:
    """n_per_class=0 은 '원본 train 60k 전체(클래스 불균형 그대로)'를 뜻하는 규약.

    주의: MNIST train은 클래스당 개수가 5421(숫자 5)~6742(숫자 1)로 불균형이라
    '클래스당 6000장 균등 추출'은 불가능하다. 6만장 실험의 원본 쪽은 반드시
    n_per_class=0(전체)을 사용한다.
    """
    if n_per_class == 0:
        return DATA_DIR / 'real_full'
    return DATA_DIR / f'real_n{n_per_class}_seed{seed}'


# ---------------------------------------------------------------------------
# 데이터셋 저장 포맷 — generate.py / make_real_subset.py 공통 (동일 포맷 필수)
#   out_dir/images.pt : uint8 (N, 1, 28, 28), 값 0~255
#   out_dir/labels.pt : int64 (N,)
#   out_dir/meta.json : 생성 조건 기록
# ---------------------------------------------------------------------------
def save_dataset(out_dir: Path, images01: torch.Tensor, labels: torch.Tensor, meta: dict):
    """images01: float [0,1], (N,1,28,28)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = (images01.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    torch.save(imgs, out_dir / 'images.pt')
    torch.save(labels.to(torch.int64).cpu(), out_dir / 'labels.pt')
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_dataset(data_dir: Path):
    """returns (images float [0,1] (N,1,28,28), labels int64 (N,), meta dict)"""
    data_dir = Path(data_dir)
    imgs = torch.load(data_dir / 'images.pt').float() / 255.0
    labels = torch.load(data_dir / 'labels.pt')
    meta = json.loads((data_dir / 'meta.json').read_text())
    return imgs, labels, meta


def dataset_exists(data_dir: Path) -> bool:
    data_dir = Path(data_dir)
    return all((data_dir / f).exists() for f in ('images.pt', 'labels.pt', 'meta.json'))


def normalize_for_clf(images01: torch.Tensor) -> torch.Tensor:
    """[0,1] 이미지 → 분류기 입력 정규화."""
    return (images01 - MNIST_MEAN) / MNIST_STD
