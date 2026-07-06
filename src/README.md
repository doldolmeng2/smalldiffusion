# MNIST 조건부 DiT (smalldiffusion)

숫자 라벨(0~9)을 입력하면 그 숫자처럼 생긴 이미지를 생성하는 class-conditional
diffusion transformer. `smalldiffusion` 라이브러리의 `DiT`를 사용합니다.
원래 Colab 노트북(`mnist_dit_colab.ipynb`)에서 하나씩 실행하던 과정을
서버에서 돌리기 좋게 `main.py` 하나로 정리한 버전입니다.

## 파일 구조

```
mnist-dit/
├── main.py              # 학습 / 샘플링 / 평가 진입점 (서브커맨드: train, sample, eval)
├── requirements.txt
├── README.md
├── datasets/            # MNIST 자동 다운로드 위치 (첫 실행 시 자동 생성)
├── checkpoints/         # 체크포인트 저장 위치 (자동 생성)
└── outputs/
    └── samples/         # 생성된 이미지 저장 위치 (자동 생성)
```

`datasets/`, `checkpoints/`, `outputs/samples/`는 미리 만들 필요 없이
`main.py` 실행 시 자동으로 생성됩니다.

## 1. 환경 설정

```bash
cd mnist-dit
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 마지막 줄이 `smalldiffusion`을 GitHub에서 직접 설치합니다
(PyPI 버전이 아니라 최신 소스 기준). 이 프로젝트에서 쓰는 `CondEmbedderLabel`,
`DiT` 등은 GitHub 저장소 기준으로 확인한 API입니다.

만약 라이브러리 소스 자체를 직접 열어보거나 수정하고 싶다면, 대신 아래처럼
로컬에 clone해서 editable 모드로 설치할 수도 있습니다:

```bash
git clone https://github.com/yuanchenyang/smalldiffusion.git
pip install -e smalldiffusion/
```

멀티 GPU를 쓸 경우에만 `accelerate config`로 분산 설정을 해주고
`accelerate launch main.py ...`로 실행하세요. 단일 GPU면 아래처럼
`python main.py ...`로 바로 실행하면 됩니다.

## 2. 학습

```bash
python main.py train --epochs 300 --batch-size 1024
```

- 체크포인트(`checkpoints/mnist_dit.pth`)가 이미 있으면 자동으로 이어서 학습합니다.
  `--epochs`를 더 크게 주고 다시 실행하면 그만큼만 추가로 학습합니다.
- 처음부터 새로 학습하고 싶으면 `--fresh`를 추가하세요.
- 다른 이름으로 별도 체크포인트를 만들고 싶으면 `--ckpt-name mnist_dit_v2.pth`처럼
  파일명만 바꾸면 됩니다 (같은 `checkpoints/` 폴더 안에 별도 파일로 저장, 기존 파일은
  그대로 유지).
- GPU 메모리가 부족하면 `--batch-size`를 줄이거나(256, 128, ...) `--mixed-precision fp16`
  (기본값)을 확인하세요. CPU만 있는 서버에선 `--mixed-precision no`로 두세요.

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--epochs` | 300 | 목표 epoch 수 |
| `--batch-size` | 256 | 배치 크기 |
| `--lr` | 1e-3 | learning rate (학습 내내 고정, 스케줄러 없음) |
| `--save-every` | 1 | 몇 epoch마다 체크포인트 저장할지 |
| `--mixed-precision` | fp16 | `no` / `fp16` / `bf16` |
| `--fresh` | (off) | 체크포인트 무시하고 새로 학습 |
| `--depth`, `--head-dim`, `--num-heads`, `--patch-size` | 6 / 32 / 6 / 2 | 모델 구조 |
| `--cfg-scale` | 4.0 | classifier-free guidance 강도 (학습에는 영향 없고 sample/eval에서 사용) |

`--depth`, `--head-dim`, `--num-heads`, `--patch-size`, `--num-classes`는
**train/sample/eval에서 반드시 동일한 값을 써야 합니다.** 다르면 체크포인트를
불러올 때 shape mismatch 에러가 납니다.

## 3. 샘플링

```bash
# 숫자 7을 16장 생성
python main.py sample --digit 7 --n-samples 16 --cfg-scale 4.0

# 0~9 전부 한 장의 그리드로 생성
python main.py sample --all-digits
```

결과는 `outputs/samples/digit_7.png` 또는 `outputs/samples/digits_0_to_9.png`로 저장됩니다.

## 4. 평가 (FID)

```bash
python main.py eval
```

실제 MNIST 이미지 1000장과 클래스별 생성 이미지 100장씩을 비교해 FID를 계산합니다.
`--fid-real-samples`, `--fid-per-class`로 샘플 수를 조절할 수 있습니다.

## 알아두면 좋은 것들 (Colab 개발 중 확인한 사항)

- **`cond_embed`는 `nn.Embedding`이 아니라 `CondEmbedderLabel`을 씁니다.** 후자는
  classifier-free guidance에 필요한 "unconditional" 토큰과, 학습 중 라벨을
  `dropout_prob` 확률로 드롭하는 로직을 내장하고 있습니다. `training_loop` 자체는
  이 드롭 로직을 갖고 있지 않습니다.
- **학습 데이터 변환에서 `RandomHorizontalFlip`을 뺐습니다.** 라이브러리 기본
  `img_train_transform`에는 flip이 포함되어 있는데, 숫자 이미지는 좌우 반전하면
  다른 숫자처럼 보일 수 있어 이 프로젝트에는 부적절합니다(`main.py`의
  `build_transform()`).
- **EMA 로드 순서에 주의했습니다.** `ema.to(device)`를 체크포인트 로드보다 먼저
  호출하면, `torch_ema`의 `load_state_dict()`가 shadow params를 "로드 시점 model의
  device"로 다시 캐스팅하면서 CPU에 고정돼버려, 이후 model만 GPU로 옮겨지면
  "cuda:0 and cpu" 에러가 납니다. `main.py`는 체크포인트 로드 후에 `ema.to()`를
  호출하도록 순서를 맞춰뒀습니다.
- lr은 학습 내내 고정이고 별도 스케줄러는 없습니다. `training_loop`가 매 배치 yield하는
  `ns` 객체에 `ns.optimizer`가 그대로 노출되어 있어서, 직접 lr 스케줄링을 넣고
  싶다면 `main.py`의 학습 루프 안에서 `ns.optimizer.param_groups[0]['lr']`을
  조작하면 됩니다 (현재는 구현되어 있지 않습니다).

## 참고 자료

- smalldiffusion 레포: https://github.com/yuanchenyang/smalldiffusion
- DiT 논문: https://arxiv.org/abs/2212.09748
- Classifier-free guidance 논문: https://arxiv.org/abs/2207.12598
