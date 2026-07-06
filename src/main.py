#!/usr/bin/env python3
"""
MNIST 조건부(class-conditional) Diffusion Transformer (DiT) — smalldiffusion 기반

서브커맨드:
    train   학습 (체크포인트 있으면 자동 이어서 학습)
    sample  조건부 이미지 생성
    eval    FID 평가

사용 예:
    python main.py train --epochs 300 --batch-size 256
    python main.py train --epochs 600                      # 이어서 300 epoch 더 학습
    python main.py train --fresh                            # 체크포인트 무시하고 새로 학습
    python main.py sample --digit 7 --n-samples 16 --cfg-scale 4.0
    python main.py sample --all-digits
    python main.py eval

주의: train/sample/eval 사이에는 --depth, --head-dim, --num-heads, --patch-size 등
모델 구조 관련 인자를 반드시 동일하게 유지해야 합니다. 다르면 체크포인트 로드 시
shape mismatch 에러가 납니다.
"""
import argparse
import os
import random

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA

from smalldiffusion import (
    ScheduleDDPM, samples, training_loop,
    img_normalize, DiT, CondEmbedderLabel,
)


# ---------------------------------------------------------------------------
# 데이터 / 모델 빌더
# ---------------------------------------------------------------------------

def build_transform():
    # smalldiffusion 기본 제공 img_train_transform엔 RandomHorizontalFlip이 들어있는데,
    # 숫자 이미지는 좌우 반전하면 다른 숫자처럼 보일 수 있어(2,3,5,7 등) 이 프로젝트에는
    # 부적절합니다. flip을 뺀 버전을 직접 정의해서 사용합니다.
    return tf.Compose([
        tf.ToTensor(),
        tf.Lambda(lambda t: (t * 2) - 1),
    ])


def build_dataset(args):
    return MNIST(args.data_dir, train=True, download=True, transform=build_transform())


def build_model(args):
    dim = args.head_dim * args.num_heads
    return DiT(
        in_dim=28, channels=1, patch_size=args.patch_size, depth=args.depth,
        head_dim=args.head_dim, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        cond_embed=CondEmbedderLabel(dim, args.num_classes, dropout_prob=args.dropout_prob),
    )


def ckpt_path(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    return os.path.join(args.ckpt_dir, args.ckpt_name)


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def cmd_train(args):
    accel = Accelerator(mixed_precision=args.mixed_precision)

    # wandb 초기화 (--no-wandb로 끌 수 있음). main 프로세스에서만 로깅.
    use_wandb = (not args.no_wandb) and accel.is_main_process
    if use_wandb:
        import wandb
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers)

    schedule = ScheduleDDPM(beta_start=args.beta_start, beta_end=args.beta_end,
                             N=args.diffusion_steps)
    model = build_model(args)
    ema = EMA(model.parameters(), decay=args.ema_decay)

    path = ckpt_path(args)
    start_epoch = 0
    loss_history = []
    if os.path.exists(path) and not args.fresh:
        print(f'[train] 기존 체크포인트 발견: {path} - 이어서 학습합니다.')
        ckpt = torch.load(path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        start_epoch = ckpt.get('epoch', 0)
        loss_history = ckpt.get('loss_history', [])
        print(f'[train] {start_epoch} epoch까지 학습된 상태에서 재개합니다.')
    else:
        print('[train] 체크포인트 없음(또는 --fresh 지정됨) - 새로 학습을 시작합니다.')

    # 주의: ema.to()는 반드시 checkpoint 로드 이후에 호출해야 합니다.
    # torch_ema의 load_state_dict()는 shadow_params를 "로드 시점 model 파라미터의 device"에
    # 맞춰 다시 캐스팅합니다. 이 시점엔 model이 아직 GPU로 옮겨지기 전(CPU)이므로, 순서가
    # 바뀌면 이후 accelerator.prepare()가 model만 GPU로 옮기면서
    # "Expected all tensors to be on the same device, cuda:0 and cpu" 에러가 납니다.
    ema.to(accel.device)

    remaining = max(args.epochs - start_epoch, 0)
    print(f'[train] 목표 epoch: {args.epochs}, 남은 epoch: {remaining}')
    if remaining == 0:
        print('[train] 이미 목표 epoch만큼 학습되어 있습니다. '
              '--epochs를 늘리거나 --fresh로 새로 시작하세요.')
        return

    epoch_counter = start_epoch
    current_epoch_loss = None
    global_step = 0

    def save(epoch):
        torch.save({
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'epoch': epoch,
            'loss_history': loss_history,
            'args': vars(args),
        }, path)

    for ns in training_loop(loader, model, schedule, epochs=remaining,
                             lr=args.lr, accelerator=accel, conditional=True):
        ns.pbar.set_description(f'Epoch {epoch_counter+1}/{args.epochs} Loss={ns.loss.item():.5f}')
        ema.update()

        step_loss = ns.loss.item()
        global_step += 1
        if use_wandb:
            wandb.log({'train/loss': step_loss, 'epoch': epoch_counter}, step=global_step)

        # pbar.n은 "완료된 epoch 수". 값이 바뀐 시점 = 직전 epoch이 막 끝난 시점이고,
        # current_epoch_loss에는 그 직전 epoch의 마지막 batch loss가 들어있음.
        if ns.pbar.n != epoch_counter - start_epoch:
            if current_epoch_loss is not None:
                loss_history.append(current_epoch_loss)
                if use_wandb:
                    wandb.log({'train/epoch_loss': current_epoch_loss,
                               'epoch': epoch_counter}, step=global_step)
            epoch_counter = start_epoch + ns.pbar.n
            if epoch_counter % args.save_every == 0:
                save(epoch_counter)

        current_epoch_loss = step_loss

    if current_epoch_loss is not None:
        loss_history.append(current_epoch_loss)
    save(args.epochs)
    print(f'[train] 학습 완료. 체크포인트 저장: {path}')
    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# 추론 공통 (sample / eval)
# ---------------------------------------------------------------------------

def load_for_inference(args):
    accel = Accelerator()
    model = build_model(args)
    ema = EMA(model.parameters(), decay=args.ema_decay)

    path = ckpt_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(f'체크포인트가 없습니다: {path}. 먼저 `train`을 실행하세요.')

    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    ema.to(accel.device)
    model.to(accel.device)
    model.eval()

    schedule = ScheduleDDPM(beta_start=args.beta_start, beta_end=args.beta_end,
                             N=args.diffusion_steps)
    return model, ema, schedule, accel


# ---------------------------------------------------------------------------
# 샘플링
# ---------------------------------------------------------------------------

def cmd_sample(args):
    model, ema, schedule, accel = load_for_inference(args)
    os.makedirs(args.sample_dir, exist_ok=True)

    digits = list(range(10)) if args.all_digits else [args.digit]
    grids = []
    with ema.average_parameters():
        for digit in digits:
            cond = torch.full((args.n_samples,), digit, dtype=torch.long)
            *_, x0 = samples(model, schedule.sample_sigmas(args.steps), gam=args.gam,
                              batchsize=args.n_samples, cond=cond, cfg_scale=args.cfg_scale,
                              accelerator=accel)
            grids.append(x0)

    all_imgs = torch.cat(grids, dim=0)
    grid_img = make_grid(all_imgs, nrow=args.n_samples)
    name = 'digits_0_to_9.png' if args.all_digits else f'digit_{args.digit}.png'
    out_path = os.path.join(args.sample_dir, name)
    save_image(img_normalize(grid_img), out_path)
    print(f'[sample] 저장 완료: {out_path}')


# ---------------------------------------------------------------------------
# 평가 (FID)
# ---------------------------------------------------------------------------

def cmd_eval(args):
    from torchmetrics.image.fid import FrechetInceptionDistance

    model, ema, schedule, accel = load_for_inference(args)
    dataset = build_dataset(args)

    fid = FrechetInceptionDistance(feature=64, normalize=True).to(accel.device)

    def to_fid_batch(x):
        x = img_normalize(x).clamp(0, 1)
        return x.repeat(1, 3, 1, 1)

    real_idx = random.sample(range(len(dataset)), min(args.fid_real_samples, len(dataset)))
    real_imgs = torch.stack([dataset[i][0] for i in real_idx])
    fid.update(to_fid_batch(real_imgs).to(accel.device), real=True)

    with ema.average_parameters():
        for digit in range(10):
            cond = torch.full((args.fid_per_class,), digit, dtype=torch.long)
            *_, x0 = samples(model, schedule.sample_sigmas(args.steps), gam=args.gam,
                              batchsize=args.fid_per_class, cond=cond, cfg_scale=args.cfg_scale,
                              accelerator=accel)
            fid.update(to_fid_batch(x0).to(accel.device), real=False)

    score = fid.compute().item()
    print(f'[eval] FID: {score:.4f}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(sp):
    # 경로
    sp.add_argument('--data-dir', default='datasets', help='MNIST 다운로드 경로')
    sp.add_argument('--ckpt-dir', default='checkpoints', help='체크포인트 저장 폴더')
    sp.add_argument('--ckpt-name', default='mnist_dit.pth',
                     help='체크포인트 파일명 (바꾸면 기존 파일을 건드리지 않고 새로 저장)')
    sp.add_argument('--sample-dir', default='outputs/samples', help='생성 이미지 저장 폴더')

    # 모델 구조 (train/sample/eval 간 반드시 동일해야 함)
    sp.add_argument('--num-classes', type=int, default=10)
    sp.add_argument('--patch-size', type=int, default=2)
    sp.add_argument('--depth', type=int, default=6)
    sp.add_argument('--head-dim', type=int, default=32)
    sp.add_argument('--num-heads', type=int, default=6)
    sp.add_argument('--mlp-ratio', type=float, default=4.0)
    sp.add_argument('--dropout-prob', type=float, default=0.1,
                     help='classifier-free guidance용 조건 드롭 확률 (CondEmbedderLabel)')
    sp.add_argument('--ema-decay', type=float, default=0.99)

    # 노이즈 스케줄 / 샘플링
    sp.add_argument('--beta-start', type=float, default=0.0001)
    sp.add_argument('--beta-end', type=float, default=0.02)
    sp.add_argument('--diffusion-steps', type=int, default=1000, help='ScheduleDDPM의 N')
    sp.add_argument('--gam', type=float, default=1.6, help='샘플러 gam (1=DDPM/DDIM, 2=accelerated)')
    sp.add_argument('--steps', type=int, default=20, help='샘플링 스텝 수')
    sp.add_argument('--cfg-scale', type=float, default=4.0, help='classifier-free guidance 강도')


def build_parser():
    p = argparse.ArgumentParser(
        description='MNIST 조건부 DiT (smalldiffusion) 학습/샘플링/평가',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest='mode', required=True)

    train_p = sub.add_parser('train', help='학습 실행 (체크포인트 있으면 자동 이어서 학습)')
    add_common_args(train_p)
    train_p.add_argument('--epochs', type=int, default=300)
    train_p.add_argument('--batch-size', type=int, default=256)
    train_p.add_argument('--lr', type=float, default=1e-3)
    train_p.add_argument('--save-every', type=int, default=1, help='몇 epoch마다 체크포인트 저장할지')
    train_p.add_argument('--num-workers', type=int, default=4)
    train_p.add_argument('--mixed-precision', default='fp16', choices=['no', 'fp16', 'bf16'])
    train_p.add_argument('--fresh', action='store_true',
                          help='기존 체크포인트를 무시하고 처음부터 새로 학습')
    # wandb
    train_p.add_argument('--no-wandb', action='store_true', help='wandb 로깅 비활성화')
    train_p.add_argument('--wandb-entity', default='ktypet13-hanyang-university')
    train_p.add_argument('--wandb-project', default='mnist-diffusion')
    train_p.add_argument('--wandb-run-name', default=None, help='wandb run 이름 (기본: 자동 생성)')
    train_p.set_defaults(func=cmd_train)

    sample_p = sub.add_parser('sample', help='조건부 이미지 생성')
    add_common_args(sample_p)
    sample_p.add_argument('--digit', type=int, default=0, choices=range(10))
    sample_p.add_argument('--n-samples', type=int, default=16, help='생성할 장수')
    sample_p.add_argument('--all-digits', action='store_true', help='0~9 전부 생성 (한 장의 그리드로 저장)')
    sample_p.set_defaults(func=cmd_sample)

    eval_p = sub.add_parser('eval', help='FID 정량 평가')
    add_common_args(eval_p)
    eval_p.add_argument('--fid-real-samples', type=int, default=1000)
    eval_p.add_argument('--fid-per-class', type=int, default=100)
    eval_p.set_defaults(func=cmd_eval)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
