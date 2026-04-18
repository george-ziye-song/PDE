"""
LoRA V3 Model Evaluation for 3D Advection.

Multi-GPU evaluation using Accelerate (same pattern as shear flow / Taylor-Green).

Usage:
    torchrun --nproc_per_node=4 tools/visualize_advection_3d_lora.py \
        --config configs/finetune_advection_3d_v3_rescaled_norm.yaml \
        --checkpoint checkpoints_advection_3d_lora_v3_rescaled_norm/best_lora.pt --scan_all
"""

import argparse
import yaml
import torch
from pathlib import Path

from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint
from finetune.pde_loss_verified import Advection3DPDELoss

# 3D Advection channel: u -> scalar[11] = channel 14 (3+11)
CH_U = 14


def _vrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - gt) ** 2)
    var = torch.mean((gt - gt.mean()) ** 2)
    return torch.sqrt(mse / (var + eps))


def _nrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse_pred = torch.mean((pred - gt) ** 2)
    mse_zero = torch.mean(gt ** 2)
    return torch.sqrt(mse_pred / (mse_zero + eps))


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA V3 Eval for 3D Advection")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--scan_all',   action='store_true',
                        help='Scan all validation clips (multi-GPU)')
    parser.add_argument('--t_input',    type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, is_main: bool = True):
    ckpt_probe = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'config' in ckpt_probe:
        ckpt_model_cfg = ckpt_probe['config'].get('model', {})
        arch_keys = ['decoder', 'patch_smoother', 'encoder', 'intra_patch', 'na',
                     'in_channels', 'hidden_dim', 'patch_size', 'num_layers', 'num_heads',
                     'vector_channels', 'scalar_channels', 'enable_1d', 'enable_3d']
        model_cfg = dict(config.get('model', {}))
        for k in arch_keys:
            if k in ckpt_model_cfg:
                model_cfg[k] = ckpt_model_cfg[k]
        config = {**config, 'model': model_cfg}
    del ckpt_probe

    model = PDELoRAModelV3(
        config=config,
        pretrained_path=config['model'].get('pretrained_path'),
        freeze_encoder=config['model'].get('freeze_encoder', False),
        freeze_decoder=config['model'].get('freeze_decoder', False),
    )
    checkpoint = load_lora_checkpoint(model, checkpoint_path)
    if is_main:
        if 'metrics' in checkpoint:
            print(f"  Checkpoint metrics: {checkpoint['metrics']}")
        if 'global_step' in checkpoint:
            print(f"  Global step: {checkpoint['global_step']}")
    return model.float()


def create_pde_loss_fn(config: dict) -> Advection3DPDELoss:
    physics = config.get('physics', {})
    L = physics.get('Lx', 2 * 3.141592653589793)
    N = physics.get('nx', 64)
    d = L / N
    return Advection3DPDELoss(
        dx=physics.get('dx', d),
        dy=physics.get('dy', d),
        dz=physics.get('dz', d),
        dt=physics.get('dt', 0.05),
        eq_scales=physics.get('eq_scales'),
        eq_weights=physics.get('eq_weights'),
    )


@torch.no_grad()
def scan_all_distributed(accelerator, model, val_loader, config, t_input):
    accelerator.wait_for_everyone()
    model.eval()

    pde_loss_fn = create_pde_loss_fn(config)

    max_batches = len(val_loader)
    nan = float('nan')
    local_pde   = torch.full((max_batches,), nan, device=accelerator.device)
    local_vrmse = torch.full((max_batches,), nan, device=accelerator.device)
    local_nrmse = torch.full((max_batches,), nan, device=accelerator.device)
    local_rmse  = torch.full((max_batches,), nan, device=accelerator.device)

    for i, batch in enumerate(val_loader):
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        a    = batch['params_a'].to(device=accelerator.device, dtype=torch.float32)
        b    = batch['params_b'].to(device=accelerator.device, dtype=torch.float32)
        c    = batch['params_c'].to(device=accelerator.device, dtype=torch.float32)

        input_data  = data[:, :t_input]           # [B, t_input, X, Y, Z, C]
        target_data = data[:, 1:t_input + 1]      # [B, t_input, X, Y, Z, C]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean          # [B, t_input, X, Y, Z, C]

        # PDE loss: prepend t0 frame
        with torch.autocast(device_type='cuda', enabled=False):
            t0_u  = input_data[:, 0:1, :, :, :, CH_U].float()
            out_u = output[:, :, :, :, :, CH_U].float()
            u_seq = torch.cat([t0_u, out_u], dim=1)  # [B, T+1, X, Y, Z]
            pde_loss, _ = pde_loss_fn(u_seq, a, b, c)

        gt_u   = target_data[..., CH_U]
        pred_u = output[..., CH_U]
        vrmse  = _vrmse_torch(gt_u, pred_u)
        nrmse  = _nrmse_torch(gt_u, pred_u)
        rmse   = torch.sqrt(torch.mean((pred_u - gt_u) ** 2) + 1e-8)

        local_pde  [i] = pde_loss.detach()
        local_vrmse[i] = vrmse.detach()
        local_nrmse[i] = nrmse.detach()
        local_rmse [i] = rmse.detach()

    accelerator.wait_for_everyone()
    all_pde   = accelerator.gather(local_pde)
    all_vrmse = accelerator.gather(local_vrmse)
    all_nrmse = accelerator.gather(local_nrmse)
    all_rmse  = accelerator.gather(local_rmse)
    accelerator.wait_for_everyone()

    mask = ~torch.isnan(all_pde)
    n = mask.sum().item()
    if n > 0:
        return {
            'num_batches': n,
            'pde':   float(all_pde  [mask].mean()),
            'rmse':  float(all_rmse [mask].mean()),
            'vrmse': float(all_vrmse[mask].mean()),
            'nrmse': float(all_nrmse[mask].mean()),
        }
    return {'num_batches': 0, 'pde': 0.0, 'rmse': 0.0, 'vrmse': 0.0, 'nrmse': 0.0}


def main():
    args   = parse_args()
    config = load_config(args.config)

    t_input    = args.t_input    or config.get('dataset', {}).get('t_input', 8)
    batch_size = args.batch_size or config.get('dataloader', {}).get('batch_size', 2)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"{'='*60}")
        print(f"3D Advection LoRA Evaluation")
        print(f"{'='*60}")
        print(f"  Devices:    {accelerator.num_processes}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  t_input:    {t_input}")

    model = load_model(config, args.checkpoint, is_main=is_main)

    val_dataset = FinetuneDataset(
        data_path=config['dataset']['path'],
        temporal_length=t_input + 1,
        split='val',
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        seed=config['dataset'].get('seed', 42),
        clips_per_sample=None,
        vector_dim=config['dataset'].get('vector_dim', 0),
        val_time_interval=config['dataset'].get('val_time_interval', 8),
    )

    if is_main:
        print(f"  Val clips:  {len(val_dataset)}")

    if not args.scan_all:
        if is_main:
            print("\nUse --scan_all for full evaluation.")
        return

    val_sampler = FinetuneSampler(
        val_dataset, batch_size, shuffle=False,
        seed=config['dataset'].get('seed', 42),
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        collate_fn=finetune_collate_fn,
        num_workers=config.get('dataloader', {}).get('num_workers', 2),
        pin_memory=True,
    )

    model, val_loader = accelerator.prepare(model, val_loader)

    results = scan_all_distributed(accelerator, model, val_loader, config, t_input)

    if is_main:
        print(f"\n{'='*60}")
        print(f"Results ({results['num_batches']} batches):")
        print(f"  PDE Loss:  {results['pde']:.6f}")
        print(f"  RMSE (u):  {results['rmse']:.6f}")
        print(f"  VRMSE (u): {results['vrmse']:.6f}")
        print(f"  NRMSE (u): {results['nrmse']:.6f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
