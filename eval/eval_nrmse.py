"""
Evaluate LoRA finetuned models using paper-standard nRMSE metric.

nRMSE = sqrt( MSE(pred, gt) / MSE(0, gt) )
       = sqrt( mean((pred - gt)²) / mean(gt²) )

This is the relative RMSE normalized by GT energy, commonly used in
PDE benchmark papers (e.g., PDEBench, FactFormer).

Supports single-GPU and multi-GPU (torchrun) evaluation.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python eval/eval_nrmse.py \
        --config configs/finetune_shear_flow_v3.yaml \
        --checkpoint checkpoints_shear_flow_lora_v3/best_lora.pt

    # Multi-GPU
    torchrun --nproc_per_node=4 eval/eval_nrmse.py \
        --config configs/finetune_shear_flow_v3.yaml \
        --checkpoint checkpoints_shear_flow_lora_v3/best_lora.pt
"""

import os
import sys
import argparse
import warnings

if os.environ.get('LOCAL_RANK', '0') != '0':
    warnings.filterwarnings('ignore')

import torch
import yaml
import logging
from pathlib import Path
from accelerate import Accelerator
from rich.console import Console
from rich.table import Table

torch.set_float32_matmul_precision('high')

logging.basicConfig(
    level=logging.INFO if os.environ.get('LOCAL_RANK', '0') == '0' else logging.CRITICAL,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

from finetune.dataset_finetune import create_finetune_dataloaders
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint

# Channel name mappings for known datasets
CHANNEL_NAMES = {
    0: 'Vx', 1: 'Vy', 2: 'Vz',
    3: 'scalar_0', 4: 'scalar_1', 5: 'scalar_2', 6: 'scalar_3',
    14: 'tracer', 15: 'pressure',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate nRMSE = sqrt(MSE(pred,gt)/MSE(0,gt))")
    parser.add_argument('--config', type=str, required=True, help='Config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='LoRA checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'train'])
    parser.add_argument('--per_timestep', action='store_true', help='Report per-timestep nRMSE')
    parser.add_argument('--per_sample', action='store_true', help='Report per-sample nRMSE')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    accelerator: Accelerator,
    t_input: int,
    per_timestep: bool = False,
) -> dict:
    """
    Compute nRMSE = sqrt(MSE(pred, gt) / MSE(0, gt)) per channel.

    Returns dict with:
      - per_channel_nrmse: {ch_idx: nrmse}
      - overall_nrmse: float
      - per_channel_rmse: {ch_idx: rmse}
      - per_channel_gt_rms: {ch_idx: rms of gt}
      - per_timestep_nrmse: {ch_idx: [nrmse_t0, ...]} (if per_timestep)
      - num_samples: int
    """
    model.eval()
    accelerator.wait_for_everyone()

    # Accumulators: per-channel MSE(pred,gt) and MSE(0,gt)
    channel_mse_sum = {}       # {ch_idx: sum of (pred-gt)²}
    channel_gt_sq_sum = {}     # {ch_idx: sum of gt²}
    channel_count = {}         # {ch_idx: num elements}
    num_samples = 0

    # Per-timestep accumulators
    if per_timestep:
        ts_mse_sum = {}        # {ch_idx: {t: sum}}
        ts_gt_sq_sum = {}      # {ch_idx: {t: sum}}
        ts_count = {}          # {ch_idx: {t: count}}

    for batch in dataloader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target = data[:, 1:t_input + 1]
        B = data.shape[0]

        # Forward
        output_norm, mean, std = model(input_data, return_normalized=True)
        pred = output_norm * std + mean

        # Valid channels
        valid_ch = torch.where(channel_mask[0] > 0)[0] if channel_mask.dim() > 1 else torch.where(channel_mask > 0)[0]

        num_samples += B

        for ch in valid_ch:
            ch_idx = ch.item()
            p = pred[..., ch_idx]   # [B, T, H, W]
            g = target[..., ch_idx]

            mse_val = (p - g).pow(2)
            gt_sq = g.pow(2)

            if ch_idx not in channel_mse_sum:
                channel_mse_sum[ch_idx] = 0.0
                channel_gt_sq_sum[ch_idx] = 0.0
                channel_count[ch_idx] = 0

            channel_mse_sum[ch_idx] += mse_val.sum().item()
            channel_gt_sq_sum[ch_idx] += gt_sq.sum().item()
            channel_count[ch_idx] += mse_val.numel()

            if per_timestep:
                if ch_idx not in ts_mse_sum:
                    ts_mse_sum[ch_idx] = {}
                    ts_gt_sq_sum[ch_idx] = {}
                    ts_count[ch_idx] = {}
                T_out = p.shape[1]
                for t in range(T_out):
                    if t not in ts_mse_sum[ch_idx]:
                        ts_mse_sum[ch_idx][t] = 0.0
                        ts_gt_sq_sum[ch_idx][t] = 0.0
                        ts_count[ch_idx][t] = 0
                    ts_mse_sum[ch_idx][t] += mse_val[:, t].sum().item()
                    ts_gt_sq_sum[ch_idx][t] += gt_sq[:, t].sum().item()
                    ts_count[ch_idx][t] += mse_val[:, t].numel()

    # Reduce across GPUs
    accelerator.wait_for_everyone()

    def reduce_scalar(val):
        t = torch.tensor([val], device=accelerator.device)
        t = accelerator.reduce(t, reduction='sum')
        return t.item()

    num_samples = int(reduce_scalar(num_samples))

    per_channel_nrmse = {}
    per_channel_rmse = {}
    per_channel_gt_rms = {}

    for ch_idx in sorted(channel_mse_sum.keys()):
        mse_total = reduce_scalar(channel_mse_sum[ch_idx])
        gt_sq_total = reduce_scalar(channel_gt_sq_sum[ch_idx])
        count = reduce_scalar(channel_count[ch_idx])

        mse_mean = mse_total / count
        gt_sq_mean = gt_sq_total / count

        rmse = mse_mean ** 0.5
        gt_rms = gt_sq_mean ** 0.5
        nrmse = (mse_mean / (gt_sq_mean + 1e-10)) ** 0.5

        per_channel_nrmse[ch_idx] = nrmse
        per_channel_rmse[ch_idx] = rmse
        per_channel_gt_rms[ch_idx] = gt_rms

    # Overall nRMSE (average across channels)
    overall_nrmse = sum(per_channel_nrmse.values()) / max(len(per_channel_nrmse), 1)

    result = {
        'per_channel_nrmse': per_channel_nrmse,
        'per_channel_rmse': per_channel_rmse,
        'per_channel_gt_rms': per_channel_gt_rms,
        'overall_nrmse': overall_nrmse,
        'num_samples': num_samples,
    }

    # Per-timestep
    if per_timestep:
        per_ts = {}
        for ch_idx in sorted(ts_mse_sum.keys()):
            per_ts[ch_idx] = []
            for t in sorted(ts_mse_sum[ch_idx].keys()):
                mse_t = reduce_scalar(ts_mse_sum[ch_idx][t])
                gt_sq_t = reduce_scalar(ts_gt_sq_sum[ch_idx][t])
                cnt_t = reduce_scalar(ts_count[ch_idx][t])
                nrmse_t = ((mse_t / cnt_t) / (gt_sq_t / cnt_t + 1e-10)) ** 0.5
                per_ts[ch_idx].append(nrmse_t)
        result['per_timestep_nrmse'] = per_ts

    model.train()
    return result


def main():
    args = parse_args()
    config = load_config(args.config)
    console = Console()

    accelerator = Accelerator(mixed_precision='no')

    t_input = config['dataset'].get('t_input', 8)

    # Override architecture keys from checkpoint config to avoid decoder mismatch
    ckpt_probe = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'config' in ckpt_probe:
        ckpt_model_cfg = ckpt_probe['config'].get('model', {})
        arch_keys = ['decoder', 'patch_smoother', 'encoder', 'intra_patch', 'na',
                     'in_channels', 'hidden_dim', 'patch_size', 'num_layers', 'num_heads',
                     'vector_channels', 'scalar_channels', 'enable_1d', 'enable_3d']
        model_cfg = dict(config.get('model', {}))
        for k in arch_keys:
            if k in ckpt_model_cfg:
                if accelerator.is_main_process and k == 'decoder' and model_cfg.get(k) != ckpt_model_cfg[k]:
                    logger.info(f"Overriding decoder config from checkpoint: {ckpt_model_cfg[k]}")
                model_cfg[k] = ckpt_model_cfg[k]
        config = {**config, 'model': model_cfg}
    del ckpt_probe

    # Create model
    model = PDELoRAModelV3(
        config=config,
        pretrained_path=config['model'].get('pretrained_path'),
    )
    model = model.float()

    # Load checkpoint
    load_lora_checkpoint(model, args.checkpoint, optimizer=None, scheduler=None)
    model = accelerator.prepare(model)

    # Create dataloaders
    train_loader, val_loader, _, _ = create_finetune_dataloaders(
        data_path=config['dataset']['path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        seed=config['dataset']['seed'],
        temporal_length=t_input + 1,
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        clips_per_sample=config['dataset'].get('clips_per_sample', 100),
        vector_dim=config['dataset'].get('vector_dim', 0),
        val_time_interval=config['dataset'].get('val_time_interval', 8),
    )

    loader = val_loader if args.split == 'val' else train_loader

    # Evaluate
    result = evaluate(model, loader, accelerator, t_input, per_timestep=args.per_timestep)

    # Print results
    if accelerator.is_main_process:
        console.print()
        table = Table(
            title=f"nRMSE Evaluation — {Path(args.checkpoint).parent.name}",
            show_header=True,
        )
        table.add_column("Channel", style="cyan")
        table.add_column("nRMSE", style="bold yellow", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("GT RMS", justify="right")

        for ch_idx in sorted(result['per_channel_nrmse'].keys()):
            ch_name = CHANNEL_NAMES.get(ch_idx, f'ch_{ch_idx}')
            nrmse = result['per_channel_nrmse'][ch_idx]
            rmse = result['per_channel_rmse'][ch_idx]
            gt_rms = result['per_channel_gt_rms'][ch_idx]
            table.add_row(
                f"{ch_name} ({ch_idx})",
                f"{nrmse:.6f}",
                f"{rmse:.6e}",
                f"{gt_rms:.6e}",
            )

        table.add_section()
        table.add_row("Overall (avg)", f"{result['overall_nrmse']:.6f}", "", "")

        console.print(table)
        console.print(f"\nSplit: {args.split}, Samples: {result['num_samples']}")
        console.print(f"Checkpoint: {args.checkpoint}")
        console.print(f"nRMSE = sqrt(MSE(pred,gt) / MSE(0,gt))\n")

        # Per-timestep
        if args.per_timestep and 'per_timestep_nrmse' in result:
            ts_table = Table(title="Per-Timestep nRMSE", show_header=True)
            ts_table.add_column("t", style="cyan", justify="right")
            for ch_idx in sorted(result['per_timestep_nrmse'].keys()):
                ch_name = CHANNEL_NAMES.get(ch_idx, f'ch_{ch_idx}')
                ts_table.add_column(ch_name, justify="right")

            channels = sorted(result['per_timestep_nrmse'].keys())
            n_ts = len(result['per_timestep_nrmse'][channels[0]])
            for t in range(n_ts):
                row = [str(t)]
                for ch_idx in channels:
                    row.append(f"{result['per_timestep_nrmse'][ch_idx][t]:.6f}")
                ts_table.add_row(*row)

            console.print(ts_table)


if __name__ == '__main__':
    main()
