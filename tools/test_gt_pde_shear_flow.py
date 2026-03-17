"""
Test PDE residual on GT shear flow data (no model needed).

Reports per-clip PDE loss and flags clips that violate the ~1e-5 constraint.

Usage:
    torchrun --nproc_per_node=8 tools/test_gt_pde_shear_flow.py \
        --config configs/finetune_shear_flow_v3.yaml --batch_size 4

    # single GPU
    python tools/test_gt_pde_shear_flow.py \
        --config configs/finetune_shear_flow_v3.yaml --batch_size 4
"""

import argparse
import yaml
import torch
import numpy as np

from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from finetune.pde_loss_verified import ShearFlowPDELossNPINN

# Channel indices
CH_VX = 0
CH_VY = 1
CH_TRACER = 14
CH_PRESS = 15


def parse_args():
    p = argparse.ArgumentParser(description="Test GT PDE residual (Shear Flow)")
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=1,
                   help='Use 1 for per-clip granularity')
    p.add_argument('--threshold', type=float, default=1e-5,
                   help='PDE loss threshold to flag')
    p.add_argument('--t_input', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    physics = config.get('physics', {})
    pde_loss_fn = ShearFlowPDELossNPINN(
        nx=physics.get('nx', 256),
        ny=physics.get('ny', 512),
        Lx=physics.get('Lx', 1.0),
        Ly=physics.get('Ly', 2.0),
        dt=physics.get('dt', 0.1),
        nu=physics.get('nu', 1e-4),
        D=physics.get('D', 1e-3),
        use_div_correction=True,
    )

    t_input = args.t_input or config.get('dataset', {}).get('t_input', 8)
    temporal_length = t_input + 1

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    val_dataset = FinetuneDataset(
        data_path=config['dataset']['path'],
        temporal_length=temporal_length,
        split='val',
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        seed=config['dataset'].get('seed', 42),
        clips_per_sample=None,
        vector_dim=config['dataset'].get('vector_dim', 2),
        val_time_interval=config['dataset'].get('val_time_interval', 8),
    )

    if is_main:
        print(f"{'=' * 60}")
        print(f"GT PDE Residual Test — Shear Flow")
        print(f"{'=' * 60}")
        print(f"  Val clips:  {len(val_dataset)}")
        print(f"  t_input:    {t_input}")
        print(f"  batch_size: {args.batch_size}")
        print(f"  threshold:  {args.threshold:.0e}")
        print(f"  GPUs:       {accelerator.num_processes}")
        print(f"{'=' * 60}")

    val_sampler = FinetuneSampler(
        val_dataset, args.batch_size, shuffle=False,
        seed=config['dataset'].get('seed', 42),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=finetune_collate_fn,
        num_workers=config.get('dataloader', {}).get('num_workers', 4),
        pin_memory=True,
    )

    val_loader = accelerator.prepare(val_loader)

    # Pre-allocate per-batch storage
    max_batches = len(val_loader)
    local_pde = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_cont = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_u_mom = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_v_mom = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_tracer = torch.full((max_batches,), float('nan'), device=accelerator.device)

    if is_main:
        print(f"\nScanning {len(val_dataset)} clips...")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            data = batch['data'].to(device=accelerator.device, dtype=torch.float32)

            # Full GT sequence: all timesteps
            u = data[:, :, :, :, CH_VX]       # [B, T, H, W]
            v = data[:, :, :, :, CH_VY]
            p = data[:, :, :, :, CH_PRESS]
            s = data[:, :, :, :, CH_TRACER]

            total_loss, losses = pde_loss_fn(u, v, p, s)

            local_pde[i] = total_loss.detach()
            local_cont[i] = losses.get('continuity', torch.tensor(0.0)).detach()
            local_u_mom[i] = losses.get('u_momentum', torch.tensor(0.0)).detach()
            local_v_mom[i] = losses.get('v_momentum', torch.tensor(0.0)).detach()
            local_tracer[i] = losses.get('tracer', torch.tensor(0.0)).detach()

            if is_main and (i + 1) % 50 == 0:
                print(f"  batch {i + 1}/{max_batches}")

    accelerator.wait_for_everyone()
    all_pde = accelerator.gather(local_pde)
    all_cont = accelerator.gather(local_cont)
    all_u_mom = accelerator.gather(local_u_mom)
    all_v_mom = accelerator.gather(local_v_mom)
    all_tracer = accelerator.gather(local_tracer)
    accelerator.wait_for_everyone()

    if is_main:
        valid = ~torch.isnan(all_pde)
        pde_arr = all_pde[valid].cpu().numpy()
        cont_arr = all_cont[valid].cpu().numpy()
        u_mom_arr = all_u_mom[valid].cpu().numpy()
        v_mom_arr = all_v_mom[valid].cpu().numpy()
        tracer_arr = all_tracer[valid].cpu().numpy()

        n = len(pde_arr)
        print(f"\n{'=' * 60}")
        print(f"Results ({n} batches)")
        print(f"{'=' * 60}")
        print(f"  PDE Total:   mean={np.mean(pde_arr):.6e}  median={np.median(pde_arr):.6e}")
        print(f"               min={np.min(pde_arr):.6e}   max={np.max(pde_arr):.6e}")
        print(f"               std={np.std(pde_arr):.6e}")
        print(f"  Continuity:  mean={np.mean(cont_arr):.6e}")
        print(f"  u_Momentum:  mean={np.mean(u_mom_arr):.6e}")
        print(f"  v_Momentum:  mean={np.mean(v_mom_arr):.6e}")
        print(f"  Tracer:      mean={np.mean(tracer_arr):.6e}")

        # Percentiles
        print(f"\n  Percentiles (PDE total):")
        for p_val in [50, 75, 90, 95, 99, 100]:
            print(f"    P{p_val:3d}: {np.percentile(pde_arr, p_val):.6e}")

        # Outliers
        threshold = args.threshold
        outlier_mask = pde_arr > threshold
        n_outliers = outlier_mask.sum()
        print(f"\n  Clips with PDE > {threshold:.0e}: {n_outliers}/{n}")

        if n_outliers > 0:
            outlier_idx = np.where(outlier_mask)[0]
            sorted_idx = outlier_idx[np.argsort(pde_arr[outlier_idx])[::-1]]
            print(f"  Top outliers (batch idx → PDE loss):")
            for idx in sorted_idx[:30]:
                print(f"    batch {idx:4d}: total={pde_arr[idx]:.6e}  "
                      f"cont={cont_arr[idx]:.6e}  u_mom={u_mom_arr[idx]:.6e}  "
                      f"v_mom={v_mom_arr[idx]:.6e}  tracer={tracer_arr[idx]:.6e}")
            if len(sorted_idx) > 30:
                print(f"    ... {len(sorted_idx) - 30} more outliers")

        # eq_scales: RMS = sqrt(mean_MSE) per equation
        print(f"\n  --- eq_scales for config (RMS = sqrt(mean_MSE)) ---")
        print("  physics:")
        print("    eq_scales:")
        for name, arr in [('continuity', cont_arr), ('u_momentum', u_mom_arr),
                          ('v_momentum', v_mom_arr), ('tracer', tracer_arr)]:
            rms = float(np.sqrt(np.mean(arr)))
            print(f"      {name}: {rms:.4e}")

        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
