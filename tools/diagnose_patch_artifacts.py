"""
Diagnose patch boundary artifacts: is the problem BETWEEN patches or WITHIN?

Computes pixel-to-pixel curvature (2nd derivative) everywhere, then compares:
  - Curvature at patch boundaries (every P-th pixel)
  - Curvature at interior positions

If boundary_curvature >> interior_curvature → problem is between patches
If they're similar → problem is within patches (e.g., ConvTranspose2d checkerboard)
"""

import argparse
import torch
import yaml
import numpy as np
from pathlib import Path

from finetune.dataset_finetune import create_finetune_dataloaders
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint


def compute_curvature_map(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute absolute 2nd derivative at every pixel.
    x: [H, W]
    Returns: (h_curv [H, W-2], v_curv [H-2, W])
    """
    # Horizontal: d²f/dx² ≈ f[i-1] - 2f[i] + f[i+1]
    h_curv = (x[:, :-2] - 2 * x[:, 1:-1] + x[:, 2:]).abs()
    # Vertical: d²f/dy²
    v_curv = (x[:-2, :] - 2 * x[1:-1, :] + x[2:, :]).abs()
    return h_curv, v_curv


def analyze_curvature_by_position(curv: torch.Tensor, patch_size: int, axis: str) -> dict:
    """
    Split curvature into boundary vs interior.
    curv: [H, W-2] for horizontal or [H-2, W] for vertical
    axis: 'h' or 'v'
    """
    P = patch_size
    if axis == 'h':
        # curv has W-2 columns. Original pixel i maps to curv column i-1.
        # Boundary positions: pixels P-1, P, 2P-1, 2P, ... (edges of patches)
        # In curv indexing: P-2, P-1, 2P-2, 2P-1, ...
        n_cols = curv.shape[1]
        boundary_mask = torch.zeros(n_cols, dtype=torch.bool)
        for b in range(P - 2, n_cols, P):
            # Mark pixels near boundary: P-2, P-1 (= boundary pixel ± 1 in curv space)
            for offset in range(-1, 2):
                idx = b + offset
                if 0 <= idx < n_cols:
                    boundary_mask[idx] = True
        boundary_vals = curv[:, boundary_mask]
        interior_vals = curv[:, ~boundary_mask]
    else:  # 'v'
        n_rows = curv.shape[0]
        boundary_mask = torch.zeros(n_rows, dtype=torch.bool)
        for b in range(P - 2, n_rows, P):
            for offset in range(-1, 2):
                idx = b + offset
                if 0 <= idx < n_rows:
                    boundary_mask[idx] = True
        boundary_vals = curv[boundary_mask, :]
        interior_vals = curv[~boundary_mask, :]

    return {
        'boundary_mean': boundary_vals.mean().item(),
        'boundary_median': boundary_vals.median().item(),
        'boundary_p95': boundary_vals.quantile(0.95).item(),
        'interior_mean': interior_vals.mean().item(),
        'interior_median': interior_vals.median().item(),
        'interior_p95': interior_vals.quantile(0.95).item(),
        'ratio_mean': boundary_vals.mean().item() / (interior_vals.mean().item() + 1e-10),
        'ratio_p95': boundary_vals.quantile(0.95).item() / (interior_vals.quantile(0.95).item() + 1e-10),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    patch_size = config['model'].get('patch_size', 16)
    t_input = config['dataset'].get('t_input', 8)
    device = torch.device(args.device)

    # Load model
    # Disable smoother to see raw decoder output
    config_no_smoother = {**config, 'model': {**config['model'], 'patch_smoother': {'enabled': False}}}
    model_raw = PDELoRAModelV3(config=config_no_smoother, pretrained_path=config['model'].get('pretrained_path'))
    load_lora_checkpoint(model_raw, args.checkpoint, optimizer=None, scheduler=None)
    model_raw = model_raw.to(device).float().eval()

    # Load model WITH smoother
    model_smooth = PDELoRAModelV3(config=config, pretrained_path=config['model'].get('pretrained_path'))
    load_lora_checkpoint(model_smooth, args.checkpoint, optimizer=None, scheduler=None)
    model_smooth = model_smooth.to(device).float().eval()

    # Load data
    _, val_loader, _, _ = create_finetune_dataloaders(
        data_path=config['dataset']['path'],
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        seed=config['dataset']['seed'],
        temporal_length=t_input + 1,
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        clips_per_sample=config['dataset'].get('clips_per_sample', 100),
        vector_dim=config['dataset'].get('vector_dim', 0),
        val_time_interval=config['dataset'].get('val_time_interval', 8),
    )

    # Get channel mask from first batch
    first_batch = next(iter(val_loader))
    channel_mask = first_batch['channel_mask']
    valid_ch = torch.where(channel_mask[0] > 0)[0]
    print(f"Valid channels: {valid_ch.tolist()}")
    print(f"Patch size: {patch_size}")
    print(f"=" * 70)

    all_results = {'raw': [], 'smoothed': [], 'gt': []}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_samples:
                break

            data = batch['data'].to(device=device, dtype=torch.float32)
            input_data = data[:, :t_input]
            target = data[:, 1:t_input + 1]  # GT

            # Raw prediction (no smoother)
            out_raw_norm, mean, std = model_raw(input_data, return_normalized=True)
            pred_raw = out_raw_norm * std + mean

            # Smoothed prediction
            out_sm_norm, mean, std = model_smooth(input_data, return_normalized=True)
            pred_smooth = out_sm_norm * std + mean

            # Analyze middle timestep, each valid channel
            t_mid = t_input // 2
            for tag, tensor in [('raw', pred_raw), ('smoothed', pred_smooth), ('gt', target)]:
                for ch_idx in valid_ch:
                    field = tensor[0, t_mid, :, :, ch_idx].cpu()
                    h_curv, v_curv = compute_curvature_map(field)
                    h_stats = analyze_curvature_by_position(h_curv, patch_size, 'h')
                    v_stats = analyze_curvature_by_position(v_curv, patch_size, 'v')
                    all_results[tag].append({
                        'sample': i, 'channel': ch_idx.item(),
                        'h': h_stats, 'v': v_stats,
                    })

    # Aggregate and print
    for tag in ['gt', 'raw', 'smoothed']:
        print(f"\n{'=' * 70}")
        print(f"  {tag.upper()} — Curvature Analysis (|f[i-1] - 2f[i] + f[i+1]|)")
        print(f"{'=' * 70}")

        h_ratios_mean, h_ratios_p95 = [], []
        v_ratios_mean, v_ratios_p95 = [], []
        h_bnd_means, h_int_means = [], []
        v_bnd_means, v_int_means = [], []

        for r in all_results[tag]:
            h_ratios_mean.append(r['h']['ratio_mean'])
            h_ratios_p95.append(r['h']['ratio_p95'])
            v_ratios_mean.append(r['v']['ratio_mean'])
            v_ratios_p95.append(r['v']['ratio_p95'])
            h_bnd_means.append(r['h']['boundary_mean'])
            h_int_means.append(r['h']['interior_mean'])
            v_bnd_means.append(r['v']['boundary_mean'])
            v_int_means.append(r['v']['interior_mean'])

        print(f"\n  Horizontal (W-axis):")
        print(f"    Boundary curvature (mean): {np.mean(h_bnd_means):.6e}")
        print(f"    Interior curvature (mean): {np.mean(h_int_means):.6e}")
        print(f"    Ratio (boundary/interior): mean={np.mean(h_ratios_mean):.2f}x, p95={np.mean(h_ratios_p95):.2f}x")

        print(f"\n  Vertical (H-axis):")
        print(f"    Boundary curvature (mean): {np.mean(v_bnd_means):.6e}")
        print(f"    Interior curvature (mean): {np.mean(v_int_means):.6e}")
        print(f"    Ratio (boundary/interior): mean={np.mean(v_ratios_mean):.2f}x, p95={np.mean(v_ratios_p95):.2f}x")

    # Per-sample details
    print(f"\n{'=' * 70}")
    print(f"  Per-sample boundary/interior ratio (mean curvature)")
    print(f"{'=' * 70}")
    print(f"  {'Sample':>6} {'Ch':>4}   {'GT_h':>6} {'GT_v':>6}   {'Raw_h':>6} {'Raw_v':>6}   {'Sm_h':>6} {'Sm_v':>6}")
    for i in range(len(all_results['gt'])):
        gt_r = all_results['gt'][i]
        raw_r = all_results['raw'][i]
        sm_r = all_results['smoothed'][i]
        print(f"  {gt_r['sample']:>6d} {gt_r['channel']:>4d}   "
              f"{gt_r['h']['ratio_mean']:>6.2f} {gt_r['v']['ratio_mean']:>6.2f}   "
              f"{raw_r['h']['ratio_mean']:>6.2f} {raw_r['v']['ratio_mean']:>6.2f}   "
              f"{sm_r['h']['ratio_mean']:>6.2f} {sm_r['v']['ratio_mean']:>6.2f}")


if __name__ == '__main__':
    main()
