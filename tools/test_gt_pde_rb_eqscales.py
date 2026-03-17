"""
Compute eq_scales for Rayleigh-Benard using RayleighBenardFullPDELoss on preprocessed H5.

Loads GT data, feeds through PDE loss with eq_scales=1.0 (no normalization),
reports per-equation RMS = sqrt(mean_MSE) → use as eq_scales in config.

Usage:
    python tools/test_gt_pde_rb_eqscales.py --data ./data/finetune/rayleigh_benard_pr1.h5
"""

import argparse
import h5py
import torch
import numpy as np
from typing import Dict

from finetune.pde_loss_verified import RayleighBenardFullPDELoss

# Channel indices in 18-channel layout
CH_VX = 0
CH_VY = 1
CH_BUOY = 3    # scalar[0] = buoyancy
CH_PRESS = 15  # scalar[12] = pressure


def parse_args():
    p = argparse.ArgumentParser(description="Compute eq_scales for Rayleigh-Benard")
    p.add_argument('--data', type=str, default='./data/finetune/rayleigh_benard_pr1.h5')
    p.add_argument('--max_samples', type=int, default=50)
    p.add_argument('--t_input', type=int, default=9, help='Number of consecutive frames (need >=3 for time derivative)')
    p.add_argument('--device', type=str, default='cpu')
    # Physics params (matching config)
    p.add_argument('--nx', type=int, default=512)
    p.add_argument('--ny', type=int, default=128)
    p.add_argument('--Lx', type=float, default=4.0)
    p.add_argument('--Ly', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=0.25)
    p.add_argument('--kappa', type=float, default=1e-5)
    p.add_argument('--skip_bl', type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading: {args.data}")
    with h5py.File(args.data, 'r') as f:
        has_vector = 'vector' in f
        has_scalar = 'scalar' in f
        if has_vector:
            vec_shape = f['vector'].shape
            print(f"  vector shape: {vec_shape}")
        if has_scalar:
            scl_shape = f['scalar'].shape
            scalar_indices = f['scalar_indices'][:]
            print(f"  scalar shape: {scl_shape}")
            print(f"  scalar_indices: {scalar_indices}")

        n_samples = vec_shape[0] if has_vector else scl_shape[0]
        n_timesteps = vec_shape[1] if has_vector else scl_shape[1]
        print(f"  Samples: {n_samples}, Timesteps: {n_timesteps}")

    # PDE loss with eq_scales=1.0 (raw residuals)
    pde_loss_fn = RayleighBenardFullPDELoss(
        nx=args.nx, ny=args.ny,
        Lx=args.Lx, Ly=args.Ly,
        dt=args.dt, kappa=args.kappa,
        skip_bl=args.skip_bl,
        eq_scales={'continuity': 1.0, 'buoyancy': 1.0},
        eq_weights={'continuity': 1.0, 'buoyancy': 1.0},
    )

    n_proc = min(n_samples, args.max_samples)
    T = args.t_input  # frames per clip

    all_cont_mse = []
    all_buoy_mse = []

    print(f"\nProcessing {n_proc} samples, {T} frames each...")

    with h5py.File(args.data, 'r') as f:
        for s_idx in range(n_proc):
            # Load T consecutive frames starting from t=0
            max_start = n_timesteps - T
            # Sample a few clips per sample for robustness
            starts = list(range(0, max_start + 1, max(1, max_start // 3)))[:4]

            for start_t in starts:
                end_t = start_t + T

                # Load channels
                vec = np.array(f['vector'][s_idx, start_t:end_t], dtype=np.float32)  # [T, H, W, 3]
                scl = np.array(f['scalar'][s_idx, start_t:end_t], dtype=np.float32)  # [T, H, W, C_s]

                ux = torch.from_numpy(vec[..., 0]).unsqueeze(0).to(device)   # [1, T, H, W]
                uy = torch.from_numpy(vec[..., 1]).unsqueeze(0).to(device)   # [1, T, H, W]

                # Find buoyancy in scalar_indices
                buoy_local_idx = np.where(scalar_indices == 0)[0]  # buoyancy = scalar index 0
                if len(buoy_local_idx) == 0:
                    raise ValueError(f"Buoyancy (scalar_index=0) not found in scalar_indices={scalar_indices}")
                buoy_local_idx = buoy_local_idx[0]
                b = torch.from_numpy(scl[..., buoy_local_idx]).unsqueeze(0).to(device)  # [1, T, H, W]

                with torch.no_grad():
                    total_loss, losses = pde_loss_fn(ux, uy, b)

                all_cont_mse.append(losses['continuity'].item())
                all_buoy_mse.append(losses['buoyancy'].item())

            if s_idx < 3 or s_idx == n_proc - 1:
                print(f"  Sample {s_idx}: cont_mse={all_cont_mse[-1]:.4e}, buoy_mse={all_buoy_mse[-1]:.4e}")

    cont_arr = np.array(all_cont_mse)
    buoy_arr = np.array(all_buoy_mse)

    cont_rms = float(np.sqrt(np.mean(cont_arr)))
    buoy_rms = float(np.sqrt(np.mean(buoy_arr)))

    print(f"\n{'='*60}")
    print(f"Results over {len(cont_arr)} clips from {n_proc} samples")
    print(f"{'='*60}")
    print(f"  Continuity: mean_MSE={np.mean(cont_arr):.4e}, RMS={cont_rms:.4e}")
    print(f"  Buoyancy:   mean_MSE={np.mean(buoy_arr):.4e}, RMS={buoy_rms:.4e}")

    print(f"\n--- eq_scales for config ---")
    print(f"  eq_scales:")
    print(f"    continuity: {cont_rms:.4e}")
    print(f"    buoyancy: {buoy_rms:.4e}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
