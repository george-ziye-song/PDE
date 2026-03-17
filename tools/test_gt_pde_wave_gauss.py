"""
Compute eq_scales for Wave-Gauss (acoustic wave equation) on preprocessed H5.

PDE: u_tt = c(x)^2 * nabla^2 u

Loads GT data, feeds through WaveGaussPDELoss with eq_scales=1.0 (no normalization),
reports per-equation RMS = sqrt(mean_MSE) -> use as eq_scales in config.

Usage:
    python tools/test_gt_pde_wave_gauss.py --data ./data/finetune/wave_gauss.h5
"""

import argparse
import h5py
import torch
import numpy as np

from finetune.pde_loss_verified import WaveGaussPDELoss


def parse_args():
    p = argparse.ArgumentParser(description="Compute eq_scales for Wave-Gauss")
    p.add_argument('--data', type=str, default='./data/finetune/wave_gauss.h5')
    p.add_argument('--max_samples', type=int, default=50)
    p.add_argument('--device', type=str, default='cpu')
    # Physics params
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--ny', type=int, default=128)
    p.add_argument('--Lx', type=float, default=1.0)
    p.add_argument('--Ly', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=1.0 / 14)
    p.add_argument('--skip_boundary', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading: {args.data}")
    with h5py.File(args.data, 'r') as f:
        has_scalar = 'scalar' in f
        if has_scalar:
            scl_shape = f['scalar'].shape
            scalar_indices = f['scalar_indices'][:]
            print(f"  scalar shape: {scl_shape}")
            print(f"  scalar_indices: {scalar_indices}")

        n_samples = scl_shape[0]
        n_timesteps = scl_shape[1]
        print(f"  Samples: {n_samples}, Timesteps: {n_timesteps}")

    # PDE loss with eq_scales=1.0 (raw residuals)
    pde_loss_fn = WaveGaussPDELoss(
        nx=args.nx, ny=args.ny,
        Lx=args.Lx, Ly=args.Ly,
        dt=args.dt,
        skip_boundary=args.skip_boundary,
        eq_scales={'wave': 1.0},
        eq_weights={'wave': 1.0},
    )

    n_proc = min(n_samples, args.max_samples)
    T = n_timesteps  # use all frames

    all_wave_mse = []

    print(f"\nProcessing {n_proc} samples, {T} frames each...")

    with h5py.File(args.data, 'r') as f:
        for s_idx in range(n_proc):
            # Load all frames for this sample
            scl = np.array(f['scalar'][s_idx], dtype=np.float32)  # [T, H, W, C_s]

            # scalar[..., 0] = displacement u, scalar[..., 1] = wave speed c
            u_disp = torch.from_numpy(scl[..., 0]).unsqueeze(0).to(device)  # [1, T, H, W]
            c_speed = torch.from_numpy(scl[0, ..., 1]).unsqueeze(0).to(device)  # [1, H, W] (constant across time)

            with torch.no_grad():
                total_loss, losses = pde_loss_fn(u_disp, c_speed)

            all_wave_mse.append(losses['wave'].item())

            if s_idx < 3 or s_idx == n_proc - 1:
                print(f"  Sample {s_idx}: wave_mse={all_wave_mse[-1]:.4e}")

    wave_arr = np.array(all_wave_mse)
    wave_rms = float(np.sqrt(np.mean(wave_arr)))

    print(f"\n{'='*60}")
    print(f"Results over {len(wave_arr)} samples")
    print(f"{'='*60}")
    print(f"  Wave: mean_MSE={np.mean(wave_arr):.4e}, RMS={wave_rms:.4e}")

    print(f"\n--- eq_scales for config ---")
    print(f"  eq_scales:")
    print(f"    wave: {wave_rms:.4e}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
