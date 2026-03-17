"""
Turbulent Radiative Layer 2D — GT PDE residual verification.

Verifies the TurbulentRadiativePDELoss class against ground-truth data
from the preprocessed HDF5 file (turbulent_radiative_2d.hdf5).

Compressible Euler equations (mass + momentum conservation):
    mass:   dρ/dt + d(ρvx)/dx + d(ρvy)/dy = 0
    x-mom:  d(ρvx)/dt + d(ρvx²+P)/dx + d(ρvxvy)/dy = 0
    y-mom:  d(ρvy)/dt + d(ρvxvy)/dx + d(ρvy²+P)/dy = 0

HDF5 format:
    vector: (N, T, H, W, 3) — vx=vector[...,0], vy=vector[...,1]
    scalar: (N, T, H, W, 2) — density=scalar[...,0], pressure=scalar[...,1]
    scalar_indices: [4, 12]
    nu: (N,) storing tcool

Grid: 128x384 (Nx=128, Ny=384), Lx=1.0, Ly=3.0
BCs: x periodic, y open (Neumann)

Usage:
    python tools/test_gt_pde_turbulent_radiative_2d.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import h5py
import numpy as np
from finetune.pde_loss_verified import TurbulentRadiativePDELoss


def main():
    data_path = "/scratch-share/SONG0304/finetune/turbulent_radiative_2d.hdf5"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Physics parameters
    nx = 128
    ny = 384
    Lx = 1.0
    Ly = 3.0
    dt = 1.597033
    skip_bl = 10

    print(f"Loading data from {data_path}")
    print(f"Physics: nx={nx}, ny={ny}, Lx={Lx}, Ly={Ly}, dt={dt}, skip_bl={skip_bl}")
    print(f"Device: {device}")
    print()

    # Load data
    with h5py.File(data_path, 'r') as f:
        vector = f['vector'][:]   # (N, T, H, W, 3)
        scalar = f['scalar'][:]   # (N, T, H, W, 2)
        nu_arr = f['nu'][:]       # (N,) — tcool values
        print(f"vector shape: {vector.shape}")
        print(f"scalar shape: {scalar.shape}")
        print(f"nu (tcool) values: {nu_arr}")

    N, T, H, W = vector.shape[:4]
    print(f"N={N}, T={T}, H={H}, W={W}")
    print()

    # Channel mapping
    # vx = vector[..., 0], vy = vector[..., 1]
    # density = scalar[..., 0], pressure = scalar[..., 1]

    # Create PDE loss function
    pde_loss_fn = TurbulentRadiativePDELoss(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, skip_bl=skip_bl,
    )

    # Accumulate per-equation RMS residuals across all samples
    all_mass_rms = []
    all_xmom_rms = []
    all_ymom_rms = []

    for sample_idx in range(N):
        print(f"{'='*70}")
        print(f"Sample {sample_idx}, tcool={nu_arr[sample_idx]:.6f}")
        print(f"{'='*70}")

        # Extract fields: [T, H, W] -> [1, T, H, W] for batch dim
        vx = torch.from_numpy(vector[sample_idx, :, :, :, 0]).float().unsqueeze(0).to(device)
        vy = torch.from_numpy(vector[sample_idx, :, :, :, 1]).float().unsqueeze(0).to(device)
        rho = torch.from_numpy(scalar[sample_idx, :, :, :, 0]).float().unsqueeze(0).to(device)
        P = torch.from_numpy(scalar[sample_idx, :, :, :, 1]).float().unsqueeze(0).to(device)

        print(f"  vx:  shape={vx.shape}, range=[{vx.min():.4f}, {vx.max():.4f}]")
        print(f"  vy:  shape={vy.shape}, range=[{vy.min():.4f}, {vy.max():.4f}]")
        print(f"  rho: shape={rho.shape}, range=[{rho.min():.4f}, {rho.max():.4f}]")
        print(f"  P:   shape={P.shape}, range=[{P.min():.4f}, {P.max():.4f}]")

        # PDE loss expects [B, T, Nx, Ny] where Nx=128 (periodic), Ny=384 (open)
        # The stored data is [B, T, H, W] = [B, T, 128, 384] or [B, T, 384, 128]
        # Need to check which is correct.
        # If H=128, W=384, then data is already [B, T, Nx, Ny] = [B, T, 128, 384]
        # If H=384, W=128, we need transpose.
        if H == nx and W == ny:
            # Data is [B, T, 128, 384] = [B, T, Nx, Ny] — correct
            pass
        elif H == ny and W == nx:
            # Data is [B, T, 384, 128] — need to transpose to [B, T, 128, 384]
            vx = vx.transpose(-1, -2)
            vy = vy.transpose(-1, -2)
            rho = rho.transpose(-1, -2)
            P = P.transpose(-1, -2)
            print(f"  [Transposed spatial dims: {H}x{W} -> {W}x{H}]")
        else:
            print(f"  WARNING: unexpected grid dimensions H={H}, W={W}")

        # Run PDE loss on full temporal extent
        # Use sliding windows of size ~20 frames to keep memory manageable
        window_size = min(T, 20)
        stride = window_size - 2  # overlap of 2 for time derivative continuity

        sample_mass_mse_sum = 0.0
        sample_xmom_mse_sum = 0.0
        sample_ymom_mse_sum = 0.0
        n_windows = 0

        t_start = 0
        while t_start + 3 <= T:  # need at least 3 frames for 2nd-order time derivative
            t_end = min(t_start + window_size, T)
            if t_end - t_start < 3:
                break

            vx_w = vx[:, t_start:t_end]
            vy_w = vy[:, t_start:t_end]
            rho_w = rho[:, t_start:t_end]
            P_w = P[:, t_start:t_end]

            with torch.no_grad():
                total_loss, losses = pde_loss_fn(vx_w, vy_w, rho_w, P_w)

            sample_mass_mse_sum += losses['mass'].item()
            sample_xmom_mse_sum += losses['x_momentum'].item()
            sample_ymom_mse_sum += losses['y_momentum'].item()
            n_windows += 1

            t_start += stride
            if t_start + 3 > T:
                break

        if n_windows > 0:
            avg_mass_mse = sample_mass_mse_sum / n_windows
            avg_xmom_mse = sample_xmom_mse_sum / n_windows
            avg_ymom_mse = sample_ymom_mse_sum / n_windows

            mass_rms = np.sqrt(avg_mass_mse)
            xmom_rms = np.sqrt(avg_xmom_mse)
            ymom_rms = np.sqrt(avg_ymom_mse)

            all_mass_rms.append(mass_rms)
            all_xmom_rms.append(xmom_rms)
            all_ymom_rms.append(ymom_rms)

            print(f"\n  Per-equation residuals (MSE, then RMS):")
            print(f"    Mass:       MSE={avg_mass_mse:.6e}  RMS={mass_rms:.6e}")
            print(f"    x-Momentum: MSE={avg_xmom_mse:.6e}  RMS={xmom_rms:.6e}")
            print(f"    y-Momentum: MSE={avg_ymom_mse:.6e}  RMS={ymom_rms:.6e}")
            print(f"    Total MSE:  {avg_mass_mse + avg_xmom_mse + avg_ymom_mse:.6e}")
        print()

    # Summary across all samples
    print(f"\n{'='*70}")
    print(f"SUMMARY (across {N} samples)")
    print(f"{'='*70}")

    if len(all_mass_rms) > 0:
        mass_rms_mean = np.mean(all_mass_rms)
        xmom_rms_mean = np.mean(all_xmom_rms)
        ymom_rms_mean = np.mean(all_ymom_rms)

        print(f"\nMean RMS residuals (for eq_scales):")
        print(f"  mass:       {mass_rms_mean:.6e}")
        print(f"  x_momentum: {xmom_rms_mean:.6e}")
        print(f"  y_momentum: {ymom_rms_mean:.6e}")

        print(f"\nSuggested eq_scales for config YAML:")
        print(f"  eq_scales:")
        print(f"    mass: {mass_rms_mean:.4f}")
        print(f"    x_momentum: {xmom_rms_mean:.4f}")
        print(f"    y_momentum: {ymom_rms_mean:.4f}")

        print(f"\nPer-sample breakdown:")
        for i in range(len(all_mass_rms)):
            print(f"  Sample {i}: mass={all_mass_rms[i]:.6e}, "
                  f"x_mom={all_xmom_rms[i]:.6e}, y_mom={all_ymom_rms[i]:.6e}")


if __name__ == '__main__':
    main()
