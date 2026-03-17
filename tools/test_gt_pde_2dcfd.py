"""
GT PDE residual verification for PDEBench 2D CFD (Compressible Navier-Stokes).

Dataset: 2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
Equations (conservative form, Euler since η≈ζ≈0):
  Continuity:  ∂ρ/∂t + ∂(ρVx)/∂x + ∂(ρVy)/∂y = 0
  x-Momentum:  ∂(ρVx)/∂t + ∂(ρVx²+p)/∂x + ∂(ρVxVy)/∂y = 0
  y-Momentum:  ∂(ρVy)/∂t + ∂(ρVxVy)/∂x + ∂(ρVy²+p)/∂y = 0
  Energy:      ∂E/∂t + ∂((E+p)Vx)/∂x + ∂((E+p)Vy)/∂y = 0
               E = p/(γ-1) + 0.5*ρ*(Vx²+Vy²), γ=5/3

Parameters: γ=5/3, η=ζ=1e-8 (inviscid), M₀=1.0, periodic BC.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def fd4_dx(f: np.ndarray, dx: float) -> np.ndarray:
    """4th-order central FD ∂f/∂x with periodic BC. f: (..., H, W)."""
    return (
        -np.roll(f, -2, axis=-1) + 8 * np.roll(f, -1, axis=-1)
        - 8 * np.roll(f, 1, axis=-1) + np.roll(f, 2, axis=-1)
    ) / (12 * dx)


def fd4_dy(f: np.ndarray, dy: float) -> np.ndarray:
    """4th-order central FD ∂f/∂y with periodic BC. f: (..., H, W)."""
    return (
        -np.roll(f, -2, axis=-2) + 8 * np.roll(f, -1, axis=-2)
        - 8 * np.roll(f, 1, axis=-2) + np.roll(f, 2, axis=-2)
    ) / (12 * dy)


def compute_residuals(
    rho: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    p: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    gamma: float = 5.0 / 3.0,
) -> dict[str, np.ndarray]:
    """Compute Euler equation residuals.

    Args:
        rho, vx, vy, p: shape (T, H, W)
        dt, dx, dy: grid spacing
        gamma: adiabatic index

    Returns:
        dict of residual arrays, shape (T-2, H, W) (interior time frames)
    """
    # Conserved variables
    rho_vx = rho * vx
    rho_vy = rho * vy
    energy = p / (gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2)

    # Time derivatives (2nd-order central, interior frames 1..T-2)
    drho_dt = (rho[2:] - rho[:-2]) / (2 * dt)
    drhovx_dt = (rho_vx[2:] - rho_vx[:-2]) / (2 * dt)
    drhovy_dt = (rho_vy[2:] - rho_vy[:-2]) / (2 * dt)
    dE_dt = (energy[2:] - energy[:-2]) / (2 * dt)

    # Mid-frame fields for spatial derivatives
    rho_m = rho[1:-1]
    vx_m = vx[1:-1]
    vy_m = vy[1:-1]
    p_m = p[1:-1]
    rho_vx_m = rho_vx[1:-1]
    rho_vy_m = rho_vy[1:-1]
    E_m = energy[1:-1]

    # Spatial fluxes
    # Continuity: ∂(ρVx)/∂x + ∂(ρVy)/∂y
    R_cont = drho_dt + fd4_dx(rho_vx_m, dx) + fd4_dy(rho_vy_m, dy)

    # x-Momentum: ∂(ρVx²+p)/∂x + ∂(ρVxVy)/∂y
    R_xmom = drhovx_dt + fd4_dx(rho_m * vx_m ** 2 + p_m, dx) + fd4_dy(rho_vx_m * vy_m, dy)

    # y-Momentum: ∂(ρVxVy)/∂x + ∂(ρVy²+p)/∂y
    R_ymom = drhovy_dt + fd4_dx(rho_vx_m * vy_m, dx) + fd4_dy(rho_m * vy_m ** 2 + p_m, dy)

    # Energy: ∂((E+p)Vx)/∂x + ∂((E+p)Vy)/∂y
    Ep = E_m + p_m
    R_energy = dE_dt + fd4_dx(Ep * vx_m, dx) + fd4_dy(Ep * vy_m, dy)

    return {
        "continuity": R_cont,
        "x_momentum": R_xmom,
        "y_momentum": R_ymom,
        "energy": R_energy,
    }


def main():
    parser = argparse.ArgumentParser(description="GT PDE residual for 2D CFD")
    parser.add_argument(
        "--data",
        type=str,
        default="/scratch-share/SONG0304/finetune/2D_CFD_M1_subset.hdf5",
    )
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples to process")
    args = parser.parse_args()

    with h5py.File(args.data, "r") as f:
        gamma = 5.0 / 3.0
        eta = f.attrs.get("eta", 1e-8)
        zeta = f.attrs.get("zeta", 1e-8)
        mach = f.attrs.get("M", 1.0)

        t_coords = f["t-coordinate"][:]
        x_coords = f["x-coordinate"][:]
        n_samples = f["density"].shape[0]
        n_time = f["density"].shape[1]
        nx = f["density"].shape[2]

        dt = float(t_coords[1] - t_coords[0])
        dx = float(x_coords[1] - x_coords[0])
        dy = dx  # uniform grid

        print(f"Dataset: {Path(args.data).name}")
        print(f"  M={mach}, η={eta}, ζ={zeta}, γ={gamma:.4f}")
        print(f"  Samples={n_samples}, T={n_time}, Grid={nx}×{nx}")
        print(f"  dt={dt:.4f}, dx={dx:.6f}")
        print(f"  CFL (approx): see per-sample output\n")

        n_proc = min(n_samples, args.max_samples)

        # Accumulators for per-equation RMS across all samples
        eq_names = ["continuity", "x_momentum", "y_momentum", "energy"]
        all_rms = {k: [] for k in eq_names}
        # Per-window accumulators
        window_rms = {w: {k: [] for k in eq_names} for w in ["early", "mid", "late"]}

        for s_idx in range(n_proc):
            rho = f["density"][s_idx].astype(np.float64)   # (T, H, W)
            vx_arr = f["Vx"][s_idx].astype(np.float64)
            vy_arr = f["Vy"][s_idx].astype(np.float64)
            p_arr = f["pressure"][s_idx].astype(np.float64)

            residuals = compute_residuals(rho, vx_arr, vy_arr, p_arr, dt, dx, dy, gamma)

            for eq in eq_names:
                rms = np.sqrt(np.mean(residuals[eq] ** 2))
                all_rms[eq].append(rms)

            # Time windows (residuals have T-2 frames = 19 frames, t_idx 1..19)
            n_int = residuals["continuity"].shape[0]  # 19
            early_sl = slice(0, max(1, n_int // 3))
            mid_sl = slice(n_int // 3, 2 * n_int // 3)
            late_sl = slice(2 * n_int // 3, n_int)

            for eq in eq_names:
                window_rms["early"][eq].append(np.sqrt(np.mean(residuals[eq][early_sl] ** 2)))
                window_rms["mid"][eq].append(np.sqrt(np.mean(residuals[eq][mid_sl] ** 2)))
                window_rms["late"][eq].append(np.sqrt(np.mean(residuals[eq][late_sl] ** 2)))

            if s_idx < 3 or s_idx == n_proc - 1:
                cs_max = np.sqrt(gamma * p_arr.max() / rho.min())
                v_max = max(np.abs(vx_arr).max(), np.abs(vy_arr).max())
                cfl = (v_max + cs_max) * dt / dx
                print(f"  Sample {s_idx}: CFL={cfl:.1f}, "
                      f"cont={all_rms['continuity'][-1]:.4e}, "
                      f"xmom={all_rms['x_momentum'][-1]:.4e}, "
                      f"ymom={all_rms['y_momentum'][-1]:.4e}, "
                      f"energy={all_rms['energy'][-1]:.4e}")

    # Summary
    print("\n" + "=" * 80)
    print(f"Summary over {n_proc} samples")
    print("=" * 80)

    print(f"\n{'Equation':<16} | {'Mean RMS':>12} | {'Std RMS':>12} | {'Min RMS':>12} | {'Max RMS':>12}")
    print("-" * 75)
    for eq in eq_names:
        arr = np.array(all_rms[eq])
        print(f"{eq:<16} | {arr.mean():>12.4e} | {arr.std():>12.4e} | "
              f"{arr.min():>12.4e} | {arr.max():>12.4e}")

    print(f"\n--- Per-window Mean RMS (across {n_proc} samples) ---")
    print(f"{'Window':<8} | {'Continuity':>12} | {'x-Momentum':>12} | {'y-Momentum':>12} | {'Energy':>12}")
    print("-" * 67)
    for w in ["early", "mid", "late"]:
        vals = {eq: np.mean(window_rms[w][eq]) for eq in eq_names}
        print(f"{w:<8} | {vals['continuity']:>12.4e} | {vals['x_momentum']:>12.4e} | "
              f"{vals['y_momentum']:>12.4e} | {vals['energy']:>12.4e}")

    # eq_scales suggestion (for PDE loss normalization)
    print(f"\n--- Suggested eq_scales (GT Mean RMS) ---")
    for eq in eq_names:
        arr = np.array(all_rms[eq])
        print(f"  {eq}: {arr.mean():.4e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
