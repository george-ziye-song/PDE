"""
GT PDE residual verification for POSEIDON NS-PwC dataset.

Dataset: Incompressible Navier-Stokes with piecewise constant vorticity IC + passive tracer.
Solver: Spectral method (AZEBAN) with hyperviscosity, ν ≈ 4×10⁻⁴
Grid: 128×128, periodic BC, [0,1]², T=1, 21 snapshots (dt=0.05)
Channels: [u_x, u_y, passive_tracer]

PDE equations:
  Divergence:  ∂u_x/∂x + ∂u_y/∂y = 0
  Vorticity:   ∂ω/∂t + u·∇ω = ν∆ω,  ω = ∂u_y/∂x - ∂u_x/∂y
  Tracer:      ∂c/∂t + u·∇c = κ∆c,   κ ≈ ν

Note: Vorticity formulation eliminates pressure, making it ideal for verification.
The spectral solver uses hyperviscosity (Fourier multiplier Q_N), not standard Laplacian
viscosity. We test both standard ν∆ω and also report inviscid residuals for comparison.

References:
  - POSEIDON paper (NeurIPS 2024), Section B.1 and B.2.1
  - Solver: AZEBAN spectral hyperviscosity code
  - ε_N = 0.05/N = 0.05/128 ≈ 3.9×10⁻⁴, ν ≈ 4×10⁻⁴
"""

import argparse
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset as NCDataset
except ImportError:
    NCDataset = None


# =============================================================================
# IMPORTANT: Data layout convention for NS-PwC (NetCDF)
#   velocity shape: (sample, time, channel, x, y)
#   axis -2 = x (rows),  axis -1 = y (columns)
#   So ∂/∂x operates on axis -2, ∂/∂y operates on axis -1
# =============================================================================


def ddx(f: np.ndarray, dx: float) -> np.ndarray:
    """4th-order central FD ∂f/∂x with periodic BC. x = axis -2 (rows)."""
    return (
        -np.roll(f, -2, axis=-2) + 8 * np.roll(f, -1, axis=-2)
        - 8 * np.roll(f, 1, axis=-2) + np.roll(f, 2, axis=-2)
    ) / (12 * dx)


def ddy(f: np.ndarray, dy: float) -> np.ndarray:
    """4th-order central FD ∂f/∂y with periodic BC. y = axis -1 (columns)."""
    return (
        -np.roll(f, -2, axis=-1) + 8 * np.roll(f, -1, axis=-1)
        - 8 * np.roll(f, 1, axis=-1) + np.roll(f, 2, axis=-1)
    ) / (12 * dy)


def fd4_laplacian(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """4th-order central FD Laplacian with periodic BC. x=axis-2, y=axis-1."""
    d2f_dx2 = (
        -np.roll(f, -2, axis=-2) + 16 * np.roll(f, -1, axis=-2)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-2) - np.roll(f, 2, axis=-2)
    ) / (12 * dx ** 2)
    d2f_dy2 = (
        -np.roll(f, -2, axis=-1) + 16 * np.roll(f, -1, axis=-1)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-1) - np.roll(f, 2, axis=-1)
    ) / (12 * dy ** 2)
    return d2f_dx2 + d2f_dy2


def fft_ddx(f: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Spectral ∂f/∂x with periodic BC. x = axis -2."""
    Nx = f.shape[-2]
    kx = np.fft.fftfreq(Nx, d=L / Nx) * 2 * np.pi
    f_hat = np.fft.fft2(f)
    df_hat = 1j * kx[:, None] * f_hat
    return np.real(np.fft.ifft2(df_hat))


def fft_ddy(f: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Spectral ∂f/∂y with periodic BC. y = axis -1."""
    Ny = f.shape[-1]
    ky = np.fft.fftfreq(Ny, d=L / Ny) * 2 * np.pi
    f_hat = np.fft.fft2(f)
    df_hat = 1j * ky[None, :] * f_hat
    return np.real(np.fft.ifft2(df_hat))


def fft_laplacian(f: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Spectral Laplacian with periodic BC."""
    Nx, Ny = f.shape[-2], f.shape[-1]
    kx = np.fft.fftfreq(Nx, d=L / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=L / Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    f_hat = np.fft.fft2(f)
    return np.real(np.fft.ifft2(-k2 * f_hat))


def compute_residuals_fd(
    ux: np.ndarray,
    uy: np.ndarray,
    tracer: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    nu: float,
    kappa: float,
) -> dict[str, np.ndarray]:
    """Compute PDE residuals using 4th-order FD.

    Args:
        ux, uy, tracer: shape (T, H, W)
    Returns:
        dict of residual arrays
    """
    # Divergence (at all timesteps): ∂u_x/∂x + ∂u_y/∂y
    div = ddx(ux, dx) + ddy(uy, dy)

    # Vorticity: ω = ∂u_y/∂x - ∂u_x/∂y
    omega = ddx(uy, dx) - ddy(ux, dy)

    # Time derivatives (2nd-order central, interior frames 1..T-2)
    domega_dt = (omega[2:] - omega[:-2]) / (2 * dt)
    dtracer_dt = (tracer[2:] - tracer[:-2]) / (2 * dt)

    # Mid-frame fields
    ux_m = ux[1:-1]
    uy_m = uy[1:-1]
    omega_m = omega[1:-1]
    tracer_m = tracer[1:-1]

    # Advection: u·∇ω = u_x * ∂ω/∂x + u_y * ∂ω/∂y
    u_grad_omega = ux_m * ddx(omega_m, dx) + uy_m * ddy(omega_m, dy)

    # Diffusion: ν∆ω
    nu_lap_omega = nu * fd4_laplacian(omega_m, dx, dy)

    # Vorticity residual: ∂ω/∂t + u·∇ω - ν∆ω = 0
    R_vort = domega_dt + u_grad_omega - nu_lap_omega

    # Inviscid vorticity residual (for comparison)
    R_vort_inviscid = domega_dt + u_grad_omega

    # Tracer: ∂c/∂t + u·∇c - κ∆c = 0
    u_grad_c = ux_m * ddx(tracer_m, dx) + uy_m * ddy(tracer_m, dy)
    kappa_lap_c = kappa * fd4_laplacian(tracer_m, dx, dy)
    R_tracer = dtracer_dt + u_grad_c - kappa_lap_c
    R_tracer_inviscid = dtracer_dt + u_grad_c

    return {
        "divergence": div,
        "vorticity_ns": R_vort,
        "vorticity_inviscid": R_vort_inviscid,
        "tracer_ns": R_tracer,
        "tracer_inviscid": R_tracer_inviscid,
    }


def compute_residuals_fft(
    ux: np.ndarray,
    uy: np.ndarray,
    tracer: np.ndarray,
    dt: float,
    nu: float,
    kappa: float,
    L: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute PDE residuals using FFT (spectral) derivatives.

    More accurate for spectral-solver-generated data.
    """
    # Divergence: ∂u_x/∂x + ∂u_y/∂y
    div = fft_ddx(ux, L) + fft_ddy(uy, L)

    # Vorticity: ω = ∂u_y/∂x - ∂u_x/∂y
    omega = fft_ddx(uy, L) - fft_ddy(ux, L)

    # Time derivatives
    domega_dt = (omega[2:] - omega[:-2]) / (2 * dt)
    dtracer_dt = (tracer[2:] - tracer[:-2]) / (2 * dt)

    # Mid-frame
    ux_m = ux[1:-1]
    uy_m = uy[1:-1]
    omega_m = omega[1:-1]
    tracer_m = tracer[1:-1]

    # Advection: u·∇ω
    u_grad_omega = ux_m * fft_ddx(omega_m, L) + uy_m * fft_ddy(omega_m, L)
    nu_lap_omega = nu * fft_laplacian(omega_m, L)

    R_vort = domega_dt + u_grad_omega - nu_lap_omega
    R_vort_inviscid = domega_dt + u_grad_omega

    # Tracer
    u_grad_c = ux_m * fft_ddx(tracer_m, L) + uy_m * fft_ddy(tracer_m, L)
    kappa_lap_c = kappa * fft_laplacian(tracer_m, L)
    R_tracer = dtracer_dt + u_grad_c - kappa_lap_c
    R_tracer_inviscid = dtracer_dt + u_grad_c

    return {
        "divergence": div,
        "vorticity_ns": R_vort,
        "vorticity_inviscid": R_vort_inviscid,
        "tracer_ns": R_tracer,
        "tracer_inviscid": R_tracer_inviscid,
    }


def main():
    parser = argparse.ArgumentParser(description="GT PDE residual for NS-PwC")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch-share/SONG0304/finetune/NS-PwC",
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--nu", type=float, default=4e-4, help="Kinematic viscosity")
    parser.add_argument("--kappa", type=float, default=4e-4, help="Tracer diffusivity (≈ν)")
    args = parser.parse_args()

    if NCDataset is None:
        raise ImportError("netCDF4 is required. Install: pip install netCDF4")

    data_dir = Path(args.data_dir)
    nc_files = sorted(data_dir.glob("velocity_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No velocity_*.nc files found in {data_dir}")

    # Use first file
    nc_path = nc_files[0]
    print(f"Loading: {nc_path.name}")

    with NCDataset(str(nc_path), "r") as f:
        n_samples_file = f.dimensions["sample"].size
        n_time = f.dimensions["time"].size
        n_ch = f.dimensions["channel"].size
        nx = f.dimensions["x"].size
        ny = f.dimensions["y"].size
        print(f"  Samples={n_samples_file}, T={n_time}, Ch={n_ch}, Grid={nx}×{ny}")

        # Grid parameters
        L = 1.0
        dx = L / nx
        dy = L / ny
        dt = L / (n_time - 1)  # T=1, 21 steps → dt=0.05
        nu = args.nu
        kappa = args.kappa

        print(f"  dx={dx:.6f}, dt={dt:.4f}, ν={nu:.1e}, κ={kappa:.1e}")

        n_proc = min(n_samples_file, args.max_samples)

        eq_names_fd = ["divergence", "vorticity_ns", "vorticity_inviscid", "tracer_ns", "tracer_inviscid"]
        eq_names_fft = ["divergence", "vorticity_ns", "vorticity_inviscid", "tracer_ns", "tracer_inviscid"]

        all_rms_fd = {k: [] for k in eq_names_fd}
        all_rms_fft = {k: [] for k in eq_names_fft}

        # Per-window accumulators
        window_rms_fd = {w: {k: [] for k in eq_names_fd} for w in ["early", "mid", "late"]}

        for s_idx in range(n_proc):
            # velocity shape: (sample, time, channel, x, y)
            vel = f.variables["velocity"][s_idx].astype(np.float64)  # (T, C, H, W)
            ux = vel[:, 0, :, :]   # (T, H, W)
            uy = vel[:, 1, :, :]
            tracer = vel[:, 2, :, :]

            # FD residuals
            res_fd = compute_residuals_fd(ux, uy, tracer, dt, dx, dy, nu, kappa)
            for eq in eq_names_fd:
                rms = np.sqrt(np.mean(res_fd[eq] ** 2))
                all_rms_fd[eq].append(rms)

            # FFT residuals
            res_fft = compute_residuals_fft(ux, uy, tracer, dt, nu, kappa, L)
            for eq in eq_names_fft:
                rms = np.sqrt(np.mean(res_fft[eq] ** 2))
                all_rms_fft[eq].append(rms)

            # Per-window (FD)
            n_int = res_fd["vorticity_ns"].shape[0]  # 19 interior frames
            early_sl = slice(0, max(1, n_int // 3))
            mid_sl = slice(n_int // 3, 2 * n_int // 3)
            late_sl = slice(2 * n_int // 3, n_int)

            for eq in eq_names_fd:
                r = res_fd[eq]
                if eq == "divergence":
                    # divergence is (T, H, W), use all frames for window
                    T_all = r.shape[0]
                    e_sl = slice(0, max(1, T_all // 3))
                    m_sl = slice(T_all // 3, 2 * T_all // 3)
                    l_sl = slice(2 * T_all // 3, T_all)
                    window_rms_fd["early"][eq].append(np.sqrt(np.mean(r[e_sl] ** 2)))
                    window_rms_fd["mid"][eq].append(np.sqrt(np.mean(r[m_sl] ** 2)))
                    window_rms_fd["late"][eq].append(np.sqrt(np.mean(r[l_sl] ** 2)))
                else:
                    window_rms_fd["early"][eq].append(np.sqrt(np.mean(r[early_sl] ** 2)))
                    window_rms_fd["mid"][eq].append(np.sqrt(np.mean(r[mid_sl] ** 2)))
                    window_rms_fd["late"][eq].append(np.sqrt(np.mean(r[late_sl] ** 2)))

            if s_idx < 3 or s_idx == n_proc - 1:
                print(f"  Sample {s_idx}: "
                      f"div_fd={all_rms_fd['divergence'][-1]:.4e}, "
                      f"vort_fd={all_rms_fd['vorticity_ns'][-1]:.4e}, "
                      f"vort_fft={all_rms_fft['vorticity_ns'][-1]:.4e}, "
                      f"tracer_fd={all_rms_fd['tracer_ns'][-1]:.4e}")

    # Summary
    print("\n" + "=" * 90)
    print(f"Summary over {n_proc} samples (ν={nu:.1e}, κ={kappa:.1e})")
    print("=" * 90)

    print(f"\n--- 4th-order FD residuals ---")
    print(f"{'Equation':<22} | {'Mean RMS':>12} | {'Std RMS':>12} | {'Min RMS':>12} | {'Max RMS':>12}")
    print("-" * 80)
    for eq in eq_names_fd:
        arr = np.array(all_rms_fd[eq])
        print(f"{eq:<22} | {arr.mean():>12.4e} | {arr.std():>12.4e} | "
              f"{arr.min():>12.4e} | {arr.max():>12.4e}")

    print(f"\n--- FFT (spectral) residuals ---")
    print(f"{'Equation':<22} | {'Mean RMS':>12} | {'Std RMS':>12} | {'Min RMS':>12} | {'Max RMS':>12}")
    print("-" * 80)
    for eq in eq_names_fft:
        arr = np.array(all_rms_fft[eq])
        print(f"{eq:<22} | {arr.mean():>12.4e} | {arr.std():>12.4e} | "
              f"{arr.min():>12.4e} | {arr.max():>12.4e}")

    print(f"\n--- Per-window Mean RMS (FD, across {n_proc} samples) ---")
    display_eqs = ["divergence", "vorticity_ns", "tracer_ns"]
    print(f"{'Window':<8} | {'Divergence':>12} | {'Vorticity NS':>12} | {'Tracer NS':>12}")
    print("-" * 55)
    for w in ["early", "mid", "late"]:
        vals = {eq: np.mean(window_rms_fd[w][eq]) for eq in display_eqs}
        print(f"{w:<8} | {vals['divergence']:>12.4e} | {vals['vorticity_ns']:>12.4e} | "
              f"{vals['tracer_ns']:>12.4e}")

    # Improvement from viscosity
    print(f"\n--- Viscosity improvement (inviscid → NS) ---")
    for base in ["vorticity", "tracer"]:
        inv_arr = np.array(all_rms_fd[f"{base}_inviscid"])
        ns_arr = np.array(all_rms_fd[f"{base}_ns"])
        ratio = inv_arr.mean() / ns_arr.mean()
        print(f"  {base}: inviscid={inv_arr.mean():.4e} → NS={ns_arr.mean():.4e} "
              f"(improvement: {ratio:.2f}x)")

    # eq_scales suggestion
    print(f"\n--- Suggested eq_scales (FD, GT Mean RMS) ---")
    for eq in ["divergence", "vorticity_ns", "tracer_ns"]:
        arr = np.array(all_rms_fd[eq])
        print(f"  {eq}: {arr.mean():.4e}")
    print(f"\n--- Suggested eq_scales (FFT, GT Mean RMS) ---")
    for eq in ["divergence", "vorticity_ns", "tracer_ns"]:
        arr = np.array(all_rms_fft[eq])
        print(f"  {eq}: {arr.mean():.4e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
