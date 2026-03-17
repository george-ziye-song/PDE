"""
Kelvin-Helmholtz instability — GT PDE residual verification.

Governing equations (2D compressible Navier-Stokes, conservative form):

    ∂ρ/∂t + ∂(ρVx)/∂x + ∂(ρVy)/∂y = 0                              (Continuity)

    ∂(ρVx)/∂t + ∂(ρVx²+p)/∂x + ∂(ρVxVy)/∂y = ∂τxx/∂x + ∂τxy/∂y    (x-Momentum)

    ∂(ρVy)/∂t + ∂(ρVxVy)/∂x + ∂(ρVy²+p)/∂y = ∂τxy/∂x + ∂τyy/∂y    (y-Momentum)

For nearly incompressible (M=0.1, div V ≈ 0), the viscous term simplifies to:
    viscous_x ≈ μ * ∇²Vx
    viscous_y ≈ μ * ∇²Vy

Parameters:
    M  = 0.1    (Mach number)
    Re = 1000   (Reynolds number)
    γ  = 5/3    (monatomic ideal gas, verified: c²=γp/ρ=100, M=V/c=0.1)
    dk = {1, 2, 10}  (perturbation wavenumber)

Boundary: periodic in both x and y
Grid: 1024×1024, domain [0,1]×[0,1]
Time: 51 snapshots, dt = 0.1

Numerical methods:
    Spatial: 4th-order central FD (periodic, torch.roll)
    Time:    2nd-order central FD

Usage:
    python tools/test_gt_pde_kh.py [path_to_hdf5 ...]
"""

import torch
import numpy as np
import h5py
import sys


# =============================================================================
# FD operators (periodic, 4th-order)
# =============================================================================

def fd4_dx(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order ∂f/∂x, periodic in dim -2."""
    return (-torch.roll(f, -2, -2) + 8 * torch.roll(f, -1, -2)
            - 8 * torch.roll(f, 1, -2) + torch.roll(f, 2, -2)) / (12 * dx)


def fd4_dy(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order ∂f/∂y, periodic in dim -1."""
    return (-torch.roll(f, -2, -1) + 8 * torch.roll(f, -1, -1)
            - 8 * torch.roll(f, 1, -1) + torch.roll(f, 2, -1)) / (12 * dy)


def fd4_d2x(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order ∂²f/∂x², periodic in dim -2."""
    return (-torch.roll(f, -2, -2) + 16 * torch.roll(f, -1, -2)
            - 30 * f + 16 * torch.roll(f, 1, -2)
            - torch.roll(f, 2, -2)) / (12 * dx ** 2)


def fd4_d2y(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order ∂²f/∂y², periodic in dim -1."""
    return (-torch.roll(f, -2, -1) + 16 * torch.roll(f, -1, -1)
            - 30 * f + 16 * torch.roll(f, 1, -1)
            - torch.roll(f, 2, -1)) / (12 * dy ** 2)


def laplacian_4th(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """4th-order Laplacian, periodic."""
    return fd4_d2x(f, dx) + fd4_d2y(f, dy)


# =============================================================================
# Load data
# =============================================================================

def load_kh_data(path: str) -> dict:
    """Load KH HDF5 and return as torch tensors (float64)."""
    with h5py.File(path, 'r') as f:
        rho = torch.from_numpy(f['density'][:]).double()
        p = torch.from_numpy(f['pressure'][:]).double()
        vx = torch.from_numpy(f['Vx'][:]).double()
        vy = torch.from_numpy(f['Vy'][:]).double()
        t = f['t-coordinate'][:]
        x = f['x-coordinate'][:]

        M = float(f.attrs['M'])
        Re = float(f.attrs['Re'])
        dk = float(f.attrs['dk'])

    dt = float(t[1] - t[0])
    dx = float(x[1] - x[0])
    dy = dx  # square grid

    return {
        'rho': rho, 'p': p, 'vx': vx, 'vy': vy,
        'dt': dt, 'dx': dx, 'dy': dy,
        'M': M, 'Re': Re, 'dk': dk,
        'gamma': 5.0 / 3.0,
    }


# =============================================================================
# PDE residuals
# =============================================================================

def compute_euler_residuals(d: dict) -> dict:
    """Compute inviscid (Euler) residuals."""
    rho, p, vx, vy = d['rho'], d['p'], d['vx'], d['vy']
    dx, dy, dt = d['dx'], d['dy'], d['dt']

    rho_vx = rho * vx
    rho_vy = rho * vy

    # Mid-time slices
    rho_m, p_m, vx_m, vy_m = rho[1:-1], p[1:-1], vx[1:-1], vy[1:-1]

    # Time derivatives (2nd-order central)
    drho_dt = (rho[2:] - rho[:-2]) / (2 * dt)
    drhovx_dt = (rho_vx[2:] - rho_vx[:-2]) / (2 * dt)
    drhovy_dt = (rho_vy[2:] - rho_vy[:-2]) / (2 * dt)

    # Continuity
    R_cont = drho_dt + fd4_dx(rho_m * vx_m, dx) + fd4_dy(rho_m * vy_m, dy)

    # x-Momentum (Euler)
    R_xmom = drhovx_dt + fd4_dx(rho_m * vx_m ** 2 + p_m, dx) + fd4_dy(rho_m * vx_m * vy_m, dy)

    # y-Momentum (Euler)
    R_ymom = drhovy_dt + fd4_dx(rho_m * vx_m * vy_m, dx) + fd4_dy(rho_m * vy_m ** 2 + p_m, dy)

    return {'continuity': R_cont, 'x_momentum': R_xmom, 'y_momentum': R_ymom}


def compute_viscous_term(d: dict, mu: float) -> dict:
    """
    Compute viscous terms using direct Laplacian (simplified for low Mach).

    Full compressible viscous divergence for x-momentum:
        μ * [(4/3)∂²Vx/∂x² + ∂²Vx/∂y² + (1/3)∂²Vy/∂x∂y]
    Simplified (div V ≈ 0):
        μ * ∇²Vx

    Returns both full and simplified versions.
    """
    vx_m, vy_m = d['vx'][1:-1], d['vy'][1:-1]
    dx, dy = d['dx'], d['dy']

    # Direct Laplacian (simplified)
    lap_vx = laplacian_4th(vx_m, dx, dy)
    lap_vy = laplacian_4th(vy_m, dx, dy)
    visc_x_simple = mu * lap_vx
    visc_y_simple = mu * lap_vy

    # Full compressible form
    d2vx_dx2 = fd4_d2x(vx_m, dx)
    d2vx_dy2 = fd4_d2y(vx_m, dy)
    d2vy_dx2 = fd4_d2x(vy_m, dx)
    d2vy_dy2 = fd4_d2y(vy_m, dy)
    # Cross derivative: ∂²Vy/∂x∂y via fd4_dx(fd4_dy(...))
    d2vy_dxdy = fd4_dx(fd4_dy(vy_m, dy), dx)
    d2vx_dxdy = fd4_dy(fd4_dx(vx_m, dx), dy)

    visc_x_full = mu * ((4.0 / 3.0) * d2vx_dx2 + d2vx_dy2 + (1.0 / 3.0) * d2vy_dxdy)
    visc_y_full = mu * (d2vy_dx2 + (4.0 / 3.0) * d2vy_dy2 + (1.0 / 3.0) * d2vx_dxdy)

    return {
        'visc_x_simple': visc_x_simple, 'visc_y_simple': visc_y_simple,
        'visc_x_full': visc_x_full, 'visc_y_full': visc_y_full,
    }


def rms(t: torch.Tensor) -> float:
    return t.pow(2).mean().sqrt().item()


def test_mu_values(d: dict, euler: dict):
    """Try different μ scalings and find the best one."""
    M, Re = d['M'], d['Re']

    mu_candidates = {
        '1/Re':       1.0 / Re,
        'M/Re':       M / Re,
        '1/(M*Re)':   1.0 / (M * Re),
        'M²/Re':      M ** 2 / Re,
    }

    print(f"\n--- μ scaling search (x-momentum, Laplacian form) ---")
    print(f"{'μ formula':<12s} | {'μ value':>10s} | {'Euler RMS':>12s} | {'NS simple':>12s} | {'NS full':>12s}")
    print("-" * 75)

    euler_x_rms = rms(euler['x_momentum'])
    best_mu_name = None
    best_rms = euler_x_rms
    best_mu_val = 0.0

    for name, mu_val in mu_candidates.items():
        visc = compute_viscous_term(d, mu_val)
        ns_simple = euler['x_momentum'] - visc['visc_x_simple']
        ns_full = euler['x_momentum'] - visc['visc_x_full']
        rms_simple = rms(ns_simple)
        rms_full = rms(ns_full)

        marker = ""
        if rms_simple < best_rms:
            best_rms = rms_simple
            best_mu_name = name
            best_mu_val = mu_val
            marker = " ← best"

        print(f"{name:<12s} | {mu_val:>10.2e} | {euler_x_rms:>12.4e} | {rms_simple:>12.4e} | {rms_full:>12.4e}{marker}")

    return best_mu_name, best_mu_val


def full_report(d: dict, mu: float, mu_name: str):
    """Full PDE residual report with chosen μ."""
    euler = compute_euler_residuals(d)
    visc = compute_viscous_term(d, mu)

    ns_x = euler['x_momentum'] - visc['visc_x_simple']
    ns_y = euler['y_momentum'] - visc['visc_y_simple']

    print(f"\n--- Full results (μ = {mu_name} = {mu:.2e}) ---")
    print(f"{'Equation':<20s} | {'Euler RMS':>12s} | {'NS (Lap) RMS':>12s} | {'Improvement':>12s}")
    print("-" * 65)

    for eq_name, euler_r, ns_r in [
        ('continuity', euler['continuity'], euler['continuity']),
        ('x_momentum', euler['x_momentum'], ns_x),
        ('y_momentum', euler['y_momentum'], ns_y),
    ]:
        e_rms = rms(euler_r)
        n_rms = rms(ns_r)
        if eq_name == 'continuity':
            print(f"{eq_name:<20s} | {e_rms:>12.4e} | {'(no visc)':>12s} | {'N/A':>12s}")
        else:
            ratio = e_rms / n_rms if n_rms > 0 else float('inf')
            print(f"{eq_name:<20s} | {e_rms:>12.4e} | {n_rms:>12.4e} | {ratio:>11.1f}x")

    # Per-window
    n_frames = euler['continuity'].shape[0]
    mid = n_frames // 2
    windows = [
        ('early (t≈0-1)', slice(0, min(10, n_frames))),
        ('mid (t≈2.5)', slice(max(0, mid - 5), min(n_frames, mid + 5))),
        ('late (t≈4-5)', slice(max(0, n_frames - 10), n_frames)),
    ]

    print(f"\n--- Per-window RMS ---")
    print(f"{'Window':<20s} | {'Continuity':>12s} | {'x-Mom NS':>12s} | {'y-Mom NS':>12s}")
    print("-" * 65)
    for wname, wslice in windows:
        c_rms = rms(euler['continuity'][wslice])
        x_rms = rms(ns_x[wslice])
        y_rms = rms(ns_y[wslice])
        print(f"{wname:<20s} | {c_rms:>12.4e} | {x_rms:>12.4e} | {y_rms:>12.4e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import glob

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob('/scratch-share/SONG0304/finetune/KH_*.hdf5'))

    if not paths:
        print("No KH files found!")
        sys.exit(1)

    for path in paths:
        print(f"\n{'=' * 75}")
        print(f"Dataset: {path.split('/')[-1]}")
        print(f"{'=' * 75}")

        d = load_kh_data(path)
        print(f"M={d['M']}, Re={d['Re']}, dk={d['dk']}, γ={d['gamma']:.4f}")
        print(f"Grid: {d['rho'].shape[1]}×{d['rho'].shape[2]}, dx={d['dx']:.6f}")
        print(f"Time: {d['rho'].shape[0]} frames, dt={d['dt']:.4f}")

        # Verify ideal gas
        c2 = d['gamma'] * d['p'][0].mean().item() / d['rho'][0].mean().item()
        c = np.sqrt(c2)
        M_check = d['vx'][0].abs().max().item() / c
        print(f"Verification: c={c:.2f}, M={M_check:.4f}")

        # Euler residuals
        euler = compute_euler_residuals(d)

        # Search for best μ
        best_mu_name, best_mu_val = test_mu_values(d, euler)
        print(f"\nBest μ: {best_mu_name} = {best_mu_val:.2e}")

        # Full report
        full_report(d, best_mu_val, best_mu_name)

    print(f"\n{'=' * 75}")
    print("Done!")
