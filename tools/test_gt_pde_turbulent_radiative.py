"""
Turbulent Radiative Layer 2D — PDE residual verification.

Compressible Euler + radiative cooling:
    ∂ρ/∂t + ∇·(ρv⃗) = 0                     (mass)
    ∂(ρv⃗)/∂t + ∇·(ρv⃗⊗v⃗ + PI) = 0          (momentum)
    ∂E/∂t + ∇·((E+P)v⃗) = -E/t_cool         (energy)
    E = P/(γ-1),  γ = 5/3

BCs: x periodic, y open
Grid: 128×384, x∈[-0.5,0.5], y∈[-1,2]
Solver: Athena++ (Godunov FV) → conservation form is natural
"""

import torch
import numpy as np
import h5py
from typing import Dict


def load_trl_data(
    path: str,
    sample_idx: int = 0,
    t_start: int = 10,
    t_end: int = 30,
) -> dict:
    f = h5py.File(path, 'r')
    x = f['dimensions/x'][:]
    y = f['dimensions/y'][:]
    t = f['dimensions/time'][t_start:t_end]

    rho = torch.from_numpy(f['t0_fields/density'][sample_idx, t_start:t_end]).double()
    P = torch.from_numpy(f['t0_fields/pressure'][sample_idx, t_start:t_end]).double()
    vel = torch.from_numpy(f['t1_fields/velocity'][sample_idx, t_start:t_end]).double()
    vx = vel[..., 0]
    vy = vel[..., 1]

    tcool = float(f.attrs['tcool'])
    f.close()

    Nx, Ny = len(x), len(y)
    Lx = 1.0  # [-0.5, 0.5]
    Ly = 3.0  # [-1, 2]
    dx = Lx / Nx   # periodic: dx = Lx/Nx
    dy = Ly / (Ny - 1)  # open: dy = Ly/(Ny-1)
    dt = float(t[1] - t[0])
    gamma = 5.0 / 3.0

    return {
        'rho': rho, 'P': P, 'vx': vx, 'vy': vy,
        'x': x, 'y': y, 't': t,
        'dx': dx, 'dy': dy, 'dt': dt,
        'Lx': Lx, 'Ly': Ly,
        'Nx': Nx, 'Ny': Ny,
        'tcool': tcool, 'gamma': gamma,
    }


# ============================================================
# FD operators
# ============================================================

def fd4_dx_periodic(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order periodic x-derivative. f: [..., Nx, Ny]"""
    return (-torch.roll(f, -2, -2) + 8*torch.roll(f, -1, -2)
            - 8*torch.roll(f, 1, -2) + torch.roll(f, 2, -2)) / (12 * dx)


def central_dy(f: torch.Tensor, dy: float) -> torch.Tensor:
    """2nd-order central y-derivative (interior only, Ny-2 points)."""
    return (f[..., 2:] - f[..., :-2]) / (2 * dy)


# ============================================================
# n-PINN (FVM face-flux) operators
# ============================================================

def face_flux_dx_periodic(flux: torch.Tensor, dx: float) -> torch.Tensor:
    """
    ∂F/∂x via face flux (periodic).
    F_e = 0.5*(F[i] + F[i+1]),  ∂F/∂x ≈ (F_e - F_w) / dx
    """
    F_e = 0.5 * (flux + torch.roll(flux, -1, -2))
    F_w = 0.5 * (flux + torch.roll(flux, 1, -2))
    return (F_e - F_w) / dx


def face_flux_dy(flux: torch.Tensor, dy: float) -> torch.Tensor:
    """
    ∂G/∂y via face flux (interior only).
    G_n[j] = 0.5*(G[j] + G[j+1])
    ∂G/∂y[j] ≈ (G_n[j] - G_n[j-1]) / dy  → Ny-2 interior pts
    """
    G_n = 0.5 * (flux[..., 1:] + flux[..., :-1])  # Ny-1 faces
    return (G_n[..., 1:] - G_n[..., :-1]) / dy     # Ny-2 cells


# ============================================================
# PDE residuals
# ============================================================

def mass_residual_fd(d: dict) -> torch.Tensor:
    """∂ρ/∂t + ∂(ρvx)/∂x + ∂(ρvy)/∂y = 0"""
    rho, vx, vy = d['rho'], d['vx'], d['vy']
    dx, dy, dt = d['dx'], d['dy'], d['dt']

    drho_dt = (rho[2:] - rho[:-2]) / (2 * dt)
    rho_n = rho[1:-1]
    vx_n, vy_n = vx[1:-1], vy[1:-1]

    drhoux_dx = fd4_dx_periodic(rho_n * vx_n, dx)
    drhouy_dy = central_dy(rho_n * vy_n, dy)

    return drho_dt[..., 1:-1] + drhoux_dx[..., 1:-1] + drhouy_dy


def mass_residual_npinn(d: dict) -> torch.Tensor:
    """Mass conservation via face fluxes."""
    rho, vx, vy = d['rho'], d['vx'], d['vy']
    dx, dy, dt = d['dx'], d['dy'], d['dt']

    drho_dt = (rho[2:] - rho[:-2]) / (2 * dt)
    rho_n = rho[1:-1]
    vx_n, vy_n = vx[1:-1], vy[1:-1]

    dFx = face_flux_dx_periodic(rho_n * vx_n, dx)
    dGy = face_flux_dy(rho_n * vy_n, dy)

    return drho_dt[..., 1:-1] + dFx[..., 1:-1] + dGy


def xmom_residual_fd(d: dict) -> torch.Tensor:
    """∂(ρvx)/∂t + ∂(ρvx²+P)/∂x + ∂(ρvx*vy)/∂y = 0"""
    rho, vx, vy, P = d['rho'], d['vx'], d['vy'], d['P']
    dx, dy, dt = d['dx'], d['dy'], d['dt']

    rhovx = rho * vx
    drhovx_dt = (rhovx[2:] - rhovx[:-2]) / (2 * dt)

    rho_n, vx_n, vy_n, P_n = rho[1:-1], vx[1:-1], vy[1:-1], P[1:-1]

    dFx = fd4_dx_periodic(rho_n * vx_n**2 + P_n, dx)
    dGy = central_dy(rho_n * vx_n * vy_n, dy)

    return drhovx_dt[..., 1:-1] + dFx[..., 1:-1] + dGy


def xmom_residual_npinn(d: dict) -> torch.Tensor:
    """x-Momentum via face fluxes."""
    rho, vx, vy, P = d['rho'], d['vx'], d['vy'], d['P']
    dx, dy, dt = d['dx'], d['dy'], d['dt']

    rhovx = rho * vx
    drhovx_dt = (rhovx[2:] - rhovx[:-2]) / (2 * dt)

    rho_n, vx_n, vy_n, P_n = rho[1:-1], vx[1:-1], vy[1:-1], P[1:-1]

    dFx = face_flux_dx_periodic(rho_n * vx_n**2 + P_n, dx)
    dGy = face_flux_dy(rho_n * vx_n * vy_n, dy)

    return drhovx_dt[..., 1:-1] + dFx[..., 1:-1] + dGy


def ymom_residual_fd(d: dict) -> torch.Tensor:
    """∂(ρvy)/∂t + ∂(ρvx*vy)/∂x + ∂(ρvy²+P)/∂y = 0"""
    rho, vx, vy, P = d['rho'], d['vx'], d['vy'], d['P']
    dx, dy, dt = d['dx'], d['dy'], d['dt']

    rhovy = rho * vy
    drhovy_dt = (rhovy[2:] - rhovy[:-2]) / (2 * dt)

    rho_n, vx_n, vy_n, P_n = rho[1:-1], vx[1:-1], vy[1:-1], P[1:-1]

    dFx = fd4_dx_periodic(rho_n * vx_n * vy_n, dx)
    dGy = central_dy(rho_n * vy_n**2 + P_n, dy)

    return drhovy_dt[..., 1:-1] + dFx[..., 1:-1] + dGy


def ymom_residual_npinn(d: dict) -> torch.Tensor:
    """y-Momentum via face fluxes."""
    rho, vx, vy, P = d['rho'], d['vx'], d['vy'], d['P']
    dx, dy, dt = d['dx'], d['dy'], d['dt']

    rhovy = rho * vy
    drhovy_dt = (rhovy[2:] - rhovy[:-2]) / (2 * dt)

    rho_n, vx_n, vy_n, P_n = rho[1:-1], vx[1:-1], vy[1:-1], P[1:-1]

    dFx = face_flux_dx_periodic(rho_n * vx_n * vy_n, dx)
    dGy = face_flux_dy(rho_n * vy_n**2 + P_n, dy)

    return drhovy_dt[..., 1:-1] + dFx[..., 1:-1] + dGy


def energy_residual_fd(d: dict) -> torch.Tensor:
    """∂E/∂t + ∂((E+P)vx)/∂x + ∂((E+P)vy)/∂y = -E/t_cool"""
    rho, P, vx, vy = d['rho'], d['P'], d['vx'], d['vy']
    dx, dy, dt = d['dx'], d['dy'], d['dt']
    gamma, tcool = d['gamma'], d['tcool']

    E = P / (gamma - 1)
    dE_dt = (E[2:] - E[:-2]) / (2 * dt)

    E_n, P_n = E[1:-1], P[1:-1]
    vx_n, vy_n = vx[1:-1], vy[1:-1]

    dFx = fd4_dx_periodic((E_n + P_n) * vx_n, dx)
    dGy = central_dy((E_n + P_n) * vy_n, dy)

    source = -E_n / tcool

    return dE_dt[..., 1:-1] + dFx[..., 1:-1] + dGy - source[..., 1:-1]


def energy_residual_npinn(d: dict) -> torch.Tensor:
    """Energy via face fluxes."""
    rho, P, vx, vy = d['rho'], d['P'], d['vx'], d['vy']
    dx, dy, dt = d['dx'], d['dy'], d['dt']
    gamma, tcool = d['gamma'], d['tcool']

    E = P / (gamma - 1)
    dE_dt = (E[2:] - E[:-2]) / (2 * dt)

    E_n, P_n = E[1:-1], P[1:-1]
    vx_n, vy_n = vx[1:-1], vy[1:-1]

    dFx = face_flux_dx_periodic((E_n + P_n) * vx_n, dx)
    dGy = face_flux_dy((E_n + P_n) * vy_n, dy)

    source = -E_n / tcool

    return dE_dt[..., 1:-1] + dFx[..., 1:-1] + dGy - source[..., 1:-1]


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else \
        'data/finetune/turbulent_radiative_layer_tcool_0.03.hdf5'

    print(f"Dataset: {path}")

    for sample_idx in [0, 1]:
        for t_start, t_end in [(10, 30), (40, 60), (70, 90)]:
            print(f"\n{'='*70}")
            print(f"Sample {sample_idx}, t=[{t_start}, {t_end})")
            print(f"{'='*70}")

            d = load_trl_data(path, sample_idx, t_start, t_end)
            print(f"Grid: {d['Nx']}×{d['Ny']}, dx={d['dx']:.6f}, dy={d['dy']:.6f}")
            print(f"dt={d['dt']:.6f}, tcool={d['tcool']}, γ={d['gamma']:.4f}")

            # FD approach
            R_mass_fd = mass_residual_fd(d)
            R_xmom_fd = xmom_residual_fd(d)
            R_ymom_fd = ymom_residual_fd(d)
            R_energy_fd = energy_residual_fd(d)

            print(f"\n--- FD (4th-x periodic + 2nd-y central) ---")
            for name, R in [('Mass', R_mass_fd), ('x-Mom', R_xmom_fd),
                            ('y-Mom', R_ymom_fd), ('Energy', R_energy_fd)]:
                mse_full = (R**2).mean().item()
                # Skip boundary regions (10 rows from each y-end)
                R_int = R[..., 10:-10]
                mse_int = (R_int**2).mean().item()
                print(f"  {name:8s}: MSE={mse_full:.6e} | interior(skip10): {mse_int:.6e}")

            # n-PINN approach
            R_mass_np = mass_residual_npinn(d)
            R_xmom_np = xmom_residual_npinn(d)
            R_ymom_np = ymom_residual_npinn(d)
            R_energy_np = energy_residual_npinn(d)

            print(f"\n--- n-PINN (face-flux) ---")
            for name, R in [('Mass', R_mass_np), ('x-Mom', R_xmom_np),
                            ('y-Mom', R_ymom_np), ('Energy', R_energy_np)]:
                mse_full = (R**2).mean().item()
                R_int = R[..., 10:-10]
                mse_int = (R_int**2).mean().item()
                print(f"  {name:8s}: MSE={mse_full:.6e} | interior(skip10): {mse_int:.6e}")

            # Field magnitude for reference
            rho_n = d['rho'][1:-1, :, 1:-1]
            P_n = d['P'][1:-1, :, 1:-1]
            E_n = P_n / (d['gamma'] - 1)
            print(f"\n--- Reference magnitudes ---")
            print(f"  |ρ| mean={rho_n.mean():.4f}")
            print(f"  |E| mean={E_n.mean():.4f}")
            print(f"  |E/tcool| mean={(E_n/d['tcool']).mean():.4f} (source term)")

            if sample_idx > 0:
                break
        if sample_idx > 0:
            break
