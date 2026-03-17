"""
Turbulent Radiative Layer 2D — Compare multiple FD schemes for PDE residual.

Goal: find the numerical scheme that minimizes GT PDE residual for compressible Euler:
    Mass:  dρ/dt + d(ρvx)/dx + d(ρvy)/dy = 0
    x-Mom: d(ρvx)/dt + d(ρvx²+P)/dx + d(ρvx*vy)/dy = 0
    y-Mom: d(ρvy)/dt + d(ρvx*vy)/dx + d(ρvy²+P)/dy = 0

Data: 128×384 (x periodic, y open), Lx=1.0, Ly=3.0
Extreme density contrast: ρ ∈ [0.5, 181]

Schemes tested:
    Spatial:
      A) 2nd-order central (baseline)
      B) 4th-order central
      C) n-PINN conservative upwind (face-velocity + 2nd-order upwind interpolation)
    Temporal:
      A) 2nd-order central
      B) 4th-order central
    skip_bl sweep: 10, 20, 30, 40, 50

Usage:
    python tools/test_trl2d_fd_schemes.py
"""

import torch
import numpy as np
import h5py
import time
from typing import Dict, Tuple, List


# =============================================================================
# Data loading
# =============================================================================

def load_data(
    path: str,
    n_samples: int = 5,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """Load first n_samples from TRL2D dataset, return float32 on device."""
    print(f"Loading data from {path} ...")
    with h5py.File(path, 'r') as f:
        # scalar: (48, 101, 128, 384, 2) — density, pressure
        # vector: (48, 101, 128, 384, 3) — vx, vy, 0
        rho = torch.from_numpy(f['scalar'][:n_samples, :, :, :, 0]).float().to(device)
        P   = torch.from_numpy(f['scalar'][:n_samples, :, :, :, 1]).float().to(device)
        vx  = torch.from_numpy(f['vector'][:n_samples, :, :, :, 0]).float().to(device)
        vy  = torch.from_numpy(f['vector'][:n_samples, :, :, :, 1]).float().to(device)

    print(f"  rho shape: {rho.shape}, range: [{rho.min():.2f}, {rho.max():.2f}]")
    print(f"  P   shape: {P.shape},   range: [{P.min():.4f}, {P.max():.4f}]")
    print(f"  vx  shape: {vx.shape},  range: [{vx.min():.4f}, {vx.max():.4f}]")
    print(f"  vy  shape: {vy.shape},  range: [{vy.min():.4f}, {vy.max():.4f}]")
    return {'rho': rho, 'P': P, 'vx': vx, 'vy': vy}


# =============================================================================
# Grid constants
# =============================================================================

NX, NY = 128, 384
LX, LY = 1.0, 3.0
DX = LX / NX           # periodic
DY = LY / (NY - 1)     # open
DT = 1.597033


# =============================================================================
# Temporal derivative helpers
# =============================================================================

def dt_2nd(f: torch.Tensor) -> torch.Tensor:
    """2nd-order central time derivative. [B,T,Nx,Ny] -> [B,T-2,Nx,Ny]"""
    return (f[:, 2:] - f[:, :-2]) / (2 * DT)


def dt_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central time derivative. [B,T,Nx,Ny] -> [B,T-4,Nx,Ny]"""
    return (-f[:, 4:] + 8*f[:, 3:-1] - 8*f[:, 1:-3] + f[:, :-4]) / (12 * DT)


# =============================================================================
# Spatial derivative helpers
# =============================================================================

# --- A) 2nd-order central ---

def dx_2nd_periodic(f: torch.Tensor) -> torch.Tensor:
    """2nd-order central x-derivative (periodic). [..., Nx, Ny] -> [..., Nx, Ny]"""
    return (torch.roll(f, -1, dims=-2) - torch.roll(f, 1, dims=-2)) / (2 * DX)


def dy_2nd_interior(f: torch.Tensor) -> torch.Tensor:
    """2nd-order central y-derivative (interior). [..., Nx, Ny] -> [..., Nx, Ny-2]"""
    return (f[..., 2:] - f[..., :-2]) / (2 * DY)


# --- B) 4th-order central ---

def dx_4th_periodic(f: torch.Tensor) -> torch.Tensor:
    """4th-order central x-derivative (periodic). [..., Nx, Ny] -> [..., Nx, Ny]"""
    return (-torch.roll(f, -2, dims=-2) + 8*torch.roll(f, -1, dims=-2)
            - 8*torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * DX)


def dy_4th_interior(f: torch.Tensor) -> torch.Tensor:
    """4th-order central y-derivative (interior). [..., Nx, Ny] -> [..., Nx, Ny-4]"""
    return (-f[..., 4:] + 8*f[..., 3:-1] - 8*f[..., 1:-3] + f[..., :-4]) / (12 * DY)


# --- C) n-PINN conservative upwind ---
# x-direction: periodic (use torch.roll)
# y-direction: open (use slicing, interior only)
#
# For compressible Euler in conservative form, the transported quantity
# is the conserved variable Q (rho, rho*vx, rho*vy) and we compute
# d(flux)/dx + d(flux)/dy where flux = Q * velocity (+ pressure for momentum).
#
# n-PINN approach: interpolate Q to cell faces with 2nd-order upwind,
# multiply by face velocity, then take flux divergence.
# For momentum eqs, pressure is added separately.

def npinn_flux_div_x_periodic(
    Q: torch.Tensor,
    u_face_vel: torch.Tensor,
) -> torch.Tensor:
    """
    Conservative x-flux divergence (periodic) using n-PINN 2nd-order upwind.

    Face velocity at east face: uc_e = 0.5*(u[i] + u[i+1])
    Face value of Q (east face):
        uc_e >= 0: Q_e = 1.5*Q[i]   - 0.5*Q[i-1]   (upwind from west)
        uc_e <  0: Q_e = 1.5*Q[i+1] - 0.5*Q[i+2]   (upwind from east)
    Flux_e = uc_e * Q_e
    d(flux)/dx = (Flux_e - Flux_w) / dx

    Q, u_face_vel: [..., Nx, Ny]
    Returns: [..., Nx, Ny]
    """
    Q_E  = torch.roll(Q, -1, dims=-2)
    Q_W  = torch.roll(Q, 1, dims=-2)
    Q_EE = torch.roll(Q, -2, dims=-2)
    Q_WW = torch.roll(Q, 2, dims=-2)

    u_E = torch.roll(u_face_vel, -1, dims=-2)
    u_W = torch.roll(u_face_vel, 1, dims=-2)

    # East face velocity
    uc_e = 0.5 * (u_face_vel + u_E)
    # West face velocity
    uc_w = 0.5 * (u_W + u_face_vel)

    # East face Q (2nd-order upwind)
    Qe_pos = 1.5 * Q - 0.5 * Q_W        # upwind from west
    Qe_neg = 1.5 * Q_E - 0.5 * Q_EE     # upwind from east
    Qe = torch.where(uc_e >= 0, Qe_pos, Qe_neg)

    # West face Q
    Qw_pos = 1.5 * Q_W - 0.5 * Q_WW     # upwind from west
    Qw_neg = 1.5 * Q - 0.5 * Q_E        # upwind from east
    Qw = torch.where(uc_w >= 0, Qw_pos, Qw_neg)

    # Flux divergence
    return (uc_e * Qe - uc_w * Qw) / DX


def npinn_flux_div_y_interior(
    Q: torch.Tensor,
    v_face_vel: torch.Tensor,
    skip: int = 2,
) -> torch.Tensor:
    """
    Conservative y-flux divergence (interior only, non-periodic) using n-PINN upwind.

    Uses slicing (no torch.roll). skip >= 2 required for 2nd-order upwind stencil.

    Q, v_face_vel: [..., Nx, Ny]
    Returns: [..., Nx, Ny - 2*skip]
    """
    s = skip
    Ny = Q.shape[-1]

    # Center and neighbors via slicing
    Q_C  = Q[..., s:Ny-s]
    Q_N  = Q[..., s+1:Ny-s+1]
    Q_S  = Q[..., s-1:Ny-s-1]
    Q_NN = Q[..., s+2:Ny-s+2]
    Q_SS = Q[..., s-2:Ny-s-2]

    v_C = v_face_vel[..., s:Ny-s]
    v_N = v_face_vel[..., s+1:Ny-s+1]
    v_S = v_face_vel[..., s-1:Ny-s-1]

    # North face velocity
    vc_n = 0.5 * (v_C + v_N)
    # South face velocity
    vc_s = 0.5 * (v_S + v_C)

    # North face Q (2nd-order upwind)
    Qn_pos = 1.5 * Q_C - 0.5 * Q_S       # upwind from south
    Qn_neg = 1.5 * Q_N - 0.5 * Q_NN      # upwind from north
    Qn = torch.where(vc_n >= 0, Qn_pos, Qn_neg)

    # South face Q
    Qs_pos = 1.5 * Q_S - 0.5 * Q_SS      # upwind from south
    Qs_neg = 1.5 * Q_C - 0.5 * Q_N       # upwind from north
    Qs = torch.where(vc_s >= 0, Qs_pos, Qs_neg)

    return (vc_n * Qn - vc_s * Qs) / DY


def npinn_pressure_grad_x_periodic(P: torch.Tensor) -> torch.Tensor:
    """Face-averaged pressure gradient in x (periodic). [..., Nx, Ny] -> [..., Nx, Ny]"""
    P_E = torch.roll(P, -1, dims=-2)
    P_W = torch.roll(P, 1, dims=-2)
    # dp/dx = ((P+P_E)/2 - (P_W+P)/2) / dx = (P_E - P_W) / (2*dx)
    return (0.5 * (P + P_E) - 0.5 * (P_W + P)) / DX


def npinn_pressure_grad_y_interior(P: torch.Tensor, skip: int = 2) -> torch.Tensor:
    """Face-averaged pressure gradient in y (interior). [..., Nx, Ny] -> [..., Nx, Ny-2*skip]"""
    s = skip
    Ny = P.shape[-1]
    P_C = P[..., s:Ny-s]
    P_N = P[..., s+1:Ny-s+1]
    P_S = P[..., s-1:Ny-s-1]
    return (0.5 * (P_C + P_N) - 0.5 * (P_S + P_C)) / DY


# =============================================================================
# Scheme A: 2nd-order central (spatial) + 2nd-order central (time)
# =============================================================================

def residuals_2nd_2nd(
    data: Dict[str, torch.Tensor],
    skip_bl: int = 10,
) -> Dict[str, torch.Tensor]:
    """2nd-order spatial + 2nd-order temporal."""
    rho, P, vx, vy = data['rho'], data['P'], data['vx'], data['vy']

    # Time derivatives (2nd-order)
    drho_dt = dt_2nd(rho)
    drhovx_dt = dt_2nd(rho * vx)
    drhovy_dt = dt_2nd(rho * vy)

    # Mid-time slices
    rho_m = rho[:, 1:-1]
    vx_m = vx[:, 1:-1]
    vy_m = vy[:, 1:-1]
    P_m = P[:, 1:-1]

    # Mass
    drhoux_dx = dx_2nd_periodic(rho_m * vx_m)
    drhouy_dy = dy_2nd_interior(rho_m * vy_m)
    R_mass = drho_dt[..., 1:-1] + drhoux_dx[..., 1:-1] + drhouy_dy
    if skip_bl > 0:
        R_mass = R_mass[..., skip_bl:-skip_bl]

    # x-Mom
    dFx_x = dx_2nd_periodic(rho_m * vx_m**2 + P_m)
    dGx_y = dy_2nd_interior(rho_m * vx_m * vy_m)
    R_xmom = drhovx_dt[..., 1:-1] + dFx_x[..., 1:-1] + dGx_y
    if skip_bl > 0:
        R_xmom = R_xmom[..., skip_bl:-skip_bl]

    # y-Mom
    dFy_x = dx_2nd_periodic(rho_m * vx_m * vy_m)
    dGy_y = dy_2nd_interior(rho_m * vy_m**2 + P_m)
    R_ymom = drhovy_dt[..., 1:-1] + dFy_x[..., 1:-1] + dGy_y
    if skip_bl > 0:
        R_ymom = R_ymom[..., skip_bl:-skip_bl]

    return {'mass': R_mass, 'x_mom': R_xmom, 'y_mom': R_ymom}


# =============================================================================
# Scheme B: 4th-order central (spatial) + 2nd-order central (time)
# =============================================================================

def residuals_4th_2nd(
    data: Dict[str, torch.Tensor],
    skip_bl: int = 10,
) -> Dict[str, torch.Tensor]:
    """4th-order spatial + 2nd-order temporal (current baseline in pde_loss_verified.py)."""
    rho, P, vx, vy = data['rho'], data['P'], data['vx'], data['vy']

    drho_dt = dt_2nd(rho)
    drhovx_dt = dt_2nd(rho * vx)
    drhovy_dt = dt_2nd(rho * vy)

    rho_m = rho[:, 1:-1]
    vx_m = vx[:, 1:-1]
    vy_m = vy[:, 1:-1]
    P_m = P[:, 1:-1]

    # Mass
    drhoux_dx = dx_4th_periodic(rho_m * vx_m)
    drhouy_dy = dy_2nd_interior(rho_m * vy_m)
    R_mass = drho_dt[..., 1:-1] + drhoux_dx[..., 1:-1] + drhouy_dy
    if skip_bl > 0:
        R_mass = R_mass[..., skip_bl:-skip_bl]

    # x-Mom
    dFx_x = dx_4th_periodic(rho_m * vx_m**2 + P_m)
    dGx_y = dy_2nd_interior(rho_m * vx_m * vy_m)
    R_xmom = drhovx_dt[..., 1:-1] + dFx_x[..., 1:-1] + dGx_y
    if skip_bl > 0:
        R_xmom = R_xmom[..., skip_bl:-skip_bl]

    # y-Mom
    dFy_x = dx_4th_periodic(rho_m * vx_m * vy_m)
    dGy_y = dy_2nd_interior(rho_m * vy_m**2 + P_m)
    R_ymom = drhovy_dt[..., 1:-1] + dFy_x[..., 1:-1] + dGy_y
    if skip_bl > 0:
        R_ymom = R_ymom[..., skip_bl:-skip_bl]

    return {'mass': R_mass, 'x_mom': R_xmom, 'y_mom': R_ymom}


# =============================================================================
# Scheme B2: 4th-order central (spatial) + 4th-order central (time)
# =============================================================================

def residuals_4th_4th(
    data: Dict[str, torch.Tensor],
    skip_bl: int = 10,
) -> Dict[str, torch.Tensor]:
    """4th-order spatial + 4th-order temporal."""
    rho, P, vx, vy = data['rho'], data['P'], data['vx'], data['vy']

    drho_dt = dt_4th(rho)
    drhovx_dt = dt_4th(rho * vx)
    drhovy_dt = dt_4th(rho * vy)

    # Mid-time slices aligned with 4th-order time derivative
    rho_m = rho[:, 2:-2]
    vx_m = vx[:, 2:-2]
    vy_m = vy[:, 2:-2]
    P_m = P[:, 2:-2]

    # Mass
    drhoux_dx = dx_4th_periodic(rho_m * vx_m)
    drhouy_dy = dy_2nd_interior(rho_m * vy_m)
    R_mass = drho_dt[..., 1:-1] + drhoux_dx[..., 1:-1] + drhouy_dy
    if skip_bl > 0:
        R_mass = R_mass[..., skip_bl:-skip_bl]

    # x-Mom
    dFx_x = dx_4th_periodic(rho_m * vx_m**2 + P_m)
    dGx_y = dy_2nd_interior(rho_m * vx_m * vy_m)
    R_xmom = drhovx_dt[..., 1:-1] + dFx_x[..., 1:-1] + dGx_y
    if skip_bl > 0:
        R_xmom = R_xmom[..., skip_bl:-skip_bl]

    # y-Mom
    dFy_x = dx_4th_periodic(rho_m * vx_m * vy_m)
    dGy_y = dy_2nd_interior(rho_m * vy_m**2 + P_m)
    R_ymom = drhovy_dt[..., 1:-1] + dFy_x[..., 1:-1] + dGy_y
    if skip_bl > 0:
        R_ymom = R_ymom[..., skip_bl:-skip_bl]

    return {'mass': R_mass, 'x_mom': R_xmom, 'y_mom': R_ymom}


# =============================================================================
# Scheme C: n-PINN conservative upwind (spatial) + 2nd-order central (time)
# =============================================================================

def residuals_npinn_2nd(
    data: Dict[str, torch.Tensor],
    skip_bl: int = 10,
    use_div_correction: bool = True,
) -> Dict[str, torch.Tensor]:
    """n-PINN conservative upwind spatial + 2nd-order temporal."""
    rho, P, vx, vy = data['rho'], data['P'], data['vx'], data['vy']

    # Conserved variables
    rhovx = rho * vx
    rhovy = rho * vy

    drho_dt = dt_2nd(rho)
    drhovx_dt = dt_2nd(rhovx)
    drhovy_dt = dt_2nd(rhovy)

    rho_m = rho[:, 1:-1]
    vx_m = vx[:, 1:-1]
    vy_m = vy[:, 1:-1]
    P_m = P[:, 1:-1]
    rhovx_m = rho_m * vx_m
    rhovy_m = rho_m * vy_m

    # y skip for n-PINN (need >=2 for stencil)
    y_skip = max(skip_bl, 2)

    # --- Mass: d(rho)/dt + d(rho*vx)/dx + d(rho*vy)/dy = 0 ---
    # Transport rho with velocity (vx, vy)
    drhoux_dx = npinn_flux_div_x_periodic(rho_m, vx_m)
    drhouy_dy = npinn_flux_div_y_interior(rho_m, vy_m, skip=y_skip)

    # Trim x-flux to match y interior region
    drhoux_dx_trim = drhoux_dx[..., y_skip:NY-y_skip]
    drho_dt_trim = drho_dt[..., y_skip:NY-y_skip]

    R_mass = drho_dt_trim + drhoux_dx_trim + drhouy_dy

    # Div correction for mass: -rho * div(v)
    if use_div_correction:
        # Face velocity divergence (interior)
        v_C_y = vy_m[..., y_skip:NY-y_skip]
        v_N_y = vy_m[..., y_skip+1:NY-y_skip+1]
        v_S_y = vy_m[..., y_skip-1:NY-y_skip-1]
        vc_n = 0.5 * (v_C_y + v_N_y)
        vc_s = 0.5 * (v_S_y + v_C_y)

        vx_C = vx_m[..., y_skip:NY-y_skip]
        vx_E = torch.roll(vx_m, -1, dims=-2)[..., y_skip:NY-y_skip]
        vx_W = torch.roll(vx_m, 1, dims=-2)[..., y_skip:NY-y_skip]
        uc_e = 0.5 * (vx_C + vx_E)
        uc_w = 0.5 * (vx_W + vx_C)

        div_v = (uc_e - uc_w) / DX + (vc_n - vc_s) / DY
        rho_int = rho_m[..., y_skip:NY-y_skip]
        R_mass = R_mass - rho_int * div_v

    # Additional skip_bl trimming (if y_skip < skip_bl this is already handled)
    # Actually y_skip = max(skip_bl, 2) so no additional trim needed for mass

    # --- x-Mom: d(rho*vx)/dt + d(rho*vx^2 + P)/dx + d(rho*vx*vy)/dy = 0 ---
    # x-flux: rho*vx transported by vx, plus pressure
    # We split: upwind transport of (rho*vx) by velocity + pressure gradient
    drhovxvx_dx = npinn_flux_div_x_periodic(rhovx_m, vx_m)
    dpressure_dx = npinn_pressure_grad_x_periodic(P_m)
    drhovxvy_dy = npinn_flux_div_y_interior(rhovx_m, vy_m, skip=y_skip)
    # No separate pressure in y for x-momentum

    drhovxvx_dx_trim = drhovxvx_dx[..., y_skip:NY-y_skip]
    dpressure_dx_trim = dpressure_dx[..., y_skip:NY-y_skip]
    drhovx_dt_trim = drhovx_dt[..., y_skip:NY-y_skip]

    R_xmom = drhovx_dt_trim + drhovxvx_dx_trim + dpressure_dx_trim + drhovxvy_dy

    if use_div_correction:
        rhovx_int = rhovx_m[..., y_skip:NY-y_skip]
        R_xmom = R_xmom - rhovx_int * div_v

    # --- y-Mom: d(rho*vy)/dt + d(rho*vx*vy)/dx + d(rho*vy^2 + P)/dy = 0 ---
    drhovyvx_dx = npinn_flux_div_x_periodic(rhovy_m, vx_m)
    drhovyvy_dy = npinn_flux_div_y_interior(rhovy_m, vy_m, skip=y_skip)
    dpressure_dy = npinn_pressure_grad_y_interior(P_m, skip=y_skip)

    drhovyvx_dx_trim = drhovyvx_dx[..., y_skip:NY-y_skip]
    drhovy_dt_trim = drhovy_dt[..., y_skip:NY-y_skip]

    R_ymom = drhovy_dt_trim + drhovyvx_dx_trim + drhovyvy_dy + dpressure_dy

    if use_div_correction:
        rhovy_int = rhovy_m[..., y_skip:NY-y_skip]
        R_ymom = R_ymom - rhovy_int * div_v

    return {'mass': R_mass, 'x_mom': R_xmom, 'y_mom': R_ymom}


# =============================================================================
# Scheme C2: n-PINN conservative upwind (spatial) + 4th-order central (time)
# =============================================================================

def residuals_npinn_4th(
    data: Dict[str, torch.Tensor],
    skip_bl: int = 10,
    use_div_correction: bool = True,
) -> Dict[str, torch.Tensor]:
    """n-PINN conservative upwind spatial + 4th-order temporal."""
    rho, P, vx, vy = data['rho'], data['P'], data['vx'], data['vy']

    rhovx = rho * vx
    rhovy = rho * vy

    drho_dt = dt_4th(rho)
    drhovx_dt = dt_4th(rhovx)
    drhovy_dt = dt_4th(rhovy)

    # Mid slices for 4th-order time
    rho_m = rho[:, 2:-2]
    vx_m = vx[:, 2:-2]
    vy_m = vy[:, 2:-2]
    P_m = P[:, 2:-2]
    rhovx_m = rho_m * vx_m
    rhovy_m = rho_m * vy_m

    y_skip = max(skip_bl, 2)

    # Mass
    drhoux_dx = npinn_flux_div_x_periodic(rho_m, vx_m)
    drhouy_dy = npinn_flux_div_y_interior(rho_m, vy_m, skip=y_skip)
    drhoux_dx_trim = drhoux_dx[..., y_skip:NY-y_skip]
    drho_dt_trim = drho_dt[..., y_skip:NY-y_skip]
    R_mass = drho_dt_trim + drhoux_dx_trim + drhouy_dy

    if use_div_correction:
        v_C_y = vy_m[..., y_skip:NY-y_skip]
        v_N_y = vy_m[..., y_skip+1:NY-y_skip+1]
        v_S_y = vy_m[..., y_skip-1:NY-y_skip-1]
        vc_n = 0.5 * (v_C_y + v_N_y)
        vc_s = 0.5 * (v_S_y + v_C_y)
        vx_C = vx_m[..., y_skip:NY-y_skip]
        vx_E = torch.roll(vx_m, -1, dims=-2)[..., y_skip:NY-y_skip]
        vx_W = torch.roll(vx_m, 1, dims=-2)[..., y_skip:NY-y_skip]
        uc_e = 0.5 * (vx_C + vx_E)
        uc_w = 0.5 * (vx_W + vx_C)
        div_v = (uc_e - uc_w) / DX + (vc_n - vc_s) / DY
        rho_int = rho_m[..., y_skip:NY-y_skip]
        R_mass = R_mass - rho_int * div_v

    # x-Mom
    drhovxvx_dx = npinn_flux_div_x_periodic(rhovx_m, vx_m)
    dpressure_dx = npinn_pressure_grad_x_periodic(P_m)
    drhovxvy_dy = npinn_flux_div_y_interior(rhovx_m, vy_m, skip=y_skip)
    R_xmom = (drhovx_dt[..., y_skip:NY-y_skip]
              + drhovxvx_dx[..., y_skip:NY-y_skip]
              + dpressure_dx[..., y_skip:NY-y_skip]
              + drhovxvy_dy)
    if use_div_correction:
        rhovx_int = rhovx_m[..., y_skip:NY-y_skip]
        R_xmom = R_xmom - rhovx_int * div_v

    # y-Mom
    drhovyvx_dx = npinn_flux_div_x_periodic(rhovy_m, vx_m)
    drhovyvy_dy = npinn_flux_div_y_interior(rhovy_m, vy_m, skip=y_skip)
    dpressure_dy = npinn_pressure_grad_y_interior(P_m, skip=y_skip)
    R_ymom = (drhovy_dt[..., y_skip:NY-y_skip]
              + drhovyvx_dx[..., y_skip:NY-y_skip]
              + drhovyvy_dy
              + dpressure_dy)
    if use_div_correction:
        rhovy_int = rhovy_m[..., y_skip:NY-y_skip]
        R_ymom = R_ymom - rhovy_int * div_v

    return {'mass': R_mass, 'x_mom': R_xmom, 'y_mom': R_ymom}


# =============================================================================
# Scheme D: 4th-order x + 4th-order y (both spatial) + 2nd-order time
# =============================================================================

def residuals_4thx_4thy_2nd(
    data: Dict[str, torch.Tensor],
    skip_bl: int = 10,
) -> Dict[str, torch.Tensor]:
    """4th-order spatial (both x and y) + 2nd-order temporal."""
    rho, P, vx, vy = data['rho'], data['P'], data['vx'], data['vy']

    drho_dt = dt_2nd(rho)
    drhovx_dt = dt_2nd(rho * vx)
    drhovy_dt = dt_2nd(rho * vy)

    rho_m = rho[:, 1:-1]
    vx_m = vx[:, 1:-1]
    vy_m = vy[:, 1:-1]
    P_m = P[:, 1:-1]

    # Mass
    drhoux_dx = dx_4th_periodic(rho_m * vx_m)
    drhouy_dy = dy_4th_interior(rho_m * vy_m)  # Ny-4 output
    # dy_4th trims 2 from each side: result is [..., Nx, Ny-4]
    R_mass = drho_dt[..., 2:-2] + drhoux_dx[..., 2:-2] + drhouy_dy
    if skip_bl > 0:
        # skip_bl relative to the already-trimmed region
        effective_skip = max(skip_bl - 2, 0)
        if effective_skip > 0:
            R_mass = R_mass[..., effective_skip:-effective_skip]

    # x-Mom
    dFx_x = dx_4th_periodic(rho_m * vx_m**2 + P_m)
    dGx_y = dy_4th_interior(rho_m * vx_m * vy_m)
    R_xmom = drhovx_dt[..., 2:-2] + dFx_x[..., 2:-2] + dGx_y
    if skip_bl > 0:
        effective_skip = max(skip_bl - 2, 0)
        if effective_skip > 0:
            R_xmom = R_xmom[..., effective_skip:-effective_skip]

    # y-Mom
    dFy_x = dx_4th_periodic(rho_m * vx_m * vy_m)
    dGy_y = dy_4th_interior(rho_m * vy_m**2 + P_m)
    R_ymom = drhovy_dt[..., 2:-2] + dFy_x[..., 2:-2] + dGy_y
    if skip_bl > 0:
        effective_skip = max(skip_bl - 2, 0)
        if effective_skip > 0:
            R_ymom = R_ymom[..., effective_skip:-effective_skip]

    return {'mass': R_mass, 'x_mom': R_xmom, 'y_mom': R_ymom}


# =============================================================================
# Scheme E: n-PINN (no div correction) + 2nd time
# =============================================================================

def residuals_npinn_nodiv_2nd(
    data: Dict[str, torch.Tensor],
    skip_bl: int = 10,
) -> Dict[str, torch.Tensor]:
    """n-PINN without div correction + 2nd-order temporal."""
    return residuals_npinn_2nd(data, skip_bl=skip_bl, use_div_correction=False)


# =============================================================================
# Compute RMS
# =============================================================================

def compute_rms(residuals: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute RMS of each residual field."""
    return {
        name: torch.sqrt(torch.mean(R ** 2)).item()
        for name, R in residuals.items()
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    data_path = '/scratch-share/SONG0304/finetune/turbulent_radiative_2d.hdf5'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = load_data(data_path, n_samples=5, device=device)
    B, T = data['rho'].shape[:2]
    print(f"\nB={B}, T={T}, Nx={NX}, Ny={NY}")
    print(f"dx={DX:.6f}, dy={DY:.6f}, dt={DT:.6f}")
    print()

    # =====================================================================
    # Part 1: Compare spatial + temporal scheme combinations (skip_bl=10)
    # =====================================================================
    print("=" * 80)
    print("Part 1: Compare spatial x temporal scheme combinations (skip_bl=10)")
    print("=" * 80)

    schemes = {
        'A) 2nd-space + 2nd-time':       lambda: residuals_2nd_2nd(data, skip_bl=10),
        'B) 4thX+2ndY + 2nd-time':       lambda: residuals_4th_2nd(data, skip_bl=10),
        'C) 4thX+4thY + 2nd-time':       lambda: residuals_4thx_4thy_2nd(data, skip_bl=10),
        'D) 4thX+2ndY + 4th-time':       lambda: residuals_4th_4th(data, skip_bl=10),
        'E) nPINN(+div) + 2nd-time':     lambda: residuals_npinn_2nd(data, skip_bl=10),
        'F) nPINN(no-div) + 2nd-time':   lambda: residuals_npinn_nodiv_2nd(data, skip_bl=10),
        'G) nPINN(+div) + 4th-time':     lambda: residuals_npinn_4th(data, skip_bl=10),
    }

    results_part1: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for name, fn in schemes.items():
            t0 = time.time()
            res = fn()
            rms = compute_rms(res)
            elapsed = time.time() - t0
            results_part1[name] = rms
            print(f"\n{name} ({elapsed:.2f}s):")
            print(f"  Mass  RMS = {rms['mass']:.6f}")
            print(f"  x-Mom RMS = {rms['x_mom']:.6f}")
            print(f"  y-Mom RMS = {rms['y_mom']:.6f}")

    # =====================================================================
    # Part 2: skip_bl sweep for best spatial scheme
    # =====================================================================
    print("\n" + "=" * 80)
    print("Part 2: skip_bl sweep for each scheme")
    print("=" * 80)

    skip_bls = [10, 20, 30, 40, 50]

    # We test: best central (4thX+2ndY+2nd-time) and nPINN(+div)+2nd-time
    sweep_schemes = {
        '4thX+2ndY+2ndT':   residuals_4th_2nd,
        'nPINN(+div)+2ndT':  lambda d, s: residuals_npinn_2nd(d, skip_bl=s),
        'nPINN(nodiv)+2ndT': lambda d, s: residuals_npinn_nodiv_2nd(d, skip_bl=s),
    }

    results_part2: Dict[str, Dict[int, Dict[str, float]]] = {}

    with torch.no_grad():
        for scheme_name, fn in sweep_schemes.items():
            results_part2[scheme_name] = {}
            print(f"\n--- {scheme_name} ---")
            for sk in skip_bls:
                res = fn(data, sk)
                rms = compute_rms(res)
                results_part2[scheme_name][sk] = rms
                print(f"  skip_bl={sk:3d}: Mass={rms['mass']:.4f}  x-Mom={rms['x_mom']:.4f}  y-Mom={rms['y_mom']:.4f}")

    # =====================================================================
    # Summary table
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE — Part 1: Scheme Comparison (skip_bl=10)")
    print("=" * 80)
    print(f"{'Scheme':<35s} {'Mass RMS':>10s} {'x-Mom RMS':>10s} {'y-Mom RMS':>10s} {'Sum RMS':>10s}")
    print("-" * 80)
    for name, rms in results_part1.items():
        total = rms['mass'] + rms['x_mom'] + rms['y_mom']
        print(f"{name:<35s} {rms['mass']:10.4f} {rms['x_mom']:10.4f} {rms['y_mom']:10.4f} {total:10.4f}")

    # Find best
    best_name = min(results_part1, key=lambda n: sum(results_part1[n].values()))
    best_rms = results_part1[best_name]
    print(f"\nBest scheme: {best_name}")
    print(f"  Mass={best_rms['mass']:.4f}, x-Mom={best_rms['x_mom']:.4f}, y-Mom={best_rms['y_mom']:.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE — Part 2: skip_bl Sweep")
    print("=" * 80)
    for scheme_name, sk_results in results_part2.items():
        print(f"\n--- {scheme_name} ---")
        print(f"  {'skip_bl':>7s} {'Mass RMS':>10s} {'x-Mom RMS':>10s} {'y-Mom RMS':>10s} {'Sum RMS':>10s}")
        print(f"  {'-'*50}")
        for sk, rms in sk_results.items():
            total = rms['mass'] + rms['x_mom'] + rms['y_mom']
            print(f"  {sk:7d} {rms['mass']:10.4f} {rms['x_mom']:10.4f} {rms['y_mom']:10.4f} {total:10.4f}")

    # =====================================================================
    # Part 3: Spatial profile of residual (where is the error?)
    # =====================================================================
    print("\n" + "=" * 80)
    print("Part 3: Spatial profile of mass residual along y (2nd-order, skip_bl=0)")
    print("=" * 80)

    with torch.no_grad():
        res_profile = residuals_2nd_2nd(data, skip_bl=0)
        R_mass_full = res_profile['mass']  # [B, T-2, Nx, Ny-2]
        # RMS over B, T, Nx -> profile along y
        rms_y = torch.sqrt(torch.mean(R_mass_full ** 2, dim=(0, 1, 2)))  # [Ny-2]
        rms_y_np = rms_y.cpu().numpy()

        print(f"  Shape: {R_mass_full.shape}")
        print(f"  y-profile of mass RMS (every 20 cells):")
        for j in range(0, len(rms_y_np), 20):
            bar = '#' * int(min(rms_y_np[j] / 2.0, 40))
            print(f"    y[{j+1:3d}] = {rms_y_np[j]:8.2f}  {bar}")

        # Density profile
        rho_m = data['rho'][:, 1:-1]
        rho_mean_y = rho_m.mean(dim=(0, 1, 2)).cpu().numpy()  # [Ny]
        print(f"\n  Density profile along y (every 20 cells):")
        for j in range(0, len(rho_mean_y), 20):
            bar = '#' * int(min(rho_mean_y[j] / 2.0, 40))
            print(f"    y[{j:3d}] rho_mean = {rho_mean_y[j]:8.2f}  {bar}")

    # =====================================================================
    # Part 4: 1st-order upwind (most dissipative — closer to Godunov?)
    # =====================================================================
    print("\n" + "=" * 80)
    print("Part 4: 1st-order upwind scheme")
    print("=" * 80)

    def dx_1st_upwind_periodic(flux: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """1st-order upwind x-derivative (periodic).
        Forward: (f[i+1] - f[i]) / dx when vel < 0
        Backward: (f[i] - f[i-1]) / dx when vel >= 0
        """
        fE = torch.roll(flux, -1, dims=-2)
        fW = torch.roll(flux, 1, dims=-2)
        d_fwd = (fE - flux) / DX
        d_bwd = (flux - fW) / DX
        return torch.where(vel >= 0, d_bwd, d_fwd)

    def dy_1st_upwind_interior(flux: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """1st-order upwind y-derivative (interior, Ny-2 output).
        Forward: (f[j+1] - f[j]) / dy when vel < 0
        Backward: (f[j] - f[j-1]) / dy when vel >= 0
        """
        d_fwd = (flux[..., 2:] - flux[..., 1:-1]) / DY
        d_bwd = (flux[..., 1:-1] - flux[..., :-2]) / DY
        vel_mid = vel[..., 1:-1]
        return torch.where(vel_mid >= 0, d_bwd, d_fwd)

    def residuals_1st_upwind(
        data: Dict[str, torch.Tensor],
        skip_bl: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """1st-order upwind spatial + 2nd-order temporal."""
        rho, P, vx, vy = data['rho'], data['P'], data['vx'], data['vy']

        drho_dt = dt_2nd(rho)
        drhovx_dt = dt_2nd(rho * vx)
        drhovy_dt = dt_2nd(rho * vy)

        rho_m = rho[:, 1:-1]
        vx_m = vx[:, 1:-1]
        vy_m = vy[:, 1:-1]
        P_m = P[:, 1:-1]

        # Mass: upwind on rho*vx, rho*vy
        drhoux_dx = dx_1st_upwind_periodic(rho_m * vx_m, vx_m)
        drhouy_dy = dy_1st_upwind_interior(rho_m * vy_m, vy_m)
        R_mass = drho_dt[..., 1:-1] + drhoux_dx[..., 1:-1] + drhouy_dy
        if skip_bl > 0:
            R_mass = R_mass[..., skip_bl:-skip_bl]

        # x-Mom: upwind based on vx for x-flux, vy for y-flux
        dFx_x = dx_1st_upwind_periodic(rho_m * vx_m**2 + P_m, vx_m)
        dGx_y = dy_1st_upwind_interior(rho_m * vx_m * vy_m, vy_m)
        R_xmom = drhovx_dt[..., 1:-1] + dFx_x[..., 1:-1] + dGx_y
        if skip_bl > 0:
            R_xmom = R_xmom[..., skip_bl:-skip_bl]

        # y-Mom
        dFy_x = dx_1st_upwind_periodic(rho_m * vx_m * vy_m, vx_m)
        dGy_y = dy_1st_upwind_interior(rho_m * vy_m**2 + P_m, vy_m)
        R_ymom = drhovy_dt[..., 1:-1] + dFy_x[..., 1:-1] + dGy_y
        if skip_bl > 0:
            R_ymom = R_ymom[..., skip_bl:-skip_bl]

        return {'mass': R_mass, 'x_mom': R_xmom, 'y_mom': R_ymom}

    with torch.no_grad():
        res_1up = residuals_1st_upwind(data, skip_bl=10)
        rms_1up = compute_rms(res_1up)
        print(f"1st-order upwind + 2nd-time (skip_bl=10):")
        print(f"  Mass  RMS = {rms_1up['mass']:.4f}")
        print(f"  x-Mom RMS = {rms_1up['x_mom']:.4f}")
        print(f"  y-Mom RMS = {rms_1up['y_mom']:.4f}")

    # =====================================================================
    # Part 5: Simple face-flux (cell-averaged) approach
    # =====================================================================
    print("\n" + "=" * 80)
    print("Part 5: Face-flux (simple cell-averaged) approach")
    print("=" * 80)

    def face_flux_dx_periodic_simple(flux: torch.Tensor) -> torch.Tensor:
        """d(flux)/dx via face averaging (periodic).
        F_e = 0.5*(F[i]+F[i+1]), d/dx = (F_e - F_w)/dx"""
        F_e = 0.5 * (flux + torch.roll(flux, -1, dims=-2))
        F_w = 0.5 * (flux + torch.roll(flux, 1, dims=-2))
        return (F_e - F_w) / DX

    def face_flux_dy_simple(flux: torch.Tensor) -> torch.Tensor:
        """d(flux)/dy via face averaging (interior, Ny-2).
        G_n[j] = 0.5*(G[j]+G[j+1]), d/dy = (G_n[j] - G_n[j-1])/dy"""
        G_n = 0.5 * (flux[..., 1:] + flux[..., :-1])  # Ny-1
        return (G_n[..., 1:] - G_n[..., :-1]) / DY      # Ny-2

    def residuals_face_flux(
        data: Dict[str, torch.Tensor],
        skip_bl: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """Face-flux (simple cell-averaged) spatial + 2nd-order temporal."""
        rho, P, vx, vy = data['rho'], data['P'], data['vx'], data['vy']

        drho_dt = dt_2nd(rho)
        drhovx_dt = dt_2nd(rho * vx)
        drhovy_dt = dt_2nd(rho * vy)

        rho_m = rho[:, 1:-1]
        vx_m = vx[:, 1:-1]
        vy_m = vy[:, 1:-1]
        P_m = P[:, 1:-1]

        # Mass
        drhoux_dx = face_flux_dx_periodic_simple(rho_m * vx_m)
        drhouy_dy = face_flux_dy_simple(rho_m * vy_m)
        R_mass = drho_dt[..., 1:-1] + drhoux_dx[..., 1:-1] + drhouy_dy
        if skip_bl > 0:
            R_mass = R_mass[..., skip_bl:-skip_bl]

        # x-Mom
        dFx_x = face_flux_dx_periodic_simple(rho_m * vx_m**2 + P_m)
        dGx_y = face_flux_dy_simple(rho_m * vx_m * vy_m)
        R_xmom = drhovx_dt[..., 1:-1] + dFx_x[..., 1:-1] + dGx_y
        if skip_bl > 0:
            R_xmom = R_xmom[..., skip_bl:-skip_bl]

        # y-Mom
        dFy_x = face_flux_dx_periodic_simple(rho_m * vx_m * vy_m)
        dGy_y = face_flux_dy_simple(rho_m * vy_m**2 + P_m)
        R_ymom = drhovy_dt[..., 1:-1] + dFy_x[..., 1:-1] + dGy_y
        if skip_bl > 0:
            R_ymom = R_ymom[..., skip_bl:-skip_bl]

        return {'mass': R_mass, 'x_mom': R_xmom, 'y_mom': R_ymom}

    with torch.no_grad():
        res_ff = residuals_face_flux(data, skip_bl=10)
        rms_ff = compute_rms(res_ff)
        print(f"Face-flux (cell-avg) + 2nd-time (skip_bl=10):")
        print(f"  Mass  RMS = {rms_ff['mass']:.4f}")
        print(f"  x-Mom RMS = {rms_ff['x_mom']:.4f}")
        print(f"  y-Mom RMS = {rms_ff['y_mom']:.4f}")

    # =====================================================================
    # Part 6: Median-based residual (less sensitive to outliers)
    # =====================================================================
    print("\n" + "=" * 80)
    print("Part 6: Median absolute residual for best scheme (2nd-2nd)")
    print("=" * 80)

    with torch.no_grad():
        res_base = residuals_2nd_2nd(data, skip_bl=10)
        for eq_name in ['mass', 'x_mom', 'y_mom']:
            R = res_base[eq_name]
            abs_R = R.abs().flatten()
            # Subsample to avoid quantile OOM
            n = abs_R.numel()
            if n > 1_000_000:
                idx = torch.randperm(n, device=abs_R.device)[:1_000_000]
                abs_R_sub = abs_R[idx]
            else:
                abs_R_sub = abs_R
            median_val = torch.median(abs_R_sub).item()
            p90 = torch.quantile(abs_R_sub, 0.90).item()
            p99 = torch.quantile(abs_R_sub, 0.99).item()
            max_val = abs_R.max().item()
            print(f"  {eq_name:6s}: median={median_val:.4f}, p90={p90:.4f}, p99={p99:.4f}, max={max_val:.2f}")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — All Schemes (skip_bl=10)")
    print("=" * 80)

    all_results = {**results_part1}
    all_results['H) 1st-order upwind + 2nd-time'] = rms_1up
    all_results['I) Face-flux(cell-avg) + 2nd-time'] = rms_ff

    print(f"{'Scheme':<40s} {'Mass RMS':>10s} {'x-Mom RMS':>10s} {'y-Mom RMS':>10s} {'Sum RMS':>10s}")
    print("-" * 85)
    for name, rms in sorted(all_results.items(), key=lambda x: sum(x[1].values())):
        total = rms['mass'] + rms['x_mom'] + rms['y_mom']
        print(f"{name:<40s} {rms['mass']:10.4f} {rms['x_mom']:10.4f} {rms['y_mom']:10.4f} {total:10.4f}")

    best_name = min(all_results, key=lambda n: sum(all_results[n].values()))
    best_rms = all_results[best_name]
    print(f"\nOverall best: {best_name}")
    print(f"  Mass={best_rms['mass']:.4f}, x-Mom={best_rms['x_mom']:.4f}, y-Mom={best_rms['y_mom']:.4f}")

    print("\nDone.")
