"""
Rayleigh-Bénard PDE residual verification — n-PINN + FD methods.

Official equations (HuggingFace polymathic-ai/rayleigh_benard):
    ∂b/∂t - κΔb = -u·∇b                    (buoyancy)
    ∂u/∂t - νΔu + ∇p - b·ê_z = -u·∇u      (momentum)
    div(u) = 0                               (incompressibility)

    κ = (Ra × Pr)^(-1/2)
    ν = (Ra / Pr)^(-1/2)

BCs:
    x: periodic
    z (y-axis in data): Dirichlet
        b(z=0) = Lz = 1,  b(z=Lz) = 0
        u(z=0) = u(z=Lz) = 0

Grid: 512×128, x∈[0,4], z∈[0,1]
"""

import torch
import numpy as np
import h5py
import sys


def load_data(path: str, sample: int, t0: int, t1: int) -> dict:
    f = h5py.File(path, 'r')
    b = torch.from_numpy(f['t0_fields/buoyancy'][sample, t0:t1]).double()
    p = torch.from_numpy(f['t0_fields/pressure'][sample, t0:t1]).double()
    vel = torch.from_numpy(f['t1_fields/velocity'][sample, t0:t1]).double()
    ux = vel[..., 0]  # horizontal velocity
    uz = vel[..., 1]  # vertical velocity
    t = f['dimensions/time'][t0:t1]
    Ra = float(f.attrs['Rayleigh'])
    Pr = float(f.attrs['Prandtl'])
    f.close()

    Nx, Nz = 512, 128
    Lx, Lz = 4.0, 1.0
    dx = Lx / Nx
    dz = Lz / (Nz - 1)
    dt = float(t[1] - t[0])
    kappa = (Ra * Pr) ** (-0.5)
    nu = (Ra / Pr) ** (-0.5)

    return {
        'b': b, 'p': p, 'ux': ux, 'uz': uz,
        'dx': dx, 'dz': dz, 'dt': dt,
        'Ra': Ra, 'Pr': Pr, 'kappa': kappa, 'nu': nu,
        'Nx': Nx, 'Nz': Nz, 'Lx': Lx, 'Lz': Lz,
    }


# ============================================================
# n-PINN operators (following the notebook exactly)
# ============================================================

def get_neighbors_x(f: torch.Tensor):
    """Periodic neighbors in x. f: [..., Nx, Nz]"""
    fE = torch.roll(f, -1, -2)
    fW = torch.roll(f, 1, -2)
    fEE = torch.roll(f, -2, -2)
    fWW = torch.roll(f, 2, -2)
    return fE, fW, fEE, fWW


def get_neighbors_z(f: torch.Tensor):
    """Interior neighbors in z (Dirichlet). Returns Nz-2 points."""
    fN = f[..., 2:]    # f[j+1] for j=1..Nz-2
    fS = f[..., :-2]   # f[j-1] for j=1..Nz-2
    return fN, fS


def get_neighbors_z_2nd(f: torch.Tensor):
    """2nd-order neighbors in z. Returns Nz-4 points."""
    fN = f[..., 3:-1]
    fS = f[..., 1:-3]
    fNN = f[..., 4:]
    fSS = f[..., :-4]
    return fN, fS, fNN, fSS


# ---- Face velocities ----
def face_velocities_x(ux: torch.Tensor):
    """Face velocities in x (periodic)."""
    uxE = torch.roll(ux, -1, -2)
    uxW = torch.roll(ux, 1, -2)
    uc_e = 0.5 * (ux + uxE)
    uc_w = 0.5 * (uxW + ux)
    return uc_e, uc_w


def face_velocities_z(uz: torch.Tensor):
    """Face velocities in z (interior). Returns Nz-1 faces."""
    vc_n = 0.5 * (uz[..., 1:] + uz[..., :-1])  # face between j and j+1
    return vc_n  # vc_s for cell j is vc_n for cell j-1


# ---- Continuity: face velocity divergence ----
def continuity_npinn(ux: torch.Tensor, uz: torch.Tensor,
                     dx: float, dz: float) -> torch.Tensor:
    """
    div(u) via face velocities (n-PINN style).
    Returns interior z points (Nz-2).
    """
    uc_e, uc_w = face_velocities_x(ux)
    div_x = (uc_e - uc_w) / dx

    vc_n = face_velocities_z(uz)  # Nz-1 faces
    div_z = (vc_n[..., 1:] - vc_n[..., :-1]) / dz  # Nz-2 interior

    return div_x[..., 1:-1] + div_z


# ---- 2nd-order upwind convection (from notebook) ----
def upwind_conv_x(phi: torch.Tensor, ux: torch.Tensor,
                  dx: float) -> torch.Tensor:
    """
    x-direction face-flux convection with 2nd-order upwind.
    (uc_e * phi_e - uc_w * phi_w) / dx
    """
    phiE, phiW, phiEE, phiWW = get_neighbors_x(phi)
    uc_e, uc_w = face_velocities_x(ux)

    # 2nd-order upwind at east face
    phi_e_pos = 1.5 * phi - 0.5 * torch.roll(phi, 1, -2)   # upwind from W
    phi_e_neg = 1.5 * phiE - 0.5 * phiEE                     # upwind from E
    phi_e = torch.where(uc_e >= 0, phi_e_pos, phi_e_neg)

    # 2nd-order upwind at west face
    phi_w_pos = 1.5 * torch.roll(phi, 1, -2) - 0.5 * phiWW  # upwind from WW
    phi_w_neg = 1.5 * phi - 0.5 * phiE                        # upwind from E
    phi_w = torch.where(uc_w >= 0, phi_w_pos, phi_w_neg)

    return (uc_e * phi_e - uc_w * phi_w) / dx


def upwind_conv_z(phi: torch.Tensor, uz: torch.Tensor,
                  dz: float) -> torch.Tensor:
    """
    z-direction face-flux convection with 2nd-order upwind.
    Interior only (Nz-4 points, skip 2 from each boundary).
    """
    vc_n = face_velocities_z(uz)  # Nz-1 faces

    # For interior cells j=2..Nz-3:
    # north face j: between j and j+1
    # south face j: between j-1 and j
    # Need phi[j-1], phi[j], phi[j+1], phi[j+2] for upwind at north
    # Need phi[j-2], phi[j-1], phi[j], phi[j+1] for upwind at south

    # phi_n at face j (between j and j+1): need phi[j-1..j+2]
    # upwind+ (vc_n >= 0): 1.5*phi[j] - 0.5*phi[j-1]
    # upwind- (vc_n < 0):  1.5*phi[j+1] - 0.5*phi[j+2]

    # For cells j=2..Nz-3 (output Nz-4 = 124 points):
    # North face index j, south face index j-1
    # vc_n has Nz-1=127 faces (indices 0..126)
    phi_n_pos = 1.5 * phi[..., 2:-2] - 0.5 * phi[..., 1:-3]   # 124
    phi_n_neg = 1.5 * phi[..., 3:-1] - 0.5 * phi[..., 4:]     # 124
    vc_n_j = vc_n[..., 2:-1]   # face j for j=2..Nz-3 → 124 faces
    phi_n = torch.where(vc_n_j >= 0, phi_n_pos, phi_n_neg)

    phi_s_pos = 1.5 * phi[..., 1:-3] - 0.5 * phi[..., :-4]    # 124
    phi_s_neg = 1.5 * phi[..., 2:-2] - 0.5 * phi[..., 3:-1]   # 124
    vc_s_j = vc_n[..., 1:-2]   # face j-1 for j=2..Nz-3 → 124 faces
    phi_s = torch.where(vc_s_j >= 0, phi_s_pos, phi_s_neg)

    return (vc_n_j * phi_n - vc_s_j * phi_s) / dz


# ---- 2nd-order central for diffusion ----
def laplacian_2nd(phi: torch.Tensor, dx: float, dz: float) -> torch.Tensor:
    """
    Laplacian: 2nd-order central. x periodic, z interior (Nz-2 points).
    """
    phiE = torch.roll(phi, -1, -2)
    phiW = torch.roll(phi, 1, -2)
    d2_dx2 = (phiE - 2*phi + phiW) / dx**2

    d2_dz2 = (phi[..., 2:] - 2*phi[..., 1:-1] + phi[..., :-2]) / dz**2

    return d2_dx2[..., 1:-1] + d2_dz2


# ---- Face-averaged pressure gradient (from notebook) ----
def pressure_grad_x(p: torch.Tensor, dx: float) -> torch.Tensor:
    """∂p/∂x via face average: (pe - pw)/dx where pe = (p+pE)/2"""
    pE = torch.roll(p, -1, -2)
    pW = torch.roll(p, 1, -2)
    pe = 0.5 * (p + pE)
    pw = 0.5 * (pW + p)
    return (pe - pw) / dx


def pressure_grad_z(p: torch.Tensor, dz: float) -> torch.Tensor:
    """∂p/∂z via face average (interior, Nz-2 points)."""
    pn = 0.5 * (p[..., 1:] + p[..., :-1])  # Nz-1 faces
    return (pn[..., 1:] - pn[..., :-1]) / dz  # Nz-2 cells


# ---- FD operators for comparison ----
def fd4_dx_periodic(f: torch.Tensor, dx: float) -> torch.Tensor:
    return (-torch.roll(f, -2, -2) + 8*torch.roll(f, -1, -2)
            - 8*torch.roll(f, 1, -2) + torch.roll(f, 2, -2)) / (12 * dx)


def fd2_dz_central(f: torch.Tensor, dz: float) -> torch.Tensor:
    return (f[..., 2:] - f[..., :-2]) / (2 * dz)


def fd2_d2z_central(f: torch.Tensor, dz: float) -> torch.Tensor:
    return (f[..., 2:] - 2*f[..., 1:-1] + f[..., :-2]) / dz**2


# ============================================================
# PDE residual tests
# ============================================================

def test_continuity(d: dict):
    ux, uz = d['ux'], d['uz']
    dx, dz = d['dx'], d['dz']

    # n-PINN
    R_np = continuity_npinn(ux, uz, dx, dz)

    # FD: 4th-x + 2nd-z central
    du_dx = fd4_dx_periodic(ux, dx)
    dw_dz = fd2_dz_central(uz, dz)
    R_fd = du_dx[..., 1:-1] + dw_dz

    return R_np, R_fd


def test_buoyancy(d: dict):
    """
    Official: ∂b/∂t + u·∇b = κΔb
    Residual: ∂b/∂t + CONV_b - κ*LAP_b = 0
    n-PINN CONV = face-flux conv - b*div (div correction)
    """
    b, ux, uz = d['b'], d['ux'], d['uz']
    dx, dz, dt = d['dx'], d['dz'], d['dt']
    kappa = d['kappa']

    # Time derivative (2nd-order central)
    db_dt = (b[2:] - b[:-2]) / (2 * dt)
    b_n, ux_n, uz_n = b[1:-1], ux[1:-1], uz[1:-1]

    # n-PINN: face-flux convection (Nz-4 interior points)
    conv_x = upwind_conv_x(b_n, ux_n, dx)  # full z
    conv_z = upwind_conv_z(b_n, uz_n, dz)  # Nz-4 points
    # div correction
    div_n = continuity_npinn(ux_n, uz_n, dx, dz)  # Nz-2
    # trim conv_x and div to match conv_z (Nz-4)
    conv_np = conv_x[..., 2:-2] + conv_z - b_n[..., 2:-2] * div_n[..., 1:-1]

    # Diffusion (2nd-order central, Nz-2 then trim to Nz-4)
    lap = laplacian_2nd(b_n, dx, dz)  # Nz-2
    diff = kappa * lap[..., 1:-1]     # Nz-4

    R_np = db_dt[..., 2:-2] + conv_np - diff

    # FD comparison
    db_dx = fd4_dx_periodic(b_n, dx)
    db_dz = fd2_dz_central(b_n, dz)
    conv_fd = ux_n[..., 1:-1] * db_dx[..., 1:-1] + uz_n[..., 1:-1] * db_dz
    lap_fd = laplacian_2nd(b_n, dx, dz)
    R_fd = db_dt[..., 1:-1] + conv_fd - kappa * lap_fd

    return R_np, R_fd


def test_momentum_x(d: dict):
    """
    Official: ∂ux/∂t + u·∇ux = -∂p/∂x + ν*Δux
    Residual: ∂ux/∂t + CONV_ux + ∂p/∂x - ν*LAP_ux = 0
    """
    ux, uz, p = d['ux'], d['uz'], d['p']
    dx, dz, dt = d['dx'], d['dz'], d['dt']
    nu = d['nu']

    dux_dt = (ux[2:] - ux[:-2]) / (2 * dt)
    ux_n, uz_n, p_n = ux[1:-1], uz[1:-1], p[1:-1]

    # n-PINN convection (Nz-4)
    conv_x = upwind_conv_x(ux_n, ux_n, dx)
    conv_z = upwind_conv_z(ux_n, uz_n, dz)
    div_n = continuity_npinn(ux_n, uz_n, dx, dz)
    conv_np = conv_x[..., 2:-2] + conv_z - ux_n[..., 2:-2] * div_n[..., 1:-1]

    # Pressure gradient (face-averaged)
    dpx_np = pressure_grad_x(p_n, dx)[..., 2:-2]

    # Diffusion
    lap_np = laplacian_2nd(ux_n, dx, dz)[..., 1:-1]

    R_np = dux_dt[..., 2:-2] + conv_np + dpx_np - nu * lap_np

    # FD comparison
    dux_dx = fd4_dx_periodic(ux_n, dx)
    dux_dz = fd2_dz_central(ux_n, dz)
    conv_fd = ux_n[..., 1:-1]*dux_dx[..., 1:-1] + uz_n[..., 1:-1]*dux_dz
    dpx_fd = fd4_dx_periodic(p_n, dx)[..., 1:-1]
    lap_fd = laplacian_2nd(ux_n, dx, dz)
    R_fd = dux_dt[..., 1:-1] + conv_fd + dpx_fd - nu * lap_fd

    return R_np, R_fd


def test_momentum_z(d: dict):
    """
    Official: ∂uz/∂t + u·∇uz = -∂p/∂z + ν*Δuz + b
    Residual: ∂uz/∂t + CONV_uz + ∂p/∂z - ν*LAP_uz - b = 0
    """
    ux, uz, p, b = d['ux'], d['uz'], d['p'], d['b']
    dx, dz, dt = d['dx'], d['dz'], d['dt']
    nu = d['nu']

    duz_dt = (uz[2:] - uz[:-2]) / (2 * dt)
    ux_n, uz_n, p_n, b_n = ux[1:-1], uz[1:-1], p[1:-1], b[1:-1]

    # n-PINN convection (Nz-4)
    conv_x = upwind_conv_x(uz_n, ux_n, dx)
    conv_z = upwind_conv_z(uz_n, uz_n, dz)
    div_n = continuity_npinn(ux_n, uz_n, dx, dz)
    conv_np = conv_x[..., 2:-2] + conv_z - uz_n[..., 2:-2] * div_n[..., 1:-1]

    # Pressure gradient z (face-averaged)
    dpz_np = pressure_grad_z(p_n, dz)[..., 1:-1]  # Nz-2 → trim to Nz-4

    # Diffusion
    lap_np = laplacian_2nd(uz_n, dx, dz)[..., 1:-1]

    R_np = duz_dt[..., 2:-2] + conv_np + dpz_np - nu * lap_np - b_n[..., 2:-2]

    # FD comparison
    duz_dx = fd4_dx_periodic(uz_n, dx)
    duz_dz = fd2_dz_central(uz_n, dz)
    conv_fd = ux_n[..., 1:-1]*duz_dx[..., 1:-1] + uz_n[..., 1:-1]*duz_dz
    dpz_fd = fd2_dz_central(p_n, dz)
    lap_fd = laplacian_2nd(uz_n, dx, dz)
    R_fd = duz_dt[..., 1:-1] + conv_fd + dpz_fd - nu * lap_fd - b_n[..., 1:-1]

    return R_np, R_fd


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else \
        'data/finetune/rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5'

    print(f"Dataset: {path}")

    for sample in [0, 1]:
        for t0, t1 in [(5, 25), (50, 70), (100, 120)]:
            print(f"\n{'='*70}")
            print(f"Sample {sample}, t=[{t0}, {t1})")
            print(f"{'='*70}")

            d = load_data(path, sample, t0, t1)
            print(f"Ra={d['Ra']:.0e}, Pr={d['Pr']}, κ={d['kappa']:.2e}, ν={d['nu']:.2e}")

            for eq_name, test_fn in [
                ('Continuity', test_continuity),
                ('Buoyancy', test_buoyancy),
                ('Mom-x', test_momentum_x),
                ('Mom-z', test_momentum_z),
            ]:
                if eq_name == 'Continuity':
                    R_np, R_fd = test_fn(d)
                else:
                    R_np, R_fd = test_fn(d)

                mse_np = R_np.pow(2).mean().item()
                mse_fd = R_fd.pow(2).mean().item()

                # Also test with boundary skip
                Nz_out = R_np.shape[-1]
                skip = min(10, Nz_out // 4)
                mse_np_int = R_np[..., skip:-skip].pow(2).mean().item()
                mse_fd_int = R_fd[..., skip:-skip].pow(2).mean().item()

                print(f"  {eq_name:12s} | n-PINN: {mse_np:.4e} (skip{skip}: {mse_np_int:.4e})"
                      f" | FD: {mse_fd:.4e} (skip{skip}: {mse_fd_int:.4e})")

        if sample > 0:
            break

    # Also test Pr=10
    path10 = path.replace('Prandtl_1.', 'Prandtl_10.')
    print(f"\n\n{'#'*70}")
    print(f"Prandtl=10: {path10}")
    print(f"{'#'*70}")
    d10 = load_data(path10, 0, 5, 25)
    print(f"Ra={d10['Ra']:.0e}, Pr={d10['Pr']}, κ={d10['kappa']:.2e}, ν={d10['nu']:.2e}")

    for eq_name, test_fn in [
        ('Continuity', test_continuity),
        ('Buoyancy', test_buoyancy),
        ('Mom-x', test_momentum_x),
        ('Mom-z', test_momentum_z),
    ]:
        R_np, R_fd = test_fn(d10)
        mse_np = R_np.pow(2).mean().item()
        mse_fd = R_fd.pow(2).mean().item()
        skip = 10
        mse_np_int = R_np[..., skip:-skip].pow(2).mean().item()
        mse_fd_int = R_fd[..., skip:-skip].pow(2).mean().item()
        print(f"  {eq_name:12s} | n-PINN: {mse_np:.4e} (skip{skip}: {mse_np_int:.4e})"
              f" | FD: {mse_fd:.4e} (skip{skip}: {mse_fd_int:.4e})")
