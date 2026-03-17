"""
NS Incompressible Inhomogeneous (PDEBench) — GT PDE Verification

Compare TWO numerical discretization methods on ground truth data:
  A. Standard FD: 2nd-order central difference, primitive (advective) form
  B. n-PINN:      face velocity + 2nd-order upwind, conservative (flux) form
                   (following n-pinn_lid_driven_cavity_re400.ipynb exactly)

Domain: [0,1]^2, no-slip walls, nu=0.01, dt=0.005, nx=ny=512
"""

import numpy as np
import h5py


def rms(x: np.ndarray) -> float:
    """Root Mean Square."""
    return float(np.sqrt(np.mean(x ** 2)))


def print_pde_equations() -> None:
    """Print all PDE equations with both discretization forms."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  NS Incompressible Inhomogeneous — PDE Equations                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Physical Domain: [0,1]^2, no-slip walls (u=v=0 on boundary)           ║
║  Parameters: ν = 0.01, dt = 0.005, dx = dy = 1/512                    ║
║  Force: time-invariant per-sample random spectral field f(x,y)         ║
║                                                                        ║
║  Eq.1  Continuity:                                                     ║
║        ∂u/∂x + ∂v/∂y = 0                                              ║
║                                                                        ║
║  Eq.2  x-Momentum:                                                     ║
║        ∂u/∂t + u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ν·∇²u + d·fx            ║
║                                                                        ║
║  Eq.3  y-Momentum:                                                     ║
║        ∂v/∂t + u·∂v/∂x + v·∂v/∂y = -∂p/∂y + ν·∇²v + d·fy            ║
║                                                                        ║
║  Eq.4  Tracer transport (passive scalar):                              ║
║        ∂d/∂t + u·∂d/∂x + v·∂d/∂y = 0                                  ║
║                                                                        ║
║  Eq.5  Vorticity (pressure-free, ω = ∂v/∂x - ∂u/∂y):                 ║
║        ∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y = ν·∇²ω + curl(d·f)               ║
║        where curl(d·f) = ∂(d·fy)/∂x - ∂(d·fx)/∂y                     ║
║                                                                        ║
║  NOTE: Pressure is NOT stored in GT data.                              ║
║        Momentum residual = ∂p/∂x or ∂p/∂y (always nonzero).           ║
║        Vorticity equation eliminates pressure → direct verification.   ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def print_method_a() -> None:
    """Print Method A discretization details."""
    print("""
┌──────────────────────────────────────────────────────────────────────────┐
│  Method A: Standard FD — 2nd-order central difference, PRIMITIVE form  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Time derivative:                                                      │
│    ∂f/∂t ≈ (f[t+1] - f[t-1]) / (2·dt)                                │
│                                                                        │
│  1st-order spatial derivative (central):                               │
│    ∂f/∂x ≈ (f[i+1,j] - f[i-1,j]) / (2·dx)                           │
│    ∂f/∂y ≈ (f[i,j+1] - f[i,j-1]) / (2·dy)                           │
│                                                                        │
│  Laplacian (2nd-order central):                                        │
│    ∇²f ≈ (fE - 2f + fW)/dx² + (fN - 2f + fS)/dy²                    │
│                                                                        │
│  Continuity: (uE-uW)/(2dx) + (vN-vS)/(2dy) = 0                      │
│                                                                        │
│  Convection (primitive/advective form):                                │
│    u·∂u/∂x + v·∂u/∂y  (NOT conservative ∇·(uu))                      │
│                                                                        │
│  Interior only: skip=2 from each wall boundary                         │
└──────────────────────────────────────────────────────────────────────────┘
""")


def print_method_b() -> None:
    """Print Method B discretization details."""
    print("""
┌──────────────────────────────────────────────────────────────────────────┐
│  Method B: n-PINN — Conservative upwind FD                             │
│  (Reference: n-pinn_lid_driven_cavity_re400.ipynb)                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Face velocities (cell-face averaged):                                 │
│    uc_e = 0.5·(uE + u),  uc_w = 0.5·(uW + u)                        │
│    vc_n = 0.5·(vN + v),  vc_s = 0.5·(vS + v)                        │
│                                                                        │
│  Face velocity divergence:                                             │
│    div = (uc_e - uc_w)/dx + (vc_n - vc_s)/dy                         │
│                                                                        │
│  2nd-order upwind face value interpolation:                            │
│    East face:                                                          │
│      uc_e ≥ 0 → Fe = 1.5·f - 0.5·fW   (upwind from west)            │
│      uc_e < 0 → Fe = 1.5·fE - 0.5·fEE (upwind from east)            │
│    West face:                                                          │
│      uc_w ≥ 0 → Fw = 1.5·fW - 0.5·fWW                               │
│      uc_w < 0 → Fw = 1.5·f  - 0.5·fE                                │
│    North/South: analogous in y-direction                               │
│                                                                        │
│  Conservative convection (flux form):                                  │
│    ∇·(fu) = (uc_e·Fe - uc_w·Fw)/dx + (vc_n·Fn - vc_s·Fs)/dy         │
│                                                                        │
│  Face-averaged pressure gradient:                                      │
│    ∂p/∂x ≈ ((p+pE)/2 - (pW+p)/2) / dx                               │
│    ∂p/∂y ≈ ((p+pN)/2 - (pS+p)/2) / dy                               │
│                                                                        │
│  Laplacian (2nd-order central, same as Method A):                      │
│    ∇²f ≈ (fE - 2f + fW)/dx² + (fN - 2f + fS)/dy²                    │
│                                                                        │
│  Div correction (n-PINN key feature):                                  │
│    Subtract f·div from each residual to restore consistency            │
│    Identity: ∇·(fu) = f·∇·u + u·∇f                                   │
│    Since ∇·u=0 exactly but div≠0 numerically, subtract f·div          │
│                                                                        │
│  Continuity:  R = div                                                  │
│  x-Momentum:  R = ∂u/∂t + ∇·(uu) + ∂p/∂x - ν∇²u - d·fx - u·div    │
│  y-Momentum:  R = ∂v/∂t + ∇·(vu) + ∂p/∂y - ν∇²v - d·fy - v·div    │
│  Tracer:      R = ∂d/∂t + ∇·(du) - d·div                             │
│                                                                        │
│  Interior only: skip=2 from each wall boundary                         │
└──────────────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# Method A: Standard FD (2nd-order central, primitive form)
# =============================================================================

def method_a_fd_central(
    vx: np.ndarray, vy: np.ndarray,
    d: np.ndarray, fx: np.ndarray, fy: np.ndarray,
    dx: float, dy: float, dt: float, nu: float, skip: int = 2,
) -> dict:
    """
    Standard 2nd-order central FD, primitive (advective) form.

    Continuity: ∂u/∂x + ∂v/∂y = 0
    Momentum:   ∂u/∂t + u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ν·∇²u + d·fx
    Tracer:     ∂d/∂t + u·∂d/∂x + v·∂d/∂y = 0
    Vorticity:  ∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y = ν·∇²ω + curl(d·f)

    All spatial derivatives: (f[i+1]-f[i-1]) / (2h)
    Time derivative: (f[t+1]-f[t-1]) / (2dt)
    """
    s = skip
    T, H, W = vx.shape

    # Helper: interior slicing
    def C(f):  return f[..., s:H-s, s:W-s]       # Center
    def E(f):  return f[..., s+1:H-s+1, s:W-s]   # East  (+1 x)
    def We(f): return f[..., s-1:H-s-1, s:W-s]   # West  (-1 x)
    def N(f):  return f[..., s:H-s, s+1:W-s+1]   # North (+1 y)
    def S(f):  return f[..., s:H-s, s-1:W-s-1]   # South (-1 y)

    # ── Eq.1 Continuity: (uE-uW)/(2dx) + (vN-vS)/(2dy) = 0 ──
    div = (E(vx) - We(vx)) / (2 * dx) + (N(vy) - S(vy)) / (2 * dy)
    cont_rms = rms(div)

    # ── Time derivatives ──
    u_m = vx[1:-1]   # mid-time
    v_m = vy[1:-1]
    d_m = d[1:-1]
    du_dt = (vx[2:] - vx[:-2]) / (2 * dt)
    dv_dt = (vy[2:] - vy[:-2]) / (2 * dt)
    dd_dt = (d[2:] - d[:-2]) / (2 * dt)

    # ── Eq.2 x-Momentum (primitive): du/dt + u·du/dx + v·du/dy - ν∇²u - d·fx ≈ dp/dx ──
    du_dx = (E(u_m) - We(u_m)) / (2 * dx)
    du_dy = (N(u_m) - S(u_m)) / (2 * dy)
    lap_u = (E(u_m) - 2*C(u_m) + We(u_m)) / dx**2 + \
            (N(u_m) - 2*C(u_m) + S(u_m)) / dy**2
    frc_u = C(d_m) * fx[None, s:H-s, s:W-s]

    R_u = C(du_dt) + C(u_m) * du_dx + C(v_m) * du_dy - nu * lap_u - frc_u

    # ── Eq.3 y-Momentum (primitive): dv/dt + u·dv/dx + v·dv/dy - ν∇²v - d·fy ≈ dp/dy ──
    dv_dx = (E(v_m) - We(v_m)) / (2 * dx)
    dv_dy = (N(v_m) - S(v_m)) / (2 * dy)
    lap_v = (E(v_m) - 2*C(v_m) + We(v_m)) / dx**2 + \
            (N(v_m) - 2*C(v_m) + S(v_m)) / dy**2
    frc_v = C(d_m) * fy[None, s:H-s, s:W-s]

    R_v = C(dv_dt) + C(u_m) * dv_dx + C(v_m) * dv_dy - nu * lap_v - frc_v

    # ── Eq.4 Tracer (primitive): dd/dt + u·dd/dx + v·dd/dy = 0 ──
    dd_dx = (E(d_m) - We(d_m)) / (2 * dx)
    dd_dy = (N(d_m) - S(d_m)) / (2 * dy)
    R_d = C(dd_dt) + C(u_m) * dd_dx + C(v_m) * dd_dy

    # ── Eq.5 Vorticity: ω = ∂v/∂x - ∂u/∂y ──
    # Compute ω on full grid first, then do FD on ω
    # ω needs one ring of neighbors → compute on [1:-1, 1:-1], then FD needs another
    dv_dx_full = (vy[:, 2:, :] - vy[:, :-2, :]) / (2 * dx)   # [T, H-2, W]
    du_dy_full = (vx[:, :, 2:] - vx[:, :, :-2]) / (2 * dy)   # [T, H, W-2]
    w_full = dv_dx_full[:, :, 1:-1] - du_dy_full[:, 1:-1, :]  # [T, H-2, W-2]

    # Interior of ω grid (skip from ω boundary)
    s2 = max(s - 1, 1)
    w_int = w_full[:, s2:-s2, s2:-s2]  # [T, H-2-2s2, W-2-2s2]

    dw_dt = (w_int[2:] - w_int[:-2]) / (2 * dt)
    w_m = w_int[1:-1]

    # u, v at ω grid
    u_w = vx[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]
    v_w = vy[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]

    # Spatial derivatives of ω (one more ring)
    dw_dx = (w_m[:, 2:, :] - w_m[:, :-2, :]) / (2 * dx)
    dw_dy = (w_m[:, :, 2:] - w_m[:, :, :-2]) / (2 * dy)
    dw_dx_c = dw_dx[:, :, 1:-1]
    dw_dy_c = dw_dy[:, 1:-1, :]
    u_c = u_w[1:-1, 1:-1, 1:-1]
    v_c = v_w[1:-1, 1:-1, 1:-1]

    lap_w_x = (w_m[:, 2:, :] - 2*w_m[:, 1:-1, :] + w_m[:, :-2, :]) / dx**2
    lap_w_y = (w_m[:, :, 2:] - 2*w_m[:, :, 1:-1] + w_m[:, :, :-2]) / dy**2
    lap_w = lap_w_x[:, :, 1:-1] + lap_w_y[:, 1:-1, :]

    # Force curl: ∂(d·fy)/∂x - ∂(d·fx)/∂y
    d_fx = d * fx[None, :, :]
    d_fy = d * fy[None, :, :]
    d_fx_w = d_fx[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]
    d_fy_w = d_fy[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]
    curl_x = (d_fy_w[:, 2:, :] - d_fy_w[:, :-2, :]) / (2 * dx)
    curl_y = (d_fx_w[:, :, 2:] - d_fx_w[:, :, :-2]) / (2 * dy)
    frc_curl = curl_x[:, :, 1:-1] - curl_y[:, 1:-1, :]

    dw_dt_c = dw_dt[:, 1:-1, 1:-1]
    nt = dw_dt_c.shape[0]

    R_w = (dw_dt_c + u_c[:nt] * dw_dx_c[:nt] + v_c[:nt] * dw_dy_c[:nt]
           - nu * lap_w[:nt] - frc_curl[1:-1][:nt])
    R_w_nof = (dw_dt_c + u_c[:nt] * dw_dx_c[:nt] + v_c[:nt] * dw_dy_c[:nt]
               - nu * lap_w[:nt])

    return {
        'cont_rms': cont_rms,
        'mom_x_rms': rms(R_u),   'mom_x_ref': rms(C(du_dt)),
        'mom_y_rms': rms(R_v),   'mom_y_ref': rms(C(dv_dt)),
        'tracer_rms': rms(R_d),  'tracer_ref': rms(C(dd_dt)),
        'vort_rms': rms(R_w),    'vort_ref': rms(dw_dt_c),
        'vort_nof_rms': rms(R_w_nof),
    }


# =============================================================================
# Method B: n-PINN conservative upwind FD
# (Reference: n-pinn_lid_driven_cavity_re400.ipynb)
# =============================================================================

def method_b_npinn_conservative(
    vx: np.ndarray, vy: np.ndarray,
    d: np.ndarray, fx: np.ndarray, fy: np.ndarray,
    dx: float, dy: float, dt: float, nu: float, skip: int = 2,
) -> dict:
    """
    n-PINN conservative upwind FD.

    From notebook (adapted to unsteady with force):
      continuity = div
      momentum_x = du/dt + UUx + VUy - ν(Uxx+Uyy) - d·fx - u·div  [+ dp/dx]
      momentum_y = dv/dt + UVx + VVy - ν(Vxx+Vyy) - d·fy - v·div  [+ dp/dy]
      tracer     = dd/dt + conv_d - d·div

    Where:
      uc_e = 0.5*(uE+u), uc_w = 0.5*(uW+u)
      vc_n = 0.5*(vN+v), vc_s = 0.5*(vS+v)
      div  = (uc_e-uc_w)/dx + (vc_n-vc_s)/dy

      2nd-order upwind at east face:
        uc_e >= 0: Fe = 1.5*f - 0.5*fW
        uc_e <  0: Fe = 1.5*fE - 0.5*fEE

      Conservative flux: (uc_e*Fe - uc_w*Fw)/dx + (vc_n*Fn - vc_s*Fs)/dy
      Div correction:    -f*div
    """
    s = skip
    T, H, W = vx.shape

    # Neighbor access for interior [s:-s, s:-s]
    def C(f):   return f[..., s:H-s, s:W-s]
    def E(f):   return f[..., s+1:H-s+1, s:W-s]
    def We(f):  return f[..., s-1:H-s-1, s:W-s]
    def N(f):   return f[..., s:H-s, s+1:W-s+1]
    def S(f):   return f[..., s:H-s, s-1:W-s-1]
    def EE(f):  return f[..., s+2:H-s+2, s:W-s]
    def WW(f):  return f[..., s-2:H-s-2, s:W-s]
    def NN(f):  return f[..., s:H-s, s+2:W-s+2]
    def SS(f):  return f[..., s:H-s, s-2:W-s-2]

    def face_vel(u: np.ndarray, v: np.ndarray):
        """Face velocities (notebook: uc_e = 0.5*(uE+u), etc.)"""
        uc_e = 0.5 * (E(u) + C(u))
        uc_w = 0.5 * (We(u) + C(u))
        vc_n = 0.5 * (N(v) + C(v))
        vc_s = 0.5 * (S(v) + C(v))
        return uc_e, uc_w, vc_n, vc_s

    def conservative_conv(
        f: np.ndarray,
        uc_e: np.ndarray, uc_w: np.ndarray,
        vc_n: np.ndarray, vc_s: np.ndarray,
    ) -> np.ndarray:
        """
        Conservative convection ∇·(fu) with 2nd-order upwind.

        Notebook reference (east face):
          Uem_uw2 = 1.5*u  - 0.5*uW   (uc_e >= 0, upwind from west)
          Uep_uw2 = 1.5*uE - 0.5*uEE  (uc_e <  0, upwind from east)
          Ue_uw2  = where(uc_e >= 0, Uem_uw2, Uep_uw2)

        Flux: UUx = (uc_e*Ue - uc_w*Uw) / dx
        """
        f_C = C(f);   f_E = E(f);   f_W = We(f)
        f_N = N(f);   f_S = S(f)
        f_EE = EE(f); f_WW = WW(f)
        f_NN = NN(f); f_SS = SS(f)

        # East face
        Fe = np.where(uc_e >= 0,
                      1.5 * f_C - 0.5 * f_W,       # upwind from west
                      1.5 * f_E - 0.5 * f_EE)       # upwind from east
        # West face
        Fw = np.where(uc_w >= 0,
                      1.5 * f_W - 0.5 * f_WW,       # upwind from west
                      1.5 * f_C - 0.5 * f_E)         # upwind from east
        # North face
        Fn = np.where(vc_n >= 0,
                      1.5 * f_C - 0.5 * f_S,        # upwind from south
                      1.5 * f_N - 0.5 * f_NN)       # upwind from north
        # South face
        Fs = np.where(vc_s >= 0,
                      1.5 * f_S - 0.5 * f_SS,       # upwind from south
                      1.5 * f_C - 0.5 * f_N)         # upwind from north

        return (uc_e * Fe - uc_w * Fw) / dx + (vc_n * Fn - vc_s * Fs) / dy

    def laplacian(f: np.ndarray) -> np.ndarray:
        """2nd-order central difference Laplacian (same as Method A)."""
        return (E(f) - 2*C(f) + We(f)) / dx**2 + (N(f) - 2*C(f) + S(f)) / dy**2

    # ── Time derivatives ──
    u_m = vx[1:-1];  v_m = vy[1:-1];  d_m = d[1:-1]
    du_dt = (vx[2:] - vx[:-2]) / (2 * dt)
    dv_dt = (vy[2:] - vy[:-2]) / (2 * dt)
    dd_dt = (d[2:] - d[:-2]) / (2 * dt)

    # ── Eq.1 Continuity: face velocity divergence ──
    uc_e, uc_w, vc_n, vc_s = face_vel(u_m, v_m)
    div = (uc_e - uc_w) / dx + (vc_n - vc_s) / dy
    cont_rms = rms(div)

    # ── Eq.2 x-Momentum: du/dt + ∇·(uu) - ν∇²u - d·fx - u·div [+ dp/dx] = 0 ──
    conv_u = conservative_conv(u_m, uc_e, uc_w, vc_n, vc_s)
    lap_u = laplacian(u_m)
    frc_u = C(d_m) * fx[None, s:H-s, s:W-s]

    # Notebook: R = UUx + VUy - 1/Re*(Uxx+Uyy) - u*div + Px
    # Our residual (no pressure): R = du/dt + conv_u - ν*lap_u - d*fx - u*div ≈ dp/dx
    R_u = C(du_dt) + conv_u - nu * lap_u - frc_u - C(u_m) * div

    # ── Eq.3 y-Momentum: dv/dt + ∇·(vu) - ν∇²v - d·fy - v·div [+ dp/dy] = 0 ──
    conv_v = conservative_conv(v_m, uc_e, uc_w, vc_n, vc_s)
    lap_v = laplacian(v_m)
    frc_v = C(d_m) * fy[None, s:H-s, s:W-s]

    R_v = C(dv_dt) + conv_v - nu * lap_v - frc_v - C(v_m) * div

    # ── Eq.4 Tracer: dd/dt + ∇·(du) - d·div = 0 ──
    conv_d = conservative_conv(d_m, uc_e, uc_w, vc_n, vc_s)
    R_d = C(dd_dt) + conv_d - C(d_m) * div

    # ── Eq.5 Vorticity: same FD on ω (no conservative form for ω) ──
    dv_dx_full = (vy[:, 2:, :] - vy[:, :-2, :]) / (2 * dx)
    du_dy_full = (vx[:, :, 2:] - vx[:, :, :-2]) / (2 * dy)
    w_full = dv_dx_full[:, :, 1:-1] - du_dy_full[:, 1:-1, :]

    s2 = max(s - 1, 1)
    w_int = w_full[:, s2:-s2, s2:-s2]

    dw_dt = (w_int[2:] - w_int[:-2]) / (2 * dt)
    w_m = w_int[1:-1]

    u_w = vx[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]
    v_w = vy[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]

    dw_dx = (w_m[:, 2:, :] - w_m[:, :-2, :]) / (2 * dx)
    dw_dy = (w_m[:, :, 2:] - w_m[:, :, :-2]) / (2 * dy)
    dw_dx_c = dw_dx[:, :, 1:-1]
    dw_dy_c = dw_dy[:, 1:-1, :]
    u_c = u_w[1:-1, 1:-1, 1:-1]
    v_c = v_w[1:-1, 1:-1, 1:-1]

    lap_w_x = (w_m[:, 2:, :] - 2*w_m[:, 1:-1, :] + w_m[:, :-2, :]) / dx**2
    lap_w_y = (w_m[:, :, 2:] - 2*w_m[:, :, 1:-1] + w_m[:, :, :-2]) / dy**2
    lap_w = lap_w_x[:, :, 1:-1] + lap_w_y[:, 1:-1, :]

    d_fx = d * fx[None, :, :]
    d_fy = d * fy[None, :, :]
    d_fx_w = d_fx[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]
    d_fy_w = d_fy[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]
    curl_x = (d_fy_w[:, 2:, :] - d_fy_w[:, :-2, :]) / (2 * dx)
    curl_y = (d_fx_w[:, :, 2:] - d_fx_w[:, :, :-2]) / (2 * dy)
    frc_curl = curl_x[:, :, 1:-1] - curl_y[:, 1:-1, :]

    dw_dt_c = dw_dt[:, 1:-1, 1:-1]
    nt = dw_dt_c.shape[0]

    R_w = (dw_dt_c + u_c[:nt] * dw_dx_c[:nt] + v_c[:nt] * dw_dy_c[:nt]
           - nu * lap_w[:nt] - frc_curl[1:-1][:nt])
    R_w_nof = (dw_dt_c + u_c[:nt] * dw_dx_c[:nt] + v_c[:nt] * dw_dy_c[:nt]
               - nu * lap_w[:nt])

    return {
        'cont_rms': cont_rms,
        'mom_x_rms': rms(R_u),   'mom_x_ref': rms(C(du_dt)),
        'mom_y_rms': rms(R_v),   'mom_y_ref': rms(C(dv_dt)),
        'tracer_rms': rms(R_d),  'tracer_ref': rms(C(dd_dt)),
        'vort_rms': rms(R_w),    'vort_ref': rms(dw_dt_c),
        'vort_nof_rms': rms(R_w_nof),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    data_path = '/scratch-share/SONG0304/finetune/ns_incom_inhom_2d_512-160.h5'
    nu = 0.01
    nx = ny = 512
    Lx = Ly = 1.0
    dx = Lx / nx
    dy = Ly / ny
    dt = 0.005
    skip = 2
    t_start, t_end = 100, 130
    n_test_samples = 3

    # ── Print equations and method descriptions ──
    print_pde_equations()
    print_method_a()
    print_method_b()

    print("=" * 78)
    print(f"  Running verification on GT data")
    print(f"  File: {data_path}")
    print(f"  ν={nu}, nx={nx}, dx={dx:.6f}, dy={dy:.6f}, dt={dt}")
    print(f"  Frames: {t_start}–{t_end}, skip_bl={skip}")
    print("=" * 78)

    f = h5py.File(data_path, 'r')
    n_samples = f['velocity'].shape[0]
    n_test = min(n_samples, n_test_samples)

    all_a = []
    all_b = []

    for si in range(n_test):
        print(f"\n  Loading sample {si} ...")
        vx = f['velocity'][si, t_start:t_end, :, :, 0].astype(np.float64)
        vy = f['velocity'][si, t_start:t_end, :, :, 1].astype(np.float64)
        dd = f['particles'][si, t_start:t_end, :, :, 0].astype(np.float64)
        fx = f['force'][si, :, :, 0].astype(np.float64)
        fy = f['force'][si, :, :, 1].astype(np.float64)
        print(f"    shape: vx={vx.shape}, fx={fx.shape}")
        print(f"    |u|_max={np.abs(vx).max():.4f}, |v|_max={np.abs(vy).max():.4f}")

        res_a = method_a_fd_central(vx, vy, dd, fx, fy, dx, dy, dt, nu, skip)
        res_b = method_b_npinn_conservative(vx, vy, dd, fx, fy, dx, dy, dt, nu, skip)
        all_a.append(res_a)
        all_b.append(res_b)

    f.close()

    # ── Average across samples ──
    def avg(results: list, key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    avg_a = {k: avg(all_a, k) for k in all_a[0]}
    avg_b = {k: avg(all_b, k) for k in all_b[0]}

    # ================================================================
    # Comparison Table
    # ================================================================
    print()
    print("╔" + "═" * 96 + "╗")
    print("║" + "  COMPARISON TABLE: Method A (FD central) vs Method B (n-PINN conservative)".ljust(96) + "║")
    print("║" + f"  Averaged over {n_test} samples, frames {t_start}–{t_end}, skip_bl={skip}".ljust(96) + "║")
    print("╠" + "═" * 96 + "╣")

    hdr = f"║  {'Equation':<36} {'A: FD RMS':>12} {'B: nPINN RMS':>13} {'A rel%':>8} {'B rel%':>8} {'Winner':>8}  ║"
    sep = "║  " + "─" * 92 + "  ║"
    print(hdr)
    print(sep)

    rows = [
        ("Eq.1 Continuity (∇·u = 0)",
         'cont_rms', None),
        ("Eq.2 x-Momentum (res ≈ ∂p/∂x)",
         'mom_x_rms', 'mom_x_ref'),
        ("Eq.3 y-Momentum (res ≈ ∂p/∂y)",
         'mom_y_rms', 'mom_y_ref'),
        ("Eq.4 Tracer (∂d/∂t + u·∇d = 0)",
         'tracer_rms', 'tracer_ref'),
        ("Eq.5 Vorticity (with force curl)",
         'vort_rms', 'vort_ref'),
        ("Eq.5 Vorticity (WITHOUT force)",
         'vort_nof_rms', 'vort_ref'),
    ]

    for label, rms_key, ref_key in rows:
        a_rms = avg_a[rms_key]
        b_rms = avg_b[rms_key]
        if ref_key:
            a_rel = f"{100 * a_rms / (avg_a[ref_key] + 1e-30):.2f}%"
            b_rel = f"{100 * b_rms / (avg_b[ref_key] + 1e-30):.2f}%"
        else:
            a_rel = "—"
            b_rel = "—"

        if abs(a_rms - b_rms) / (max(a_rms, b_rms) + 1e-30) < 0.01:
            winner = "~TIE"
        elif b_rms < a_rms:
            winner = "n-PINN"
        else:
            winner = "FD"

        print(f"║  {label:<36} {a_rms:>12.4e} {b_rms:>13.4e} {a_rel:>8} {b_rel:>8} {winner:>8}  ║")

    print("╚" + "═" * 96 + "╝")

    # ================================================================
    # Per-sample detail
    # ================================================================
    print()
    print("=" * 78)
    print("  PER-SAMPLE DETAIL")
    print("=" * 78)

    for si, (ra, rb) in enumerate(zip(all_a, all_b)):
        print(f"\n  ── Sample {si} ──")
        print(f"    {'Equation':<34} {'FD RMS':>11} {'nPINN RMS':>11} {'FD rel%':>9} {'nPINN%':>9}")
        print(f"    {'─' * 74}")

        detail_rows = [
            ("Continuity",          'cont_rms',    None),
            ("Momentum-x (≈dp/dx)", 'mom_x_rms',   'mom_x_ref'),
            ("Momentum-y (≈dp/dy)", 'mom_y_rms',   'mom_y_ref'),
            ("Tracer",              'tracer_rms',   'tracer_ref'),
            ("Vorticity (w/ force)",'vort_rms',     'vort_ref'),
            ("Vorticity (no force)",'vort_nof_rms', 'vort_ref'),
        ]

        for label, rms_key, ref_key in detail_rows:
            a_r = ra[rms_key]
            b_r = rb[rms_key]
            if ref_key:
                a_pct = f"{100 * a_r / (ra[ref_key] + 1e-30):.2f}%"
                b_pct = f"{100 * b_r / (rb[ref_key] + 1e-30):.2f}%"
            else:
                a_pct = "—"
                b_pct = "—"
            print(f"    {label:<34} {a_r:>11.4e} {b_r:>11.4e} {a_pct:>9} {b_pct:>9}")

    # ================================================================
    # Analysis
    # ================================================================
    print()
    print("=" * 78)
    print("  ANALYSIS")
    print("=" * 78)
    print("""
  1. Continuity: Both methods should give very small RMS.
     FD uses (uE-uW)/(2dx); n-PINN uses face velocity divergence.
     Both reduce to (uE-uW)/(2dx) algebraically → expect ~identical.

  2. Momentum (no pressure): Residual ≈ ∂p/∂x or ∂p/∂y.
     FD uses primitive form (u·∂u/∂x); n-PINN uses conservative (∇·(uu)-u·div).
     Smaller residual → more consistent pressure gradient recovery.

  3. Tracer: No diffusion, no source.
     FD: primitive (u·∂d/∂x); n-PINN: conservative (∇·(du)-d·div).
     n-PINN's div correction enforces mass conservation of tracer.

  4. Vorticity: Pressure-free → direct verification.
     Both methods use central FD on ω (no conservative form for ω).
     With force curl: ~7% → reasonable (scheme mismatch with solver).
     Without force: ~238% → force is ESSENTIAL.

  5. For PDE loss in training: n-PINN conservative form is preferred because:
     - Guarantees local flux conservation (important for LoRA fine-tuning)
     - Div correction provides consistency (∇·u=0 approximately)
     - Matches the formulation from the reference n-PINN paper
""")


if __name__ == '__main__':
    main()
