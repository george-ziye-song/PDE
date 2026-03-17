"""
NS Incompressible Inhomogeneous — FD vs FFT derivative comparison on GT data.

Compare three methods for computing spatial derivatives:
  A. FD:  2nd-order central finite difference (current approach)
  B. FFT: Spectral derivative via FFT (treat grid as periodic, crop interior)
  C. DST: Discrete Sine Transform for velocity (proper Dirichlet BC u=v=0)

Equations verified (no pressure in GT):
  Eq.1 Continuity:  ∂u/∂x + ∂v/∂y = 0
  Eq.4 Tracer:      ∂d/∂t + u·∂d/∂x + v·∂d/∂y = 0
  Eq.5 Vorticity:   ∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y = ν·∇²ω + curl(d·f)

(Momentum skipped: no GT pressure → residual ≈ ∇p always.)
"""

import numpy as np
import h5py
from scipy.fft import dstn, idctn


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


# =============================================================================
# Derivative kernels
# =============================================================================

def fd_dx(f: np.ndarray, dx: float) -> np.ndarray:
    """2nd-order central FD: ∂f/∂x ≈ (f[i+1]-f[i-1])/(2dx). dims=-2 is x."""
    return (f[..., 2:, :] - f[..., :-2, :]) / (2 * dx)

def fd_dy(f: np.ndarray, dy: float) -> np.ndarray:
    """2nd-order central FD: ∂f/∂y ≈ (f[j+1]-f[j-1])/(2dy). dims=-1 is y."""
    return (f[..., 2:] - f[..., :-2]) / (2 * dy)

def fd_laplacian(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """2nd-order central FD Laplacian on interior [1:-1, 1:-1]."""
    lap_x = (f[..., 2:, 1:-1] - 2*f[..., 1:-1, 1:-1] + f[..., :-2, 1:-1]) / dx**2
    lap_y = (f[..., 1:-1, 2:] - 2*f[..., 1:-1, 1:-1] + f[..., 1:-1, :-2]) / dy**2
    return lap_x + lap_y


def fft_dx(f: np.ndarray, Lx: float) -> np.ndarray:
    """Spectral derivative ∂f/∂x via FFT along dims=-2."""
    N = f.shape[-2]
    kx = np.fft.fftfreq(N, d=Lx / N) * 2 * np.pi  # wavenumbers
    # Reshape for broadcasting: [..., N, 1]
    shape = [1] * (f.ndim - 2) + [N, 1]
    kx = kx.reshape(shape)
    f_hat = np.fft.fft(f, axis=-2)
    df_hat = 1j * kx * f_hat
    return np.real(np.fft.ifft(df_hat, axis=-2))

def fft_dy(f: np.ndarray, Ly: float) -> np.ndarray:
    """Spectral derivative ∂f/∂y via FFT along dims=-1."""
    N = f.shape[-1]
    ky = np.fft.fftfreq(N, d=Ly / N) * 2 * np.pi
    shape = [1] * (f.ndim - 1) + [N]
    ky = ky.reshape(shape)
    f_hat = np.fft.fft(f, axis=-1)
    df_hat = 1j * ky * f_hat
    return np.real(np.fft.ifft(df_hat, axis=-1))

def fft_laplacian(f: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Spectral Laplacian via 2D FFT."""
    Nx, Ny = f.shape[-2], f.shape[-1]
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
    kx_shape = [1] * (f.ndim - 2) + [Nx, 1]
    ky_shape = [1] * (f.ndim - 1) + [Ny]
    kx2 = (kx ** 2).reshape(kx_shape)
    ky2 = (ky ** 2).reshape(ky_shape)
    f_hat = np.fft.fft2(f, axes=(-2, -1))
    lap_hat = -(kx2 + ky2) * f_hat
    return np.real(np.fft.ifft2(lap_hat, axes=(-2, -1)))


def dst_dx(f: np.ndarray, Lx: float) -> np.ndarray:
    """
    Spectral derivative ∂f/∂x via DST for Dirichlet BC (f=0 at boundary).

    f: [..., Nx, Ny] where f[..., 0, :] = f[..., -1, :] = 0 (boundaries).
    Interior: f[..., 1:-1, :] → DST-I gives sine coefficients.
    Derivative of sine → cosine → IDCT-I.
    """
    Nx = f.shape[-2]
    Ni = Nx - 2  # interior points (excluding 2 boundary rows)
    f_int = f[..., 1:-1, :]  # [..., Ni, Ny]

    # DST-I (Type 1) along x-axis (axis=-2)
    # a_k = DST-I(f_interior)
    a_k = dstn(f_int, type=1, axes=[-2])

    # Wavenumbers for DST-I: k = 1, 2, ..., Ni
    # df/dx = sum(a_k * k*pi/Lx * cos(k*pi*x/Lx))
    k_vals = np.arange(1, Ni + 1).reshape([1] * (f.ndim - 2) + [Ni, 1])
    wavenumber = k_vals * np.pi / Lx

    # Derivative coefficients (sine → cosine)
    b_k = a_k * wavenumber

    # IDCT-I gives back the derivative at interior points
    df_int = idctn(b_k, type=1, axes=[-2])

    # Normalization: DST-I and IDCT-I normalization
    # scipy DST-I: unnormalized, need to divide by 2*(Ni+1)
    df_int = df_int / (2 * (Ni + 1))

    # Pad back to full grid (boundary derivative = finite diff approx)
    result = np.zeros_like(f)
    result[..., 1:-1, :] = df_int
    # Boundary: one-sided FD
    dx = Lx / (Nx - 1)
    result[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / dx
    result[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / dx
    return result

def dst_dy(f: np.ndarray, Ly: float) -> np.ndarray:
    """Spectral derivative ∂f/∂y via DST for Dirichlet BC (f=0 at boundary)."""
    Ny = f.shape[-1]
    Ni = Ny - 2
    f_int = f[..., 1:-1]

    a_k = dstn(f_int, type=1, axes=[-1])

    k_vals = np.arange(1, Ni + 1).reshape([1] * (f.ndim - 1) + [Ni])
    wavenumber = k_vals * np.pi / Ly

    b_k = a_k * wavenumber
    df_int = idctn(b_k, type=1, axes=[-1])
    df_int = df_int / (2 * (Ni + 1))

    result = np.zeros_like(f)
    result[..., 1:-1] = df_int
    dy = Ly / (Ny - 1)
    result[..., 0] = (f[..., 1] - f[..., 0]) / dy
    result[..., -1] = (f[..., -1] - f[..., -2]) / dy
    return result


# =============================================================================
# PDE residual computation
# =============================================================================

def compute_residuals(
    vx: np.ndarray, vy: np.ndarray,
    d: np.ndarray, fx: np.ndarray, fy: np.ndarray,
    Lx: float, Ly: float, dt: float, nu: float,
    method: str, skip: int = 2,
) -> dict:
    """
    Compute PDE residuals using specified derivative method.

    Args:
        method: "fd", "fft", or "dst"
    """
    T, H, W = vx.shape
    dx = Lx / H
    dy = Ly / W
    s = skip

    # Select derivative functions
    if method == "fd":
        def ddx(f):
            return fd_dx(f, dx)
        def ddy(f):
            return fd_dy(f, dy)
        def lap(f):
            return fd_laplacian(f, dx, dy)
    elif method == "fft":
        def ddx(f):
            return fft_dx(f, Lx)
        def ddy(f):
            return fft_dy(f, Ly)
        def lap(f):
            return fft_laplacian(f, Lx, Ly)
    elif method == "dst":
        def ddx(f):
            return dst_dx(f, Lx)
        def ddy(f):
            return dst_dy(f, Ly)
        def lap(f):
            # DST laplacian: apply DST derivative twice
            # For simplicity, use d²f/dx² + d²f/dy² via FFT laplacian on interior
            return fft_laplacian(f, Lx, Ly)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Time derivatives (2nd-order central) ──
    u_m = vx[1:-1]
    v_m = vy[1:-1]
    d_m = d[1:-1]
    du_dt = (vx[2:] - vx[:-2]) / (2 * dt)
    dv_dt = (vy[2:] - vy[:-2]) / (2 * dt)
    dd_dt = (d[2:] - d[:-2]) / (2 * dt)

    # ── Eq.1 Continuity ──
    if method == "fd":
        # Central FD: result shape [..., H-2, W] and [..., H, W-2]
        # Trim to common: [..., H-2, W-2]
        du_dx_full = fd_dx(vx, dx)  # [T, H-2, W]
        dv_dy_full = fd_dy(vy, dy)  # [T, H, W-2]
        div = du_dx_full[:, :, 1:-1] + dv_dy_full[:, 1:-1, :]  # [T, H-2, W-2]
        div_int = div[:, max(s-1,0):-(max(s-1,1)), max(s-1,0):-(max(s-1,1))]
    else:
        # FFT/DST: same shape as input
        du_dx_full = ddx(vx)
        dv_dy_full = ddy(vy)
        div = du_dx_full + dv_dy_full
        div_int = div[:, s:-s, s:-s]

    cont_rms = rms(div_int)

    # ── Eq.4 Tracer: dd/dt + u·dd/dx + v·dd/dy = 0 ──
    if method == "fd":
        dd_dx = fd_dx(d_m, dx)        # [Tm, H-2, W]
        dd_dy = fd_dy(d_m, dy)        # [Tm, H, W-2]
        # Common interior: [Tm, H-2, W-2]
        u_tr = u_m[:, 1:-1, 1:-1]
        v_tr = v_m[:, 1:-1, 1:-1]
        dd_dt_tr = du_dt[:, 1:-1, 1:-1]  # reuse shape but actually need dd_dt
        dd_dt_tr = dd_dt[:, 1:-1, 1:-1]
        R_d = dd_dt_tr + u_tr * dd_dx[:, :, 1:-1] + v_tr * dd_dy[:, 1:-1, :]
        # Further crop to skip boundary
        s2 = max(s - 1, 0)
        if s2 > 0:
            R_d = R_d[:, s2:-s2, s2:-s2]
            dd_dt_ref = dd_dt[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]
        else:
            dd_dt_ref = dd_dt[:, 1:-1, 1:-1]
    else:
        dd_dx = ddx(d_m)
        dd_dy = ddy(d_m)
        R_d = dd_dt + u_m * dd_dx + v_m * dd_dy
        R_d = R_d[:, s:-s, s:-s]
        dd_dt_ref = dd_dt[:, s:-s, s:-s]

    tracer_rms = rms(R_d)
    tracer_ref = rms(dd_dt_ref)

    # ── Eq.5 Vorticity: dw/dt + u·dw/dx + v·dw/dy = ν·∇²ω + curl(d·f) ──
    if method == "fd":
        dv_dx_w = fd_dx(vy, dx)    # [T, H-2, W]
        du_dy_w = fd_dy(vx, dy)    # [T, H, W-2]
        w_full = dv_dx_w[:, :, 1:-1] - du_dy_w[:, 1:-1, :]  # [T, H-2, W-2]
    else:
        dv_dx_w = ddx(vy)
        du_dy_w = ddy(vx)
        w_full = dv_dx_w - du_dy_w  # [T, H, W]

    if method == "fd":
        s2 = max(s - 1, 1)
        w_int = w_full[:, s2:-s2, s2:-s2]
        dw_dt = (w_int[2:] - w_int[:-2]) / (2 * dt)
        w_m = w_int[1:-1]

        u_w = vx[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]
        v_w = vy[:, 1:-1, 1:-1][:, s2:-s2, s2:-s2]

        dw_dx = fd_dx(w_m, dx)  # [..., Hw-2, Ww]
        dw_dy = fd_dy(w_m, dy)  # [..., Hw, Ww-2]
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
        curl_x = fd_dx(d_fy_w, dx)
        curl_y = fd_dy(d_fx_w, dy)
        frc_curl = curl_x[:, :, 1:-1] - curl_y[:, 1:-1, :]

        dw_dt_c = dw_dt[:, 1:-1, 1:-1]
        nt = dw_dt_c.shape[0]
        R_w = (dw_dt_c + u_c[:nt] * dw_dx_c[:nt] + v_c[:nt] * dw_dy_c[:nt]
               - nu * lap_w[:nt] - frc_curl[1:-1][:nt])
        R_w_nof = (dw_dt_c + u_c[:nt] * dw_dx_c[:nt] + v_c[:nt] * dw_dy_c[:nt]
                   - nu * lap_w[:nt])
        vort_ref = rms(dw_dt_c)
    else:
        # FFT/DST: full grid operations
        dw_dt = (w_full[2:] - w_full[:-2]) / (2 * dt)
        w_m = w_full[1:-1]
        u_mid = vx[1:-1]
        v_mid = vy[1:-1]

        dw_dx_f = ddx(w_m)
        dw_dy_f = ddy(w_m)
        lap_w = lap(w_m)

        d_fx = d * fx[None, :, :]
        d_fy = d * fy[None, :, :]
        curl_dfy_dx = ddx(d_fy)
        curl_dfx_dy = ddy(d_fx)
        frc_curl = curl_dfy_dx - curl_dfx_dy

        nt = dw_dt.shape[0]
        R_w_full = (dw_dt + u_mid[:nt] * dw_dx_f[:nt] + v_mid[:nt] * dw_dy_f[:nt]
                    - nu * lap_w[:nt] - frc_curl[1:-1][:nt])
        R_w_nof_full = (dw_dt + u_mid[:nt] * dw_dx_f[:nt] + v_mid[:nt] * dw_dy_f[:nt]
                        - nu * lap_w[:nt])
        # Crop to interior
        R_w = R_w_full[:, s:-s, s:-s]
        R_w_nof = R_w_nof_full[:, s:-s, s:-s]
        vort_ref = rms(dw_dt[:, s:-s, s:-s])

    return {
        'cont_rms': cont_rms,
        'tracer_rms': tracer_rms, 'tracer_ref': tracer_ref,
        'vort_rms': rms(R_w), 'vort_ref': vort_ref,
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
    skip = 4  # skip more for FFT to avoid Gibbs boundary artifacts
    t_start, t_end = 100, 130
    n_test = 3

    print("=" * 90)
    print("  NS Incompressible Inhomogeneous — FD vs FFT vs DST on GT Data")
    print("=" * 90)
    print(f"  ν={nu}, nx={nx}, Lx={Lx}, dx={dx:.6f}, dt={dt}")
    print(f"  Frames: {t_start}–{t_end}, skip_bl={skip}")
    print(f"  File: {data_path}")
    print()
    print("  Methods:")
    print("    A. FD:  2nd-order central finite difference")
    print("    B. FFT: Spectral derivative via FFT (treat as periodic)")
    print("    C. DST: Discrete Sine Transform (proper Dirichlet BC for velocity)")
    print()

    f = h5py.File(data_path, 'r')
    n_samples = f['velocity'].shape[0]
    n_test = min(n_samples, n_test)

    all_results = {m: [] for m in ["fd", "fft", "dst"]}

    for si in range(n_test):
        print(f"  Loading sample {si} ...")
        vx = f['velocity'][si, t_start:t_end, :, :, 0].astype(np.float64)
        vy = f['velocity'][si, t_start:t_end, :, :, 1].astype(np.float64)
        dd = f['particles'][si, t_start:t_end, :, :, 0].astype(np.float64)
        fx_arr = f['force'][si, :, :, 0].astype(np.float64)
        fy_arr = f['force'][si, :, :, 1].astype(np.float64)

        for method in ["fd", "fft", "dst"]:
            res = compute_residuals(
                vx, vy, dd, fx_arr, fy_arr,
                Lx, Ly, dt, nu, method=method, skip=skip,
            )
            all_results[method].append(res)

    f.close()

    # Average
    def avg(results: list, key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    avgs = {m: {k: avg(all_results[m], k) for k in all_results[m][0]} for m in all_results}

    # ================================================================
    # Comparison Table
    # ================================================================
    print()
    print("╔" + "═" * 94 + "╗")
    print("║" + "  COMPARISON TABLE: FD vs FFT vs DST".ljust(94) + "║")
    print("║" + f"  Averaged over {n_test} samples, frames {t_start}–{t_end}, skip_bl={skip}".ljust(94) + "║")
    print("╠" + "═" * 94 + "╣")

    hdr = f"║  {'Equation':<30} {'FD RMS':>11} {'FFT RMS':>11} {'DST RMS':>11} {'FD rel%':>8} {'FFT rel%':>8} {'DST rel%':>8}  ║"
    sep = "║  " + "─" * 90 + "  ║"
    print(hdr)
    print(sep)

    rows = [
        ("Eq.1 Continuity", 'cont_rms', None),
        ("Eq.4 Tracer",     'tracer_rms', 'tracer_ref'),
        ("Eq.5 Vorticity (+force)", 'vort_rms', 'vort_ref'),
        ("Eq.5 Vorticity (no force)", 'vort_nof_rms', 'vort_ref'),
    ]

    for label, rms_key, ref_key in rows:
        vals = {}
        rels = {}
        for m in ["fd", "fft", "dst"]:
            vals[m] = avgs[m][rms_key]
            if ref_key:
                rels[m] = f"{100 * vals[m] / (avgs[m][ref_key] + 1e-30):.2f}%"
            else:
                rels[m] = "—"

        print(f"║  {label:<30} {vals['fd']:>11.4e} {vals['fft']:>11.4e} {vals['dst']:>11.4e} "
              f"{rels['fd']:>8} {rels['fft']:>8} {rels['dst']:>8}  ║")

    print("╚" + "═" * 94 + "╝")

    # Find best method per equation
    print()
    print("  BEST METHOD PER EQUATION:")
    for label, rms_key, ref_key in rows:
        vals = {m: avgs[m][rms_key] for m in ["fd", "fft", "dst"]}
        best = min(vals, key=vals.get)
        ratio_vs_fd = vals[best] / (vals['fd'] + 1e-30)
        print(f"    {label:<30} → {best.upper():>4}  (RMS={vals[best]:.4e}, vs FD: {ratio_vs_fd:.4f}x)")

    # ================================================================
    # Per-sample detail
    # ================================================================
    print()
    print("=" * 90)
    print("  PER-SAMPLE DETAIL")
    print("=" * 90)

    for si in range(n_test):
        print(f"\n  ── Sample {si} ──")
        print(f"    {'Equation':<28} {'FD RMS':>10} {'FFT RMS':>10} {'DST RMS':>10} "
              f"{'FD%':>7} {'FFT%':>7} {'DST%':>7}")
        print(f"    {'─' * 80}")

        for label, rms_key, ref_key in rows:
            vals = {}
            rels = {}
            for m in ["fd", "fft", "dst"]:
                vals[m] = all_results[m][si][rms_key]
                if ref_key:
                    rels[m] = f"{100 * vals[m] / (all_results[m][si][ref_key] + 1e-30):.2f}%"
                else:
                    rels[m] = "—"
            print(f"    {label:<28} {vals['fd']:>10.4e} {vals['fft']:>10.4e} {vals['dst']:>10.4e} "
                  f"{rels['fd']:>7} {rels['fft']:>7} {rels['dst']:>7}")

    print()
    print("=" * 90)
    print("  NOTES")
    print("=" * 90)
    print("""
  FD:  Standard 2nd-order central finite difference. O(dx²) truncation error.
  FFT: Spectral derivative via FFT. Treats grid as periodic (Gibbs artifacts at boundary).
       Exponentially accurate for smooth periodic functions.
  DST: Discrete Sine Transform. Proper for Dirichlet BC (u=v=0 at walls).
       Uses DST for velocity derivatives; FFT for tracer/vorticity where BC is unknown.

  Momentum equations omitted: no GT pressure → residual ≈ ∇p (always large).
  Skip_bl=4 to avoid FFT Gibbs artifacts near boundary.
""")


if __name__ == '__main__':
    main()
