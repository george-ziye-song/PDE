"""
Test Darcy Flow GT PDE residual with different FVM schemes.

Darcy Flow equation: -div(a(x) * grad(u)) = f, f=1
Domain: [0,1]^2, cell-centered grid 128x128
BC: Dirichlet u=0 on boundary

Usage:
    python tools/test_darcy_pde.py --data data/finetune/2D_DarcyFlow_beta1.0_Train.hdf5
"""

import argparse
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--n_samples', type=int, default=20)
    return p.parse_args()


def fvm_residual_arithmetic(a: np.ndarray, u: np.ndarray, dx: float, dy: float, f: float = 1.0):
    """FVM with arithmetic mean face coefficients, interior only [1:-1, 1:-1]."""
    u_int = u[1:-1, 1:-1]
    a_int = a[1:-1, 1:-1]
    uE = u[2:, 1:-1]; uW = u[:-2, 1:-1]
    uN = u[1:-1, 2:]; uS = u[1:-1, :-2]
    aE = a[2:, 1:-1]; aW = a[:-2, 1:-1]
    aN = a[1:-1, 2:]; aS = a[1:-1, :-2]

    ae = 0.5 * (a_int + aE)
    aw = 0.5 * (aW + a_int)
    an = 0.5 * (a_int + aN)
    a_s = 0.5 * (aS + a_int)

    div = (ae * (uE - u_int) - aw * (u_int - uW)) / dx**2 \
        + (an * (uN - u_int) - a_s * (u_int - uS)) / dy**2
    return -div - f


def fvm_residual_harmonic(a: np.ndarray, u: np.ndarray, dx: float, dy: float, f: float = 1.0):
    """FVM with harmonic mean face coefficients (better for discontinuous a)."""
    u_int = u[1:-1, 1:-1]
    a_int = a[1:-1, 1:-1]
    uE = u[2:, 1:-1]; uW = u[:-2, 1:-1]
    uN = u[1:-1, 2:]; uS = u[1:-1, :-2]
    aE = a[2:, 1:-1]; aW = a[:-2, 1:-1]
    aN = a[1:-1, 2:]; aS = a[1:-1, :-2]

    eps = 1e-30
    ae = 2 * a_int * aE / (a_int + aE + eps)
    aw = 2 * aW * a_int / (aW + a_int + eps)
    an = 2 * a_int * aN / (a_int + aN + eps)
    a_s = 2 * aS * a_int / (aS + a_int + eps)

    div = (ae * (uE - u_int) - aw * (u_int - uW)) / dx**2 \
        + (an * (uN - u_int) - a_s * (u_int - uS)) / dy**2
    return -div - f


def fvm_residual_ghost(a: np.ndarray, u: np.ndarray, dx: float, dy: float, f: float = 1.0):
    """FVM with ghost cells for Dirichlet BC u=0, full domain."""
    nx, ny = u.shape

    # Ghost cell padding for Dirichlet u=0
    u_pad = np.zeros((nx + 2, ny + 2), dtype=u.dtype)
    u_pad[1:-1, 1:-1] = u
    u_pad[0, 1:-1] = -u[0, :]       # left ghost
    u_pad[-1, 1:-1] = -u[-1, :]     # right ghost
    u_pad[1:-1, 0] = -u[:, 0]       # bottom ghost
    u_pad[1:-1, -1] = -u[:, -1]     # top ghost

    # a padding (nearest extrapolation)
    a_pad = np.zeros((nx + 2, ny + 2), dtype=a.dtype)
    a_pad[1:-1, 1:-1] = a
    a_pad[0, 1:-1] = a[0, :]
    a_pad[-1, 1:-1] = a[-1, :]
    a_pad[1:-1, 0] = a[:, 0]
    a_pad[1:-1, -1] = a[:, -1]
    a_pad[0, 0] = a[0, 0]; a_pad[0, -1] = a[0, -1]
    a_pad[-1, 0] = a[-1, 0]; a_pad[-1, -1] = a[-1, -1]

    # Face coefficients (arithmetic)
    ae = 0.5 * (a_pad[1:-1, 1:-1] + a_pad[2:, 1:-1])
    aw = 0.5 * (a_pad[:-2, 1:-1] + a_pad[1:-1, 1:-1])
    an = 0.5 * (a_pad[1:-1, 1:-1] + a_pad[1:-1, 2:])
    a_s = 0.5 * (a_pad[1:-1, :-2] + a_pad[1:-1, 1:-1])

    flux_e = ae * (u_pad[2:, 1:-1] - u_pad[1:-1, 1:-1]) / dx
    flux_w = aw * (u_pad[1:-1, 1:-1] - u_pad[:-2, 1:-1]) / dx
    flux_n = an * (u_pad[1:-1, 2:] - u_pad[1:-1, 1:-1]) / dy
    flux_s = a_s * (u_pad[1:-1, 1:-1] - u_pad[1:-1, :-2]) / dy

    div = (flux_e - flux_w) / dx + (flux_n - flux_s) / dy
    return -div - f


def main():
    args = parse_args()

    with h5py.File(args.data, 'r') as fh:
        n_total = fh['nu'].shape[0]
        x = np.float64(fh['x-coordinate'][:])
        beta = fh.attrs.get('beta', '?')

    dx = float(x[1] - x[0])
    dy = dx
    n_test = min(args.n_samples, n_total)

    print(f"{'=' * 100}")
    print(f"  Darcy Flow GT PDE Residual: -div(a*grad(u)) = 1")
    print(f"  Data: {args.data} (beta={beta}, {n_total} samples, 128x128, dx={dx:.6f})")
    print(f"{'=' * 100}")

    header = (
        f"{'Samp':>4} | {'Arith MSE':>12} | {'Harm MSE':>12} | {'Ghost MSE':>12} | "
        f"{'Arith MAE':>12} | {'Harm MAE':>12} | {'Ghost MAE':>12}"
    )
    print(header)
    print("-" * len(header))

    all_arith, all_harm, all_ghost = [], [], []

    with h5py.File(args.data, 'r') as fh:
        for sid in range(n_test):
            a = np.float64(fh['nu'][sid])
            u = np.float64(fh['tensor'][sid, 0])

            res_a = fvm_residual_arithmetic(a, u, dx, dy)
            res_h = fvm_residual_harmonic(a, u, dx, dy)
            res_g = fvm_residual_ghost(a, u, dx, dy)

            mse_a = np.mean(res_a**2)
            mse_h = np.mean(res_h**2)
            mse_g = np.mean(res_g**2)
            mae_a = np.mean(np.abs(res_a))
            mae_h = np.mean(np.abs(res_h))
            mae_g = np.mean(np.abs(res_g))

            all_arith.append(mse_a)
            all_harm.append(mse_h)
            all_ghost.append(mse_g)

            print(f"{sid:4d} | {mse_a:12.4e} | {mse_h:12.4e} | {mse_g:12.4e} | "
                  f"{mae_a:12.4e} | {mae_h:12.4e} | {mae_g:12.4e}")

    # Analyze where residuals come from (discontinuity vs smooth regions)
    print(f"\n{'=' * 80}")
    print("  Discontinuity analysis (sample 0):")
    with h5py.File(args.data, 'r') as fh:
        a = np.float64(fh['nu'][0])
        u = np.float64(fh['tensor'][0, 0])

    a_int = a[1:-1, 1:-1]
    aE = a[2:, 1:-1]; aW = a[:-2, 1:-1]
    aN = a[1:-1, 2:]; aS = a[1:-1, :-2]
    disc_mask = (a_int != aE) | (a_int != aW) | (a_int != aN) | (a_int != aS)

    res_a = fvm_residual_arithmetic(a, u, dx, dy)
    res_h = fvm_residual_harmonic(a, u, dx, dy)

    n_disc = disc_mask.sum()
    n_smooth = (~disc_mask).sum()
    print(f"  Discontinuity cells: {n_disc}/{disc_mask.size} ({100*n_disc/disc_mask.size:.1f}%)")
    print(f"  Arith at discontinuity: MSE={np.mean(res_a[disc_mask]**2):.6e}")
    print(f"  Arith at smooth:        MSE={np.mean(res_a[~disc_mask]**2):.6e}")
    print(f"  Harm  at discontinuity: MSE={np.mean(res_h[disc_mask]**2):.6e}")
    print(f"  Harm  at smooth:        MSE={np.mean(res_h[~disc_mask]**2):.6e}")

    # Summary
    arr_a = np.array(all_arith)
    arr_h = np.array(all_harm)
    arr_g = np.array(all_ghost)

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY ({n_test} samples)")
    print(f"{'=' * 80}")
    print(f"  Arithmetic mean:  mean={np.mean(arr_a):.4e}  median={np.median(arr_a):.4e}  "
          f"min={np.min(arr_a):.4e}  max={np.max(arr_a):.4e}")
    print(f"  Harmonic mean:    mean={np.mean(arr_h):.4e}  median={np.median(arr_h):.4e}  "
          f"min={np.min(arr_h):.4e}  max={np.max(arr_h):.4e}")
    print(f"  Ghost cell:       mean={np.mean(arr_g):.4e}  median={np.median(arr_g):.4e}  "
          f"min={np.min(arr_g):.4e}  max={np.max(arr_g):.4e}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
