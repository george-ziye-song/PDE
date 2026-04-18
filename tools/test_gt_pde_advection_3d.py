"""GT PDE residual verification for 3D Advection dataset.

Computes per-equation RMS residuals to determine eq_scales
for normalized PDE loss.

Advection equation:
    u_t + a*u_x + b*u_y + c*u_z = 0

Two schemes:
  (A) 4th-order central u_x/u_y/u_z  [matches Advection3DPDELoss in training]
  (B) n-PINN 2nd-order upwind: a*(Fe-Fw)/dx per direction, sign-based face reconstruction

Usage:
    python tools/test_gt_pde_advection_3d.py \
        --data /scratch-share/SONG0304/finetune/advection_3d.hdf5
"""

import argparse
import os

import h5py
import numpy as np
import torch


# ── 4th-order central ──────────────────────────────────────────────────────────

def fd_4th(f: np.ndarray, axis: int, dx: float) -> np.ndarray:
    """4th-order central first derivative along given axis, periodic BC."""
    return (
        -np.roll(f, -2, axis=axis) + 8 * np.roll(f, -1, axis=axis)
        - 8 * np.roll(f,  1, axis=axis) + np.roll(f,  2, axis=axis)
    ) / (12 * dx)


def compute_residual_central(
    u: np.ndarray, dx: float, dy: float, dz: float, dt: float,
    a: float, b: float, c: float,
) -> np.ndarray:
    """Residual using 4th-order central spatial, 2nd-order central temporal."""
    du_dt = (u[2:] - u[:-2]) / (2 * dt)          # [T-2, X, Y, Z]
    u_n   = u[1:-1]                                # [T-2, X, Y, Z]
    R = du_dt + a * fd_4th(u_n, -3, dx) \
              + b * fd_4th(u_n, -2, dy) \
              + c * fd_4th(u_n, -1, dz)
    return R


# ── n-PINN 2nd-order upwind ───────────────────────────────────────────────────

def adv_upwind_2nd_1d(f: np.ndarray, v: float, dx: float, axis: int) -> np.ndarray:
    """n-PINN 2nd-order upwind advection term v*(Fe-Fw)/dx along one axis.

    Face reconstruction (same as Advection1DPDELoss._adv_upwind_2nd):
        v >= 0: Fe = 1.5*f - 0.5*f_{i-1},  Fw = 1.5*f_{i-1} - 0.5*f_{i-2}
        v <  0: Fe = 1.5*f_{i+1} - 0.5*f_{i+2}, Fw = 1.5*f - 0.5*f_{i+1}
    """
    f_ip1 = np.roll(f, -1, axis=axis)
    f_im1 = np.roll(f,  1, axis=axis)
    f_ip2 = np.roll(f, -2, axis=axis)
    f_im2 = np.roll(f,  2, axis=axis)
    if v >= 0:
        Fe = 1.5 * f    - 0.5 * f_im1
        Fw = 1.5 * f_im1 - 0.5 * f_im2
    else:
        Fe = 1.5 * f_ip1 - 0.5 * f_ip2
        Fw = 1.5 * f    - 0.5 * f_ip1
    return v * (Fe - Fw) / dx


def compute_residual_upwind(
    u: np.ndarray, dx: float, dy: float, dz: float, dt: float,
    a: float, b: float, c: float,
) -> np.ndarray:
    """Residual using n-PINN 2nd-order upwind spatial, 2nd-order central temporal."""
    du_dt = (u[2:] - u[:-2]) / (2 * dt)
    u_n   = u[1:-1]
    R = du_dt \
        + adv_upwind_2nd_1d(u_n, a, dx, axis=-3) \
        + adv_upwind_2nd_1d(u_n, b, dy, axis=-2) \
        + adv_upwind_2nd_1d(u_n, c, dz, axis=-1)
    return R


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GT PDE residual for 3D Advection")
    parser.add_argument(
        "--data", type=str,
        default="/scratch-share/SONG0304/finetune/advection_3d.hdf5",
    )
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    L = 2 * np.pi
    N_GRID = 64
    dx = dy = dz = L / N_GRID
    dt = 0.05

    print(f"Loading: {args.data}")
    with h5py.File(args.data, 'r') as f:
        scalar  = f['scalar']
        a_vals  = f['params_a'][:]
        b_vals  = f['params_b'][:]
        c_vals  = f['params_c'][:]
        n_total = scalar.shape[0]
        n_time  = scalar.shape[1]
        print(f"  Samples={n_total}, T={n_time}, "
              f"Grid={scalar.shape[2]}^3, dx={dx:.6f}, dt={dt}")
        print(f"  a range: [{a_vals.min():.4f}, {a_vals.max():.4f}]")
        print(f"  b range: [{b_vals.min():.4f}, {b_vals.max():.4f}]")
        print(f"  c range: [{c_vals.min():.4f}, {c_vals.max():.4f}]")

        n_proc = min(n_total, args.max_samples)
        rms_central  = []
        rms_upwind   = []
        per_t_central = None
        per_t_upwind  = None

        print(f"\nComputing residuals for {n_proc} samples...")
        print(f"  (A) 4th-order central  [matches training loss]")
        print(f"  (B) n-PINN 2nd-order upwind")

        for s in range(n_proc):
            u = np.array(scalar[s, :, :, :, :, 0], dtype=np.float64)  # [T,X,Y,Z]
            a_i, b_i, c_i = float(a_vals[s]), float(b_vals[s]), float(c_vals[s])

            R_c = compute_residual_central(u, dx, dy, dz, dt, a_i, b_i, c_i)
            R_u = compute_residual_upwind (u, dx, dy, dz, dt, a_i, b_i, c_i)

            rms_c = float(np.sqrt(np.mean(R_c**2)))
            rms_u = float(np.sqrt(np.mean(R_u**2)))
            rms_central.append(rms_c)
            rms_upwind .append(rms_u)

            pt_c = np.sqrt(np.mean(R_c**2, axis=(1, 2, 3)))  # [T-2]
            pt_u = np.sqrt(np.mean(R_u**2, axis=(1, 2, 3)))
            per_t_central = pt_c if per_t_central is None else per_t_central + pt_c
            per_t_upwind  = pt_u if per_t_upwind  is None else per_t_upwind  + pt_u

            if s < 3 or s == n_proc - 1:
                print(f"  [{s:3d}] a={a_i:.3f} b={b_i:.3f} c={c_i:.3f} | "
                      f"central={rms_c:.4e}  upwind={rms_u:.4e}")

    per_t_central /= n_proc
    per_t_upwind  /= n_proc

    arr_c = np.array(rms_central)
    arr_u = np.array(rms_upwind)

    print(f"\n{'='*70}")
    print(f"Summary ({n_proc} samples)")
    print(f"{'='*70}")
    print(f"  (A) 4th-order central:       mean={arr_c.mean():.6e}  std={arr_c.std():.4e}  max={arr_c.max():.4e}")
    print(f"  (B) n-PINN 2nd-order upwind: mean={arr_u.mean():.6e}  std={arr_u.std():.4e}  max={arr_u.max():.4e}")

    print(f"\n  >>> Using 4th-order central as eq_scales (matches Advection3DPDELoss)")
    print(f"\n  --- eq_scales for config ---")
    print(f"  physics:")
    print(f"    eq_scales:")
    print(f"      advection: {arr_c.mean():.4e}")

    gt_scales_dir = './data/gt_scales'
    os.makedirs(gt_scales_dir, exist_ok=True)
    save_path = os.path.join(gt_scales_dir, 'advection_3d_per_t.pt')
    torch.save({
        'advection':          torch.tensor(per_t_central, dtype=torch.float32),
        'scheme':             'central_4th',
        'mean_rms_central':   float(arr_c.mean()),
        'mean_rms_upwind2':   float(arr_u.mean()),
    }, save_path)
    print(f"\n  Saved per-timestep scales → {save_path}")
    print(f"    advection (central): shape={per_t_central.shape}, mean={per_t_central.mean():.6e}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
