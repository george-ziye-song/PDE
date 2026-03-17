"""Analyze where Darcy Flow PDE residual is largest (spatial + discontinuity analysis)."""
import h5py
import numpy as np


def analyze_beta(beta_str: str):
    path = f'data/finetune/2D_DarcyFlow_beta{beta_str}_Train.hdf5'
    f_val = float(beta_str)

    with h5py.File(path, 'r') as fh:
        x = np.float64(fh['x-coordinate'][:])
        dx = float(x[1] - x[0])

        # Find worst sample in first 200
        mses = []
        for sid in range(200):
            a = np.float64(fh['nu'][sid])
            u = np.float64(fh['tensor'][sid, 0])
            u_int = u[1:-1, 1:-1]; a_int = a[1:-1, 1:-1]
            uE = u[2:, 1:-1]; uW = u[:-2, 1:-1]
            uN = u[1:-1, 2:]; uS = u[1:-1, :-2]
            aE = a[2:, 1:-1]; aW = a[:-2, 1:-1]
            aN = a[1:-1, 2:]; aS = a[1:-1, :-2]
            ae = 0.5 * (a_int + aE); aw = 0.5 * (aW + a_int)
            an = 0.5 * (a_int + aN); a_s = 0.5 * (aS + a_int)
            div = (ae * (uE - u_int) - aw * (u_int - uW)) / dx**2 \
                + (an * (uN - u_int) - a_s * (u_int - uS)) / dx**2
            res = (-div - f_val) ** 2
            mses.append(res.mean())

        mses_arr = np.array(mses)
        worst_idx = int(np.argmax(mses_arr))

        # Load worst sample
        a = np.float64(fh['nu'][worst_idx])
        u = np.float64(fh['tensor'][worst_idx, 0])

    # Compute residual
    u_int = u[1:-1, 1:-1]; a_int = a[1:-1, 1:-1]
    uE = u[2:, 1:-1]; uW = u[:-2, 1:-1]
    uN = u[1:-1, 2:]; uS = u[1:-1, :-2]
    aE = a[2:, 1:-1]; aW = a[:-2, 1:-1]
    aN = a[1:-1, 2:]; aS = a[1:-1, :-2]
    ae = 0.5 * (a_int + aE); aw = 0.5 * (aW + a_int)
    an = 0.5 * (a_int + aN); a_s = 0.5 * (aS + a_int)
    div = (ae * (uE - u_int) - aw * (u_int - uW)) / dx**2 \
        + (an * (uN - u_int) - a_s * (u_int - uS)) / dx**2
    res2 = (-div - f_val) ** 2  # (126, 126)

    # Max location
    max_pos = np.unravel_index(np.argmax(res2), res2.shape)
    hi, wi = max_pos
    h_orig, w_orig = hi + 1, wi + 1

    # Discontinuity mask
    disc_mask = ((a_int != aE) | (a_int != aW) | (a_int != aN) | (a_int != aS))
    disc_pct = disc_mask.sum() / disc_mask.size * 100
    disc_mse = res2[disc_mask].mean() if disc_mask.any() else 0
    smooth_mse = res2[~disc_mask].mean() if (~disc_mask).any() else 0

    # Top-10
    flat_idx = np.argsort(res2.ravel())[::-1][:10]
    top_locs = [np.unravel_index(fi, res2.shape) for fi in flat_idx]
    top_at_disc = sum(1 for h, w in top_locs if disc_mask[h, w])

    # Row / col average
    row_mse = res2.mean(axis=1)
    col_mse = res2.mean(axis=0)

    print(f"=== beta={beta_str}, worst sample={worst_idx} (of 200), MSE={mses_arr[worst_idx]:.4e} ===")
    print(f"  T=1 (steady state), H=128, W=128, interior=126x126")
    print(f"  Max residual: {res2.max():.4e} at original (h={h_orig}, w={w_orig})")
    print(f"    a(x) neighbors: C={a_int[hi,wi]:.1f} E={aE[hi,wi]:.1f} W={aW[hi,wi]:.1f} N={aN[hi,wi]:.1f} S={aS[hi,wi]:.1f}")
    print(f"    At discontinuity: {disc_mask[hi,wi]}")
    print(f"  Discontinuity cells: {disc_pct:.1f}% of interior")
    print(f"    disc MSE = {disc_mse:.4e}")
    print(f"    smooth MSE = {smooth_mse:.4e}")
    print(f"    ratio = {disc_mse / max(smooth_mse, 1e-30):.0f}x")
    print(f"  Top-10 worst locations (h,w in 128x128): "
          f"{[(h+1, w+1) for h, w in top_locs]}")
    print(f"    At discontinuity: {top_at_disc}/10")
    print(f"  Row MSE: max at row {np.argmax(row_mse)+1} ({row_mse.max():.4e})")
    print(f"  Col MSE: max at col {np.argmax(col_mse)+1} ({col_mse.max():.4e})")

    # Percentile analysis: what fraction of total error comes from discontinuity cells?
    total_err = res2.sum()
    disc_err = res2[disc_mask].sum() if disc_mask.any() else 0
    print(f"  Error attribution: disc cells = {disc_pct:.1f}% of area, "
          f"{100*disc_err/total_err:.1f}% of total error")
    print()


if __name__ == '__main__':
    for b in ['0.01', '0.1', '1.0']:
        analyze_beta(b)
