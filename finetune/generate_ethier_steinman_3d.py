"""
Ethier-Steinman 3D Navier-Stokes 精确解数据集生成器

来源: Ethier & Steinman (1994), "Exact fully 3D Navier-Stokes solutions for benchmarking"
https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.1650190502

解析解:
    u = -a [e^(ax) sin(ay+dz) + e^(az) cos(ax+dy)] × e^(-d²νt)
    v = -a [e^(ay) sin(az+dx) + e^(ax) cos(ay+dz)] × e^(-d²νt)
    w = -a [e^(az) sin(ax+dy) + e^(ay) cos(az+dx)] × e^(-d²νt)
    p = -a²/2 × [...] × e^(-2d²νt)

域: [-1, 1]³, Dirichlet 边界
参数: a = π/4, d = π/2
"""

import numpy as np
import h5py
import torch
from pathlib import Path
from typing import Tuple
import argparse


def generate_ethier_steinman_sample(
    nu: float,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    T: int = 100,
    dt: float = 0.01,
    a: float = np.pi / 4,
    d: float = np.pi / 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成单个 Ethier-Steinman 样本

    Returns:
        u, v, w, p: [T, nx, ny, nz]
    """
    # 空间网格 [-1, 1]³
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 时间数组
    t_values = np.arange(T) * dt

    # 预分配
    u = np.zeros((T, nx, ny, nz), dtype=np.float32)
    v = np.zeros((T, nx, ny, nz), dtype=np.float32)
    w = np.zeros((T, nx, ny, nz), dtype=np.float32)
    p = np.zeros((T, nx, ny, nz), dtype=np.float32)

    # 预计算空间项
    eax = np.exp(a * X)
    eay = np.exp(a * Y)
    eaz = np.exp(a * Z)

    sin_ay_dz = np.sin(a * Y + d * Z)
    sin_az_dx = np.sin(a * Z + d * X)
    sin_ax_dy = np.sin(a * X + d * Y)

    cos_ax_dy = np.cos(a * X + d * Y)
    cos_ay_dz = np.cos(a * Y + d * Z)
    cos_az_dx = np.cos(a * Z + d * X)

    # 速度空间部分 (不含时间衰减)
    u_space = -a * (eax * sin_ay_dz + eaz * cos_ax_dy)
    v_space = -a * (eay * sin_az_dx + eax * cos_ay_dz)
    w_space = -a * (eaz * sin_ax_dy + eay * cos_az_dx)

    # 压力空间部分
    p_space = -a**2 / 2 * (
        np.exp(2*a*X) + np.exp(2*a*Y) + np.exp(2*a*Z)
        + 2 * sin_ax_dy * cos_az_dx * np.exp(a*(Y+Z))
        + 2 * sin_ay_dz * cos_ax_dy * np.exp(a*(X+Z))
        + 2 * sin_az_dx * cos_ay_dz * np.exp(a*(X+Y))
    )

    # 生成时间序列
    for i, t in enumerate(t_values):
        decay = np.exp(-d**2 * nu * t)
        u[i] = u_space * decay
        v[i] = v_space * decay
        w[i] = w_space * decay
        p[i] = p_space * decay**2

    return u, v, w, p


def verify_pde_residual_npinn(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, p: np.ndarray,
    nu: float, dx: float, dy: float, dz: float, dt: float
) -> dict:
    """
    n-PINN 风格验证 3D Navier-Stokes 残差

    使用:
    1. 面速度散度
    2. 2阶迎风对流
    3. div 修正项
    """
    u = torch.tensor(u, dtype=torch.float64)
    v = torch.tensor(v, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64)
    p = torch.tensor(p, dtype=torch.float64)

    # =========== 1. 连续性方程 div(u) = 0 ===========
    # 面速度
    uE = torch.roll(u, -1, dims=1)
    uW = torch.roll(u, 1, dims=1)
    vN = torch.roll(v, -1, dims=2)
    vS = torch.roll(v, 1, dims=2)
    wT = torch.roll(w, -1, dims=3)
    wB = torch.roll(w, 1, dims=3)

    uc_e = 0.5 * (uE + u)
    uc_w = 0.5 * (uW + u)
    vc_n = 0.5 * (vN + v)
    vc_s = 0.5 * (vS + v)
    wc_t = 0.5 * (wT + w)
    wc_b = 0.5 * (wB + w)

    div = (uc_e - uc_w) / dx + (vc_n - vc_s) / dy + (wc_t - wc_b) / dz

    # 内部点 (排除边界)
    div_inner = div[:, 1:-1, 1:-1, 1:-1]
    mse_div = torch.mean(div_inner ** 2).item()

    # =========== 2. x-动量方程 ===========
    # 时间导数
    du_dt = (u[2:] - u[:-2]) / (2 * dt)
    u_mid = u[1:-1]
    v_mid = v[1:-1]
    w_mid = w[1:-1]
    p_mid = p[1:-1]
    div_mid = div[1:-1]

    # Laplacian (2阶中心差分)
    uE_m = torch.roll(u_mid, -1, dims=1)
    uW_m = torch.roll(u_mid, 1, dims=1)
    uN_m = torch.roll(u_mid, -1, dims=2)
    uS_m = torch.roll(u_mid, 1, dims=2)
    uT_m = torch.roll(u_mid, -1, dims=3)
    uB_m = torch.roll(u_mid, 1, dims=3)

    lap_u = ((uE_m - 2*u_mid + uW_m) / dx**2 +
             (uN_m - 2*u_mid + uS_m) / dy**2 +
             (uT_m - 2*u_mid + uB_m) / dz**2)

    # 压力梯度
    pE = torch.roll(p_mid, -1, dims=1)
    pW = torch.roll(p_mid, 1, dims=1)
    dp_dx = (pE - pW) / (2 * dx)

    # 2阶迎风对流 u*du/dx
    uWW_m = torch.roll(u_mid, 2, dims=1)
    uEE_m = torch.roll(u_mid, -2, dims=1)
    # 根据 uc_e 方向选择上风点
    uc_e_mid = 0.5 * (uE_m + u_mid)
    uc_w_mid = 0.5 * (uW_m + u_mid)
    Ue = torch.where(uc_e_mid >= 0, 1.5*u_mid - 0.5*uW_m, 1.5*uE_m - 0.5*uEE_m)
    Uw = torch.where(uc_w_mid >= 0, 1.5*uW_m - 0.5*uWW_m, 1.5*u_mid - 0.5*uE_m)
    UUx = (uc_e_mid * Ue - uc_w_mid * Uw) / dx

    # v*du/dy
    vc_n_mid = 0.5 * (torch.roll(v_mid, -1, dims=2) + v_mid)
    vc_s_mid = 0.5 * (torch.roll(v_mid, 1, dims=2) + v_mid)
    uNN_m = torch.roll(u_mid, -2, dims=2)
    uSS_m = torch.roll(u_mid, 2, dims=2)
    Un = torch.where(vc_n_mid >= 0, 1.5*u_mid - 0.5*uS_m, 1.5*uN_m - 0.5*uNN_m)
    Us = torch.where(vc_s_mid >= 0, 1.5*uS_m - 0.5*uSS_m, 1.5*u_mid - 0.5*uN_m)
    VUy = (vc_n_mid * Un - vc_s_mid * Us) / dy

    # w*du/dz
    wc_t_mid = 0.5 * (torch.roll(w_mid, -1, dims=3) + w_mid)
    wc_b_mid = 0.5 * (torch.roll(w_mid, 1, dims=3) + w_mid)
    uTT_m = torch.roll(u_mid, -2, dims=3)
    uBB_m = torch.roll(u_mid, 2, dims=3)
    Ut = torch.where(wc_t_mid >= 0, 1.5*u_mid - 0.5*uB_m, 1.5*uT_m - 0.5*uTT_m)
    Ub = torch.where(wc_b_mid >= 0, 1.5*uB_m - 0.5*uBB_m, 1.5*u_mid - 0.5*uT_m)
    WUz = (wc_t_mid * Ut - wc_b_mid * Ub) / dz

    # x-动量残差 (含 div 修正)
    R_u = du_dt + UUx + VUy + WUz + dp_dx - nu * lap_u - u_mid * div_mid

    # 内部点
    R_u_inner = R_u[:, 2:-2, 2:-2, 2:-2]
    mse_u = torch.mean(R_u_inner ** 2).item()

    return {
        'div': mse_div,
        'u_momentum': mse_u,
    }


def generate_dataset(
    output_path: str,
    n_samples: int = 10,
    nu_range: Tuple[float, float] = (0.01, 0.1),
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    T: int = 50,
    dt: float = 0.01,
):
    """生成完整数据集"""
    a = np.pi / 4
    d = np.pi / 2
    Lx = Ly = Lz = 2.0  # [-1, 1]
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dz = Lz / (nz - 1)

    print(f"Generating Ethier-Steinman 3D dataset")
    print(f"  Samples: {n_samples}")
    print(f"  nu range: {nu_range}")
    print(f"  Grid: {nx}x{ny}x{nz}, T={T}, dt={dt}")
    print(f"  dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")

    # Log-uniform nu
    np.random.seed(42)
    log_nu_min = np.log10(nu_range[0])
    log_nu_max = np.log10(nu_range[1])
    nu_values = 10 ** np.random.uniform(log_nu_min, log_nu_max, n_samples)

    with h5py.File(output_path, 'w') as f:
        # 创建数据集
        vector_ds = f.create_dataset(
            'vector',
            shape=(n_samples, T, nx, ny, nz, 3),
            dtype=np.float32,
            chunks=(1, min(T, 10), nx, ny, nz, 3),
            compression='gzip',
            compression_opts=4
        )
        scalar_ds = f.create_dataset(
            'scalar',
            shape=(n_samples, T, nx, ny, nz, 1),
            dtype=np.float32,
            chunks=(1, min(T, 10), nx, ny, nz, 1),
            compression='gzip',
            compression_opts=4
        )
        f.create_dataset('scalar_indices', data=np.array([12], dtype=np.int32))  # pressure
        f.create_dataset('nu', data=nu_values.astype(np.float32))
        f.attrs['vector_dim'] = 3

        # 元数据
        f.attrs['equation'] = 'Ethier-Steinman 3D (Navier-Stokes)'
        f.attrs['source'] = 'Ethier & Steinman 1994'
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nz'] = nz
        f.attrs['T'] = T
        f.attrs['dt'] = dt
        f.attrs['Lx'] = Lx
        f.attrs['Ly'] = Ly
        f.attrs['Lz'] = Lz
        f.attrs['dx'] = dx
        f.attrs['dy'] = dy
        f.attrs['dz'] = dz
        f.attrs['a'] = a
        f.attrs['d'] = d
        f.attrs['boundary'] = 'dirichlet'
        f.attrs['domain'] = '[-1,1]^3'

        # 生成样本
        for i in range(n_samples):
            nu = nu_values[i]
            u, v, w, p = generate_ethier_steinman_sample(
                nu=nu, nx=nx, ny=ny, nz=nz, T=T, dt=dt, a=a, d=d
            )

            vector_ds[i, :, :, :, :, 0] = u
            vector_ds[i, :, :, :, :, 1] = v
            vector_ds[i, :, :, :, :, 2] = w
            scalar_ds[i, :, :, :, :, 0] = p

            # 验证
            results = verify_pde_residual_npinn(u, v, w, p, nu, dx, dy, dz, dt)
            status = "PASS" if all(val < 1e-5 for val in results.values()) else "FAIL"
            print(f"  [{i+1:3d}/{n_samples}] nu={nu:.6f}, "
                  f"div={results['div']:.2e}, u_mom={results['u_momentum']:.2e} [{status}]")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nDone! File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/ethier_steinman_3d.h5')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--nu_min', type=float, default=0.01)
    parser.add_argument('--nu_max', type=float, default=0.1)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=64)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--dt', type=float, default=0.01)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    generate_dataset(
        output_path=args.output,
        n_samples=args.n_samples,
        nu_range=(args.nu_min, args.nu_max),
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        T=args.T,
        dt=args.dt,
    )
