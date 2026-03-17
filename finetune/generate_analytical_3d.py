"""
3D 解析解 PDE 数据集生成器

包含:
1. 3D Taylor-Green Vortex (短时间精确)
2. 3D 热方程

输出格式:
    - vector: [N, T, D, H, W, 3]
    - scalar: [N, T, D, H, W, C_s]
"""

import numpy as np
import h5py
import torch
from pathlib import Path
from typing import Tuple
import argparse


def generate_taylor_green_3d_sample(
    nu: float,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    T: int = 100,
    dt: float = 0.01,
    Lx: float = 2 * np.pi,
    Ly: float = 2 * np.pi,
    Lz: float = 2 * np.pi,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 3D Taylor-Green 涡样本

    解析解 (短时间内精确):
        u = sin(x)cos(y)cos(z) * exp(-3νt)
        v = -cos(x)sin(y)cos(z) * exp(-3νt)
        w = 0
        p = (cos(2x) + cos(2y))(cos(2z) + 2) / 16 * exp(-6νt)

    Returns:
        u, v, w, p: [T, nx, ny, nz]
    """
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    z = np.linspace(0, Lz, nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    t_values = np.arange(T) * dt

    u = np.zeros((T, nx, ny, nz), dtype=np.float32)
    v = np.zeros((T, nx, ny, nz), dtype=np.float32)
    w = np.zeros((T, nx, ny, nz), dtype=np.float32)
    p = np.zeros((T, nx, ny, nz), dtype=np.float32)

    for i, t in enumerate(t_values):
        decay = np.exp(-3 * nu * t)
        u[i] = np.sin(X) * np.cos(Y) * np.cos(Z) * decay
        v[i] = -np.cos(X) * np.sin(Y) * np.cos(Z) * decay
        # w = 0
        p[i] = (np.cos(2*X) + np.cos(2*Y)) * (np.cos(2*Z) + 2) / 16 * decay**2

    return u, v, w, p


def generate_heat_3d_sample(
    alpha: float,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    T: int = 100,
    dt: float = 0.01,
    Lx: float = 2 * np.pi,
    Ly: float = 2 * np.pi,
    Lz: float = 2 * np.pi,
) -> np.ndarray:
    """
    生成 3D 热方程样本

    PDE: du/dt = alpha * (d2u/dx2 + d2u/dy2 + d2u/dz2)
    解析解: u = sin(x)sin(y)sin(z) * exp(-3*alpha*t)
    """
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    z = np.linspace(0, Lz, nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    t_values = np.arange(T) * dt

    u = np.zeros((T, nx, ny, nz), dtype=np.float32)
    for i, t in enumerate(t_values):
        u[i] = np.sin(X) * np.sin(Y) * np.sin(Z) * np.exp(-3 * alpha * t)

    return u


def verify_3d_taylor_green(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, p: np.ndarray,
    nu: float, dx: float, dy: float, dz: float, dt: float
) -> dict:
    """验证 3D Taylor-Green 涡"""
    u = torch.tensor(u, dtype=torch.float64)
    v = torch.tensor(v, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64)
    p = torch.tensor(p, dtype=torch.float64)

    # 1. 连续性方程 div(u,v,w) = 0
    uE = torch.roll(u, -1, dims=1)
    uW = torch.roll(u, 1, dims=1)
    vN = torch.roll(v, -1, dims=2)
    vS = torch.roll(v, 1, dims=2)
    wT = torch.roll(w, -1, dims=3)  # Top
    wB = torch.roll(w, 1, dims=3)   # Bottom

    # 面速度
    uc_e = 0.5 * (uE + u)
    uc_w = 0.5 * (uW + u)
    vc_n = 0.5 * (vN + v)
    vc_s = 0.5 * (vS + v)
    wc_t = 0.5 * (wT + w)
    wc_b = 0.5 * (wB + w)

    div = (uc_e - uc_w) / dx + (vc_n - vc_s) / dy + (wc_t - wc_b) / dz
    mse_div = torch.mean(div ** 2).item()

    # 2. x-动量方程
    du_dt = (u[2:] - u[:-2]) / (2 * dt)
    u_mid = u[1:-1]
    v_mid = v[1:-1]
    w_mid = w[1:-1]
    p_mid = p[1:-1]

    # 空间导数
    uE_m = torch.roll(u_mid, -1, dims=1)
    uW_m = torch.roll(u_mid, 1, dims=1)
    uN_m = torch.roll(u_mid, -1, dims=2)
    uS_m = torch.roll(u_mid, 1, dims=2)
    uT_m = torch.roll(u_mid, -1, dims=3)
    uB_m = torch.roll(u_mid, 1, dims=3)

    lap_u = ((uE_m - 2*u_mid + uW_m) / dx**2 +
             (uN_m - 2*u_mid + uS_m) / dy**2 +
             (uT_m - 2*u_mid + uB_m) / dz**2)

    pE = torch.roll(p_mid, -1, dims=1)
    pW = torch.roll(p_mid, 1, dims=1)
    dp_dx = (pE - pW) / (2 * dx)

    du_dx = (uE_m - uW_m) / (2 * dx)
    du_dy = (uN_m - uS_m) / (2 * dy)
    du_dz = (uT_m - uB_m) / (2 * dz)
    conv_u = u_mid * du_dx + v_mid * du_dy + w_mid * du_dz

    R_u = du_dt + conv_u + dp_dx - nu * lap_u
    mse_u = torch.mean(R_u ** 2).item()

    return {'div': mse_div, 'u_momentum': mse_u}


def verify_3d_heat(u: np.ndarray, alpha: float, dx: float, dy: float, dz: float, dt: float) -> float:
    """验证 3D 热方程"""
    u = torch.tensor(u, dtype=torch.float64)
    du_dt = (u[2:] - u[:-2]) / (2 * dt)
    u_mid = u[1:-1]

    uE = torch.roll(u_mid, -1, dims=1)
    uW = torch.roll(u_mid, 1, dims=1)
    uN = torch.roll(u_mid, -1, dims=2)
    uS = torch.roll(u_mid, 1, dims=2)
    uT = torch.roll(u_mid, -1, dims=3)
    uB = torch.roll(u_mid, 1, dims=3)

    lap_u = ((uE - 2*u_mid + uW) / dx**2 +
             (uN - 2*u_mid + uS) / dy**2 +
             (uT - 2*u_mid + uB) / dz**2)

    residual = du_dt - alpha * lap_u
    return torch.mean(residual ** 2).item()


def generate_3d_dataset(
    output_path: str,
    equation: str = 'taylor_green',
    n_samples: int = 20,
    param_range: Tuple[float, float] = (0.01, 0.1),
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    T: int = 100,
    dt: float = 0.01,
):
    """生成 3D PDE 数据集"""
    Lx = Ly = Lz = 2 * np.pi
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    print(f"Generating 3D {equation} dataset")
    print(f"  Samples: {n_samples}")
    print(f"  Param range: {param_range}")
    print(f"  Grid: {nx}x{ny}x{nz}, T={T}, dt={dt}")

    np.random.seed(42)
    log_min = np.log10(param_range[0])
    log_max = np.log10(param_range[1])
    params = 10 ** np.random.uniform(log_min, log_max, n_samples)

    with h5py.File(output_path, 'w') as f:
        if equation == 'taylor_green':
            # vector: [N, T, D, H, W, 3] - 注意 3D 数据维度顺序
            vector_ds = f.create_dataset(
                'vector',
                shape=(n_samples, T, nz, ny, nx, 3),
                dtype=np.float32,
                chunks=(1, min(T, 20), nz, ny, nx, 3),
                compression='gzip',
                compression_opts=4
            )
            scalar_ds = f.create_dataset(
                'scalar',
                shape=(n_samples, T, nz, ny, nx, 1),
                dtype=np.float32,
                chunks=(1, min(T, 20), nz, ny, nx, 1),
                compression='gzip',
                compression_opts=4
            )
            f.create_dataset('scalar_indices', data=np.array([12], dtype=np.int32))  # pressure
            f.create_dataset('nu', data=params.astype(np.float32))
            f.attrs['vector_dim'] = 3

        elif equation == 'heat':
            # Heat: 温度存入 scalar，scalar_indices=[14] (temperature)
            vector_ds = f.create_dataset(
                'vector',
                shape=(n_samples, T, nz, ny, nx, 3),
                dtype=np.float32,
                chunks=(1, min(T, 20), nz, ny, nx, 3),
                compression='gzip',
                compression_opts=4
            )
            scalar_ds = f.create_dataset(
                'scalar',
                shape=(n_samples, T, nz, ny, nx, 1),
                dtype=np.float32,
                chunks=(1, min(T, 20), nz, ny, nx, 1),
                compression='gzip',
                compression_opts=4
            )
            f.create_dataset('scalar_indices', data=np.array([14], dtype=np.int32))
            f.create_dataset('alpha', data=params.astype(np.float32))
            f.attrs['vector_dim'] = 0  # No actual velocity

        # 元数据
        f.attrs['equation'] = f'3D {equation}'
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nz'] = nz
        f.attrs['T'] = T
        f.attrs['dt'] = dt
        f.attrs['Lx'] = Lx
        f.attrs['Ly'] = Ly
        f.attrs['Lz'] = Lz
        f.attrs['boundary'] = 'periodic'

        for i in range(n_samples):
            param = params[i]

            if equation == 'taylor_green':
                u, v, w, p = generate_taylor_green_3d_sample(
                    nu=param, nx=nx, ny=ny, nz=nz, T=T, dt=dt
                )
                # 存储 (注意维度转置: [T, x, y, z] -> [T, z, y, x] for DHW)
                vector_ds[i, :, :, :, :, 0] = np.transpose(u, (0, 3, 2, 1))
                vector_ds[i, :, :, :, :, 1] = np.transpose(v, (0, 3, 2, 1))
                vector_ds[i, :, :, :, :, 2] = np.transpose(w, (0, 3, 2, 1))
                scalar_ds[i, :, :, :, :, 0] = np.transpose(p, (0, 3, 2, 1))

                results = verify_3d_taylor_green(u, v, w, p, param, dx, dy, dz, dt)
                mse = max(results.values())

            elif equation == 'heat':
                u = generate_heat_3d_sample(alpha=param, nx=nx, ny=ny, nz=nz, T=T, dt=dt)
                # 温度存入 scalar (vector 保持全零)
                scalar_ds[i, :, :, :, :, 0] = np.transpose(u, (0, 3, 2, 1))
                mse = verify_3d_heat(u, param, dx, dy, dz, dt)

            status = "PASS" if mse < 1e-5 else "FAIL"
            print(f"  [{i+1:3d}/{n_samples}] param={param:.6f}, MSE={mse:.2e} [{status}]")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nDone! File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--equation', type=str, default='taylor_green',
                       choices=['taylor_green', 'heat'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--param_min', type=float, default=0.01)
    parser.add_argument('--param_max', type=float, default=0.1)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=64)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    args = parser.parse_args()

    if args.output is None:
        args.output = f'data/{args.equation}_3d.h5'

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    generate_3d_dataset(
        output_path=args.output,
        equation=args.equation,
        n_samples=args.n_samples,
        param_range=(args.param_min, args.param_max),
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        T=args.T,
        dt=args.dt,
    )
