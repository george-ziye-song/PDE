"""
3D PDE 数据可视化脚本

可视化 Heat 3D 和 Ethier-Steinman 3D 数据集
显示不同样本、不同时间步的切片
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def visualize_heat_3d(
    data_path: str,
    sample_indices: list = [0, 1, 2],
    time_indices: list = [0, 10, 25, 49],
    z_slice: int = None,
    output_path: str = None,
):
    """
    可视化 3D Heat 数据

    Args:
        data_path: HDF5 文件路径
        sample_indices: 要可视化的样本索引
        time_indices: 要可视化的时间步索引
        z_slice: z 切片位置 (默认中间)
        output_path: 输出图片路径
    """
    with h5py.File(data_path, 'r') as f:
        print(f"=== Heat 3D: {data_path} ===")
        print(f"Keys: {list(f.keys())}")
        print(f"Attrs: {dict(f.attrs)}")

        scalar = f['scalar'][:]  # [N, T, D, H, W, 1]
        alpha = f['alpha'][:] if 'alpha' in f else None

        N, T, D, H, W, C = scalar.shape
        print(f"Scalar shape: {scalar.shape}")
        print(f"N={N}, T={T}, D={D}, H={H}, W={W}")

        if z_slice is None:
            z_slice = D // 4  # D//2 = π for [0,2π], sin(π)=0; D//4 = π/2, sin(π/2)=1

        # 温度在 scalar (vector_dim=0, vector 全零)
        u = scalar[:, :, :, :, :, 0]  # [N, T, D, H, W]

        n_samples = min(len(sample_indices), N)
        n_times = len(time_indices)

        fig, axes = plt.subplots(n_samples, n_times, figsize=(4 * n_times, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, si in enumerate(sample_indices[:n_samples]):
            for j, ti in enumerate(time_indices):
                if ti >= T:
                    ti = T - 1

                ax = axes[i, j]
                data = u[si, ti, z_slice, :, :]  # [H, W]

                vmax = max(abs(data.min()), abs(data.max()))
                vmin = -vmax if vmax > 0 else -1

                im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046)

                alpha_val = alpha[si] if alpha is not None else '?'
                ax.set_title(f'Sample {si}, t={ti}\nα={alpha_val:.4f}')
                ax.set_xlabel('W')
                ax.set_ylabel('H')

        plt.suptitle(f'Heat 3D (z={z_slice})\nmin={u.min():.4f}, max={u.max():.4f}', fontsize=14)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        else:
            plt.show()

        # 打印统计
        print(f"\n=== 数值统计 ===")
        print(f"Global: min={u.min():.6f}, max={u.max():.6f}")
        for si in sample_indices[:n_samples]:
            print(f"Sample {si}: min={u[si].min():.6f}, max={u[si].max():.6f}")


def visualize_ethier_steinman_3d(
    data_path: str,
    sample_indices: list = [0, 1, 2],
    time_indices: list = [0, 25, 50, 99],
    z_slice: int = None,
    output_path: str = None,
):
    """
    可视化 Ethier-Steinman 3D 数据

    Args:
        data_path: HDF5 文件路径
        sample_indices: 要可视化的样本索引
        time_indices: 要可视化的时间步索引
        z_slice: z 切片位置 (默认中间)
        output_path: 输出图片路径
    """
    with h5py.File(data_path, 'r') as f:
        print(f"=== Ethier-Steinman 3D: {data_path} ===")
        print(f"Keys: {list(f.keys())}")
        print(f"Attrs: {dict(f.attrs)}")

        vector = f['vector'][:]  # [N, T, D, H, W, 3]
        scalar = f['scalar'][:] if 'scalar' in f else None
        nu = f['nu'][:] if 'nu' in f else None

        N, T, D, H, W, C = vector.shape
        print(f"Vector shape: {vector.shape}")
        if scalar is not None:
            print(f"Scalar shape: {scalar.shape}")

        if z_slice is None:
            z_slice = D // 2

        u = vector[:, :, :, :, :, 0]  # u 分量
        v = vector[:, :, :, :, :, 1]  # v 分量
        w = vector[:, :, :, :, :, 2]  # w 分量

        n_samples = min(len(sample_indices), N)
        n_times = len(time_indices)

        # 可视化 u 分量
        fig, axes = plt.subplots(n_samples, n_times, figsize=(4 * n_times, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, si in enumerate(sample_indices[:n_samples]):
            for j, ti in enumerate(time_indices):
                if ti >= T:
                    ti = T - 1

                ax = axes[i, j]
                data = u[si, ti, z_slice, :, :]

                vmax = max(abs(data.min()), abs(data.max()))
                vmin = -vmax if vmax > 0 else -1

                im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046)

                nu_val = nu[si] if nu is not None else '?'
                ax.set_title(f'Sample {si}, t={ti}\nν={nu_val:.4f}')
                ax.set_xlabel('W')
                ax.set_ylabel('H')

        plt.suptitle(f'Ethier-Steinman 3D - u velocity (z={z_slice})\nmin={u.min():.4f}, max={u.max():.4f}', fontsize=14)
        plt.tight_layout()

        if output_path:
            out_u = output_path.replace('.png', '_u.png')
            plt.savefig(out_u, dpi=150, bbox_inches='tight')
            print(f"Saved: {out_u}")
        else:
            plt.show()

        # 可视化速度幅值
        speed = np.sqrt(u**2 + v**2 + w**2)

        fig2, axes2 = plt.subplots(n_samples, n_times, figsize=(4 * n_times, 4 * n_samples))
        if n_samples == 1:
            axes2 = axes2.reshape(1, -1)

        for i, si in enumerate(sample_indices[:n_samples]):
            for j, ti in enumerate(time_indices):
                if ti >= T:
                    ti = T - 1

                ax = axes2[i, j]
                data = speed[si, ti, z_slice, :, :]

                im = ax.imshow(data, cmap='viridis', vmin=0, vmax=data.max())
                plt.colorbar(im, ax=ax, fraction=0.046)

                nu_val = nu[si] if nu is not None else '?'
                ax.set_title(f'Sample {si}, t={ti}\nν={nu_val:.4f}')

        plt.suptitle(f'Ethier-Steinman 3D - Speed |v| (z={z_slice})\nmax={speed.max():.4f}', fontsize=14)
        plt.tight_layout()

        if output_path:
            out_speed = output_path.replace('.png', '_speed.png')
            plt.savefig(out_speed, dpi=150, bbox_inches='tight')
            print(f"Saved: {out_speed}")
        else:
            plt.show()

        # 打印统计
        print(f"\n=== 数值统计 ===")
        print(f"u: min={u.min():.6f}, max={u.max():.6f}")
        print(f"v: min={v.min():.6f}, max={v.max():.6f}")
        print(f"w: min={w.min():.6f}, max={w.max():.6f}")
        print(f"speed: max={speed.max():.6f}")

        for si in sample_indices[:n_samples]:
            print(f"Sample {si}: u=[{u[si].min():.4f}, {u[si].max():.4f}], speed_max={speed[si].max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize 3D PDE datasets')
    parser.add_argument('--heat', type=str, default=None, help='Heat 3D HDF5 path')
    parser.add_argument('--ethier', type=str, default=None, help='Ethier-Steinman 3D HDF5 path')
    parser.add_argument('--samples', type=int, nargs='+', default=[0, 1, 2], help='Sample indices')
    parser.add_argument('--times', type=int, nargs='+', default=[0, 10, 25, 49], help='Time indices')
    parser.add_argument('--z', type=int, default=None, help='Z slice index')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    args = parser.parse_args()

    if args.heat:
        out = args.output or args.heat.replace('.h5', '_vis.png')
        visualize_heat_3d(
            args.heat,
            sample_indices=args.samples,
            time_indices=args.times,
            z_slice=args.z,
            output_path=out,
        )

    if args.ethier:
        out = args.output or args.ethier.replace('.h5', '_vis.png')
        visualize_ethier_steinman_3d(
            args.ethier,
            sample_indices=args.samples,
            time_indices=args.times,
            z_slice=args.z,
            output_path=out,
        )

    if not args.heat and not args.ethier:
        print("Usage:")
        print("  python visualize_3d_data.py --heat data/finetune/heat_3d_128.h5")
        print("  python visualize_3d_data.py --ethier data/ethier_steinman_3d.h5")
        print("  python tools/visualize_3d_data.py --heat data/finetune/heat_3d_128.h5 --samples 0 1 2 --times 0 10 25 49")
