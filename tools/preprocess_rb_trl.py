"""
Preprocess Rayleigh-Bénard and Turbulent Radiative Layer datasets
into FinetuneDataset standard format.

Target format:
- vector: [N, T, H, W, 3] (fp32) — velocity, Vz padded with 0
- scalar: [N, T, H, W, C_s] (fp32) — scalar fields
- scalar_indices: [C_s] (int64) — channel index mapping
- nu: [N] (fp64) — per-sample parameter (ν for RB, tcool for TRL)
"""

import h5py
import numpy as np
import os
import glob
from pathlib import Path


def preprocess_rayleigh_benard(
    src_path: str,
    dst_path: str,
    delete_source: bool = True,
) -> None:
    """
    Convert Rayleigh-Bénard HDF5 to FinetuneDataset format.

    Source fields:
        t0_fields/buoyancy [N, T, 512, 128]
        t0_fields/pressure [N, T, 512, 128]
        t1_fields/velocity [N, T, 512, 128, 2]
    Attrs: Rayleigh, Prandtl

    Target:
        vector [N, T, 512, 128, 3] — (vx, vy, 0)
        scalar [N, T, 512, 128, 2] — (buoyancy, pressure)
        scalar_indices [0, 12]
        nu [N] — ν = (Ra/Pr)^{-1/2}
    """
    print(f"Processing Rayleigh-Bénard: {src_path}")

    with h5py.File(src_path, 'r') as src:
        N = src['t0_fields/buoyancy'].shape[0]
        T = src['t0_fields/buoyancy'].shape[1]
        Nx = src['t0_fields/buoyancy'].shape[2]
        Ny = src['t0_fields/buoyancy'].shape[3]
        Ra = float(src.attrs['Rayleigh'])
        Pr = float(src.attrs['Prandtl'])
        nu_val = (Ra / Pr) ** (-0.5)

        print(f"  N={N}, T={T}, grid={Nx}×{Ny}")
        print(f"  Ra={Ra:.0e}, Pr={Pr}, ν={nu_val:.6e}")

        with h5py.File(dst_path, 'w') as dst:
            # Pre-allocate datasets with chunking
            vec_ds = dst.create_dataset(
                'vector', shape=(N, T, Nx, Ny, 3), dtype='float32',
                chunks=(1, 1, Nx, Ny, 3),
            )
            sca_ds = dst.create_dataset(
                'scalar', shape=(N, T, Nx, Ny, 2), dtype='float32',
                chunks=(1, 1, Nx, Ny, 2),
            )
            dst.create_dataset('scalar_indices', data=np.array([0, 12], dtype=np.int64))
            dst.create_dataset('nu', data=np.full(N, nu_val, dtype=np.float64))

            # Copy sample by sample to manage memory
            for i in range(N):
                vel = src['t1_fields/velocity'][i]  # [T, Nx, Ny, 2]
                vec = np.zeros((T, Nx, Ny, 3), dtype=np.float32)
                vec[..., :2] = vel
                vec_ds[i] = vec

                buoy = src['t0_fields/buoyancy'][i]  # [T, Nx, Ny]
                pres = src['t0_fields/pressure'][i]   # [T, Nx, Ny]
                sca = np.stack([buoy, pres], axis=-1)  # [T, Nx, Ny, 2]
                sca_ds[i] = sca

                if (i + 1) % 10 == 0 or i == N - 1:
                    print(f"  Sample {i+1}/{N} done")

    print(f"  Saved to {dst_path}")

    if delete_source:
        os.remove(src_path)
        print(f"  Deleted source: {src_path}")


def preprocess_turbulent_radiative(
    src_dir: str,
    dst_path: str,
    delete_source: bool = True,
) -> None:
    """
    Merge and convert all TRL HDF5 files to FinetuneDataset format.

    Each source file:
        t0_fields/density  [8, 101, 128, 384]
        t0_fields/pressure [8, 101, 128, 384]
        t1_fields/velocity [8, 101, 128, 384, 2]
        attrs['tcool']

    Target:
        vector [N_total, T, 128, 384, 3] — (vx, vy, 0)
        scalar [N_total, T, 128, 384, 2] — (density, pressure)
        scalar_indices [4, 12]
        nu [N_total] — tcool per sample
    """
    src_files = sorted(glob.glob(os.path.join(src_dir, 'turbulent_radiative_layer_tcool_*.hdf5')))
    print(f"Processing Turbulent Radiative Layer: {len(src_files)} files")

    if not src_files:
        print("  No source files found!")
        return

    # First pass: count total samples and verify shapes
    total_n = 0
    file_info = []
    for fp in src_files:
        with h5py.File(fp, 'r') as f:
            n = f['t0_fields/density'].shape[0]
            T = f['t0_fields/density'].shape[1]
            Nx = f['t0_fields/density'].shape[2]
            Ny = f['t0_fields/density'].shape[3]
            tcool = float(f.attrs['tcool'])
            file_info.append((fp, n, T, Nx, Ny, tcool))
            total_n += n
            print(f"  {Path(fp).name}: N={n}, T={T}, grid={Nx}×{Ny}, tcool={tcool}")

    T = file_info[0][2]
    Nx = file_info[0][3]
    Ny = file_info[0][4]
    print(f"  Total samples: {total_n}, T={T}, grid={Nx}×{Ny}")

    with h5py.File(dst_path, 'w') as dst:
        vec_ds = dst.create_dataset(
            'vector', shape=(total_n, T, Nx, Ny, 3), dtype='float32',
            chunks=(1, 1, Nx, Ny, 3),
        )
        sca_ds = dst.create_dataset(
            'scalar', shape=(total_n, T, Nx, Ny, 2), dtype='float32',
            chunks=(1, 1, Nx, Ny, 2),
        )
        dst.create_dataset('scalar_indices', data=np.array([4, 12], dtype=np.int64))
        nu_ds = dst.create_dataset('nu', shape=(total_n,), dtype=np.float64)

        offset = 0
        for fp, n, _, _, _, tcool in file_info:
            print(f"  Converting {Path(fp).name} (samples {offset}..{offset+n-1})")

            with h5py.File(fp, 'r') as src:
                for i in range(n):
                    vel = src['t1_fields/velocity'][i]  # [T, Nx, Ny, 2]
                    vec = np.zeros((T, Nx, Ny, 3), dtype=np.float32)
                    vec[..., :2] = vel
                    vec_ds[offset + i] = vec

                    dens = src['t0_fields/density'][i]   # [T, Nx, Ny]
                    pres = src['t0_fields/pressure'][i]   # [T, Nx, Ny]
                    sca = np.stack([dens, pres], axis=-1)  # [T, Nx, Ny, 2]
                    sca_ds[offset + i] = sca

                    nu_ds[offset + i] = tcool

            offset += n

            if delete_source:
                os.remove(fp)
                print(f"  Deleted: {fp}")

    print(f"  Saved to {dst_path}")


if __name__ == '__main__':
    data_dir = 'data/finetune'

    # 1. Rayleigh-Bénard
    rb_src = os.path.join(data_dir, 'rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5')
    rb_dst = os.path.join(data_dir, 'rayleigh_benard_pr1.h5')
    if os.path.exists(rb_src):
        preprocess_rayleigh_benard(rb_src, rb_dst, delete_source=True)
    else:
        print(f"RB source not found: {rb_src}")

    # 2. Turbulent Radiative Layer (merge all tcool files)
    trl_dst = os.path.join(data_dir, 'turbulent_radiative_layer.h5')
    preprocess_turbulent_radiative(data_dir, trl_dst, delete_source=True)

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    for name, path in [('RB', rb_dst), ('TRL', trl_dst)]:
        if not os.path.exists(path):
            print(f"{name}: file not found")
            continue
        with h5py.File(path, 'r') as f:
            print(f"\n{name}: {path}")
            for k in f.keys():
                if hasattr(f[k], 'shape'):
                    print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")
            if 'nu' in f:
                nu = f['nu'][:]
                print(f"  nu: min={nu.min():.6e}, max={nu.max():.6e}, unique={len(np.unique(nu))}")
            if 'scalar_indices' in f:
                print(f"  scalar_indices: {f['scalar_indices'][:]}")
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  File size: {size_mb:.1f} MB")
