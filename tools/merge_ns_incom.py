"""
Merge multiple PDEBench raw ns_incom H5 files into our standard format file.

Since the original file has fixed maxshape, we create a new file with all data.

Source (PDEBench raw, one or more):
  - velocity: (N, 1000, 512, 512, 2)  → vx, vy
  - particles: (N, 1000, 512, 512, 1) → passive tracer
  - force: (N, 512, 512, 2)

Target (our format):
  - vector: (N, 1000, 512, 512, 3)    → vx, vy, 0
  - scalar: (N, 1000, 512, 512, 1)    → passive_tracer (idx=11)
  - scalar_indices: [11]
  - force: (N, 512, 512, 2)

Usage:
    python tools/merge_ns_incom.py \
        --src file1.h5 file2.h5 \
        --dst /scratch-share/SONG0304/pretrained/ns_incom_inhom_2d_512.hdf5 \
        --delete_src
"""

import argparse
import os
import shutil
import h5py
import numpy as np
from tqdm import tqdm
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="Merge new ns_incom samples into existing file")
    p.add_argument('--src', type=str, nargs='+', required=True, help='Source PDEBench raw H5 files')
    p.add_argument('--dst', type=str, required=True, help='Destination standard format H5')
    p.add_argument('--delete_src', action='store_true', help='Delete source files after merge')
    return p.parse_args()


def count_src_samples(src_paths: List[str]) -> List[int]:
    """Get sample count from each source file."""
    counts = []
    for path in src_paths:
        with h5py.File(path, 'r') as f:
            counts.append(f['velocity'].shape[0])
    return counts


def main():
    args = parse_args()

    # Read source info
    src_counts = count_src_samples(args.src)
    n_new_total = sum(src_counts)
    print(f"Sources ({len(args.src)} files, {n_new_total} new samples total):")
    for path, cnt in zip(args.src, src_counts):
        with h5py.File(path, 'r') as f:
            print(f"  {path}: N={cnt}, velocity={f['velocity'].shape}")

    # Read destination info
    with h5py.File(args.dst, 'r') as fd:
        n_old = fd['vector'].shape[0]
        T = fd['vector'].shape[1]
        H, W = fd['vector'].shape[2], fd['vector'].shape[3]
        scalar_indices = fd['scalar_indices'][:]
        print(f"\nDestination: {args.dst}")
        print(f"  vector: {fd['vector'].shape}, scalar: {fd['scalar'].shape}")
        print(f"  scalar_indices: {scalar_indices.tolist()}")
        print(f"  N_old={n_old}, T={T}, H={H}, W={W}")

    n_total = n_old + n_new_total
    print(f"\nMerge plan: {n_old} + {n_new_total} = {n_total} samples")

    # Create new file (tmp), then replace
    tmp_path = args.dst + '.tmp'
    print(f"\nCreating merged file: {tmp_path}")

    with h5py.File(tmp_path, 'w') as fout:
        vec_ds = fout.create_dataset(
            'vector', shape=(n_total, T, H, W, 3),
            dtype='float32', chunks=(1, T, H, W, 3),
        )
        scl_ds = fout.create_dataset(
            'scalar', shape=(n_total, T, H, W, 1),
            dtype='float32', chunks=(1, T, H, W, 1),
        )
        frc_ds = fout.create_dataset(
            'force', shape=(n_total, H, W, 2),
            dtype='float32', chunks=(1, H, W, 2),
        )
        fout.create_dataset('scalar_indices', data=scalar_indices)

        # Step 1: Copy old data
        print(f"\nStep 1: Copying {n_old} old samples...")
        with h5py.File(args.dst, 'r') as fd:
            for i in tqdm(range(n_old), desc="Old samples"):
                vec_ds[i] = fd['vector'][i]
                scl_ds[i] = fd['scalar'][i]
                frc_ds[i] = fd['force'][i]

        # Step 2: Convert and append new data from each source, delete after each
        write_idx = n_old
        for src_path, src_count in zip(args.src, src_counts):
            print(f"\nConverting {src_count} samples from {os.path.basename(src_path)}...")
            with h5py.File(src_path, 'r') as fs:
                for i in tqdm(range(src_count), desc=os.path.basename(src_path)):
                    # velocity (T,H,W,2) → vector (T,H,W,3): pad vz=0
                    vel = np.array(fs['velocity'][i], dtype=np.float32)
                    vec = np.zeros((T, H, W, 3), dtype=np.float32)
                    vec[..., :2] = vel
                    vec_ds[write_idx] = vec

                    # particles (T,H,W,1) → scalar (T,H,W,1)
                    scl_ds[write_idx] = np.array(fs['particles'][i], dtype=np.float32)

                    # force (H,W,2) → force (H,W,2)
                    frc_ds[write_idx] = np.array(fs['force'][i], dtype=np.float32)

                    write_idx += 1

            # Delete source immediately after copying
            if args.delete_src:
                os.remove(src_path)
                print(f"  Deleted: {src_path}")

    assert write_idx == n_total

    # Step 3: Replace old file
    print(f"\nStep 3: Replacing old file...")
    backup_path = args.dst + '.bak'
    shutil.move(args.dst, backup_path)
    shutil.move(tmp_path, args.dst)
    print(f"  Backup: {backup_path}")

    # Verify
    with h5py.File(args.dst, 'r') as fd:
        print(f"\nVerification:")
        print(f"  vector: {fd['vector'].shape}")
        print(f"  scalar: {fd['scalar'].shape}")
        print(f"  force: {fd['force'].shape}")
        print(f"  scalar_indices: {fd['scalar_indices'][:].tolist()}")
        assert fd['vector'].shape[0] == n_total
        print(f"  OK: {n_total} samples total")

    print(f"\n  Backup kept at: {backup_path}")
    print(f"  Run 'rm {backup_path}' to delete after verifying.")
    print("\nDone!")


if __name__ == '__main__':
    main()
