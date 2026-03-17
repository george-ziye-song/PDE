import h5py
import numpy as np

for fname in ['2D_DarcyFlow_beta1.0_Train.hdf5', '2D_DarcyFlow_beta10.0_Train.hdf5']:
    path = f'/home/msai/song0304/code/PDE/data/finetune/{fname}'
    print()
    print('=' * 60)
    print('File:', fname)
    print('=' * 60)
    with h5py.File(path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print('  %s: shape=%s, dtype=%s' % (name, obj.shape, obj.dtype))
                if obj.ndim >= 1 and obj.shape[0] > 0:
                    data = obj[0]
                    print('    sample[0]: min=%.6e, max=%.6e' % (np.min(data), np.max(data)))
            elif isinstance(obj, h5py.Group):
                print('  %s/ (group)' % name)
        f.visititems(print_structure)
        print()
        print('  Top-level keys:', list(f.keys()))
        print('  Attrs:', dict(f.attrs))
