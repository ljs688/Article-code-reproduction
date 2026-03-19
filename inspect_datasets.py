# inspect_datasets.py
import scipy.io
import numpy as np
import os

def inspect_mat(path):
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")
    if not os.path.exists(path):
        print("  FILE NOT FOUND")
        return
    data = scipy.io.loadmat(path)
    for key, val in data.items():
        if key.startswith('__'):
            continue
        print(f"\n  Key: '{key}', Type: {type(val).__name__}, ", end="")
        if isinstance(val, np.ndarray):
            print(f"Shape: {val.shape}, Dtype: {val.dtype}")
            # If it's an object array (cell array in MATLAB), dig deeper
            if val.dtype == object:
                for i in range(min(val.shape[0], 3)):
                    for j in range(min(val.shape[1] if val.ndim > 1 else 1, 10)):
                        elem = val[i][j] if val.ndim > 1 else val[i]
                        if isinstance(elem, np.ndarray):
                            print(f"    [{i},{j}] -> ndarray, Shape: {elem.shape}, Dtype: {elem.dtype}")
                        else:
                            print(f"    [{i},{j}] -> {type(elem).__name__}: {elem}")
            elif val.size <= 20:
                print(f"    Values: {val.flatten()}")
            else:
                flat = val.flatten()
                print(f"    Min: {flat.min()}, Max: {flat.max()}, Unique count: {len(np.unique(flat))}")
        else:
            print(f"Value: {val}")

files = [
    './data/2view-caltech101-8677sample.mat',
    './data/MSRCV1.mat',
    './data/YouTubeFace.mat',
]

for f in files:
    inspect_mat(f)
