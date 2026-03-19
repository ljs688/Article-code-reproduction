import scipy.io as sio
mat = sio.loadmat('./datasets/YoutubeFace.mat')
for k, v in mat.items():
    if not k.startswith('_'):
        if hasattr(v, 'shape') and v.dtype == object:
            print(f"{k}: shape={v.shape}")
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    print(f"  [{i}][{j}]: shape={v[i][j].shape}")
        else:
            print(f"{k}: shape={getattr(v, 'shape', 'N/A')}")
