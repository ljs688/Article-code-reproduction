import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class MultiViewDataset(Dataset):
    """通用多视图数据集加载器"""

    def __init__(self, path, dataset_name):
        mat_files = {
            'Caltech101': '2view-caltech101-8677sample.mat',
            'MSRCV1': 'MSRCV1.mat',
            'YouTubeFace': 'YouTubeFace.mat',
            'Hdigit': 'Hdigit.mat',
        }
        filename = mat_files.get(dataset_name, dataset_name + '.mat')
        data = scipy.io.loadmat(path + filename)

        # --- 解析视图数据 ---
        self.views = []
        X_key = self._find_key(data, ['X', 'data', 'x', 'fea'])
        if X_key is not None:
            X = data[X_key]
            if X.dtype == object:
                # 遍历所有元素提取视图
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        elem = X[i, j]
                        if isinstance(elem, np.ndarray):
                            if elem.dtype == object:
                                # 嵌套 cell array
                                for k in range(elem.flatten().shape[0]):
                                    v = elem.flatten()[k].astype(np.float32)
                                    if v.ndim == 1:
                                        v = v.reshape(-1, 1)
                                    self.views.append(v)
                            else:
                                v = elem.astype(np.float32)
                                if v.ndim == 1:
                                    v = v.reshape(-1, 1)
                                self.views.append(v)
            else:
                v = X.astype(np.float32)
                self.views.append(v)
        if not self.views:
            raise ValueError(f"无法解析视图数据, keys: {[k for k in data if not k.startswith('__')]}")

        # 统一方向: 确保 (n_samples, n_features)
        # 用标签数量来判断样本数
        Y_key = self._find_key(data, ['Y', 'gt', 'y', 'label', 'labels', 'truelabel', 'truth'])
        Y = data[Y_key]
        if Y.dtype == object:
            Y = Y.flatten()[0]
        Y = Y.astype(np.int32).flatten()
        n_samples = Y.shape[0]

        for i in range(len(self.views)):
            v = self.views[i]
            if v.shape[0] == n_samples:
                pass  # 已经正确
            elif v.shape[1] == n_samples:
                v = v.T
            else:
                raise ValueError(f"视图{i} shape {v.shape} 与样本数 {n_samples} 不匹配")
            self.views[i] = v

        self.n_samples = n_samples
        self.n_views = len(self.views)
        self.dims = [v.shape[1] for v in self.views]

        # 标签从0开始
        if Y.min() >= 1:
            Y = Y - Y.min()
        self.Y = Y
        self.class_num = len(np.unique(self.Y))

        print(f"[{dataset_name}] views={self.n_views}, dims={self.dims}, "
              f"samples={self.n_samples}, classes={self.class_num}")

    @staticmethod
    def _find_key(data, candidates):
        keys = [k for k in data.keys() if not k.startswith('__')]
        for c in candidates:
            for k in keys:
                if k.lower() == c.lower():
                    return k
        return None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        xs = [torch.from_numpy(self.views[v][idx]) for v in range(self.n_views)]
        return xs, self.Y[idx], torch.tensor(idx, dtype=torch.long)


def load_data(dataset):
    ds = MultiViewDataset('./data/', dataset)
    return ds, ds.dims, ds.n_views, ds.n_samples, ds.class_num
