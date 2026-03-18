import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import scipy.sparse as sp


def normalization_adj(adjacency):
    """calculate L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        The normalized adjacency matrix, the type is torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])  # add self-join
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()

    # transform to torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def _prepare_features(features, method='heat'):
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)
    if method == 'ncos':
        features = normalize(features, axis=1, norm='l1')
    return features


def get_graph(features, topk=10, method='heat'):
    """Generate a sparse top-k graph without materializing an NxN similarity matrix."""
    features = _prepare_features(features, method)
    metric = 'cosine' if method == 'cos' else 'euclidean'
    n_neighbors = min(topk + 1, features.shape[0])
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn_model.fit(features)
    indices = nn_model.kneighbors(features, return_distance=False)

    row_idx = []
    col_idx = []
    for row, neighbors in enumerate(indices):
        for col in neighbors:
            if col != row:
                row_idx.append(row)
                col_idx.append(col)

    data = np.ones(len(row_idx), dtype=np.float32)
    graph = sp.coo_matrix((data, (row_idx, col_idx)), shape=(features.shape[0], features.shape[0]))
    return graph


# single
def get_adjacency(features, n, topk=10, self_join=True, method='heat'):
    """Get the standardized adjacency matrix, sparse and dense"""
    adj = get_graph(features, topk, method)
    adj = sp.coo_matrix(
        (np.ones_like(adj.data, dtype=np.float32), (adj.row, adj.col)),
        shape=(n, n),
        dtype=np.float32,
    )
    raw_adj = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if self_join:
        adj = adj + sp.eye(adj.shape[0])  # add self-join
    # raw_adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, raw_adj


def get_edges(dist, topk=10):
    """Through the similarity matrix, the graph structure is established"""
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edges_unordered = []
    for i, ks_i in enumerate(inds):
        for k_i in ks_i:
            if k_i != i:
                edges_unordered.append([i, k_i])
    return edges_unordered
