import pickle
import numpy as np
import scipy
import networkx as nx
from dataset import Dataset


def sort(lamb, U):
    idx = lamb.argsort()
    return lamb[idx], U[:, idx]

def laplacian(W, normalized=True):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        # d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def create_basis():
    g = Dataset('amazon').graph
    adj_mat = g.adjacency_matrix(scipy_fmt='coo')
    print(adj_mat)
    # with open('data/ind.group.graph', 'rb') as f:
    #     graph = pickle.load(f)
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(mat))
    L = laplacian(adj_mat, True)
    print(L)
    lamb, U = scipy.linalg.eigh(L.toarray(), driver='evr')
    lamb, U = sort(lamb, U)
    np.savez("amazon/lamb_U.npz", lamb, U)


def study_L():
    g = Dataset('yelp').graph
    adj_mat = g.adjacency_matrix(scipy_fmt='coo')
    print((adj_mat*adj_mat*adj_mat).nnz)

# def fuck(x):
#     x[1] = 2
#
# x = np.zeros(5)
# print(x)
# fuck(x.copy())
# print(x)

study_L()


# npzfile = np.load('amazon/lamb_U.npz')
# lamb = npzfile['arr_0']
# U = npzfile['arr_1']
#
# print(lamb)
# print(lamb[:1000])
# print(lamb[-1000:])
# print(U[:1000])
# print(U[-1000:])
# print(U[0])
# npzfile = np.load('group_U2.npz')
# print(npzfile.files)


# with open("data/ind.cora.graph", 'rb') as f:
#     cora=pickle.load(f)
#
# print(cora[633])