import math
import numpy as np
import scipy.sparse as sp
import dgl


def weight_wavelet(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e,-lamb[i]*s)
        # lamb[i] = 2-lamb[i]
    # print('wavelet: ', lamb)
    Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))
    return Weight


def weight_wavelet_inverse(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e,lamb[i]*s)
        # lamb[i] = 1/ (lamb[i]+1)
        # lamb[i] = 2-lamb[i]
    # print('inv_wavelet: ', lamb)
    Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))
    return Weight


def normal_GWNN_basis():
    npzfile = np.load('group_U2.npz')
    lamb = npzfile['arr_0']
    U = npzfile['arr_1']
    s = 1.0
    print('---', lamb)
    Weight = weight_wavelet(s, lamb.copy(), U)
    print('---', lamb)
    inverse_Weight = weight_wavelet_inverse(s, lamb.copy(), U)
    print('---', lamb)

    threshold = 1e-4
    Weight[Weight < threshold] = 0.0
    inverse_Weight[inverse_Weight < threshold] = 0.0
    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)

    gwn_graph = dgl.from_scipy(Weight, eweight_name='w')
    gwn_invgraph = dgl.from_scipy(inverse_Weight, eweight_name='w')
    # print(np.dot(Weight, inverse_Weight))
    dgl.data.utils.save_graphs('gwn1', [gwn_graph, gwn_invgraph])


def our_GWNN_basis(dataset='amazon'):
    npzfile = np.load('{}/lamb_U.npz'.format(dataset))
    lamb = npzfile['arr_0']
    U = npzfile['arr_1']
    lamb1 = lamb.copy()
    lamb2 = lamb.copy()
    for i in range(len(lamb)):
        # lamb1[i] = 2-lamb[i]
        lamb1[i] = math.pow(math.e, -lamb[i])
    print('---', lamb1)
    low_weight = np.dot(np.dot(U, np.diag(lamb1)), np.transpose(U))
    for i in range(len(lamb)):
        lamb2[i] = math.pow(math.e, lamb[i])
    print('---', lamb2)
    high_weight = np.dot(np.dot(U, np.diag(lamb2)), np.transpose(U))

    threshold = 1e-5
    low_weight[np.abs(low_weight) < threshold] = 0.0
    high_weight[np.abs(high_weight) < threshold] = 0.0
    low_weight = sp.csr_matrix(low_weight)
    high_weight = sp.csr_matrix(high_weight)
    print(low_weight, high_weight)

    low_weight = dgl.from_scipy(low_weight, eweight_name='w')
    high_weight = dgl.from_scipy(high_weight, eweight_name='w')
    dgl.data.utils.save_graphs('{}/our_gwn1'.format(dataset), [low_weight, high_weight])


def our_GWNN_basis2(dataset='amazon'):
    npzfile = np.load('{}/lamb_U.npz'.format(dataset))
    lamb = npzfile['arr_0']
    U = npzfile['arr_1']
    lamb1 = lamb.copy()
    lamb2 = lamb.copy()
    for i in range(len(lamb)):
        # lamb1[i] = 2-lamb[i]
        lamb1[i] = math.pow(math.e, -lamb[i])
    print('---', lamb1)
    low_weight = np.dot(np.dot(U, np.diag(lamb1)), np.transpose(U))
    for i in range(len(lamb)):
        lamb2[i] = math.pow(math.e, lamb[i])
    print('---', lamb2)
    high_weight = np.dot(np.dot(U, np.diag(lamb2)), np.transpose(U))

    threshold = 1e-5
    low_weight[np.abs(low_weight) < threshold] = 0.0
    high_weight[np.abs(high_weight) < threshold] = 0.0
    low_weight = sp.csr_matrix(low_weight)
    high_weight = sp.csr_matrix(high_weight)
    print(low_weight, high_weight)

    low_weight = dgl.from_scipy(low_weight, eweight_name='w')
    high_weight = dgl.from_scipy(high_weight, eweight_name='w')
    dgl.data.utils.save_graphs('{}/our_gwn1'.format(dataset), [low_weight, high_weight])


# normal_GWNN_basis()
our_GWNN_basis('amazon')
