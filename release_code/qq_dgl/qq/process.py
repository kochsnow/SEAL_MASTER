
"""
Loads input data from gcn/data directory

ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
    (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
    object;
ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
attention: This is a txt file with each line an index number

All objects above must be saved using python pickle module.

:param dataset_str: Dataset name
:return: All data input files loaded (as well the training/test data).
"""
import pickle
import numpy as np
from scipy.sparse import csr_matrix


def make_dataset1():
    with open('group_feature_withoutzero', 'rb') as f:
        group_feature = pickle.load(f)

    group_all = np.zeros([len(group_feature), 10])
    for idx, feat in enumerate(group_feature):
        group_all[idx] = feat.mean(0)

    group_allx = group_all[:-10000]
    group_tx = group_all[-10000:]
    group_allx = csr_matrix(group_allx)
    group_tx = csr_matrix(group_tx)
    test_index = list(range(len(group_all)))
    test_index = test_index[-10000:]

    # x and allx are same in qq staset since all train instances has their labels
    with open('ind.group.allx', 'wb') as f:
        pickle.dump(group_allx, f)
    with open('ind.group.x', 'wb') as f:
        pickle.dump(group_allx, f)
    with open('ind.group.tx', 'wb') as f:
        pickle.dump(group_tx, f)
    with open('ind.group.test.index', 'w') as f:
        for idx in test_index:
            f.write(str(idx)+'\n')


# 训练集保证采样均衡
# feature normalize
def make_dataset2():
    # with open('ind.gr', 'rb') as f:
    #     group_feature = pickle.load(f)
    x_embed = np.load('norm_x.npy')

    group_allx = x_embed[:-10000]
    group_tx = x_embed[-10000:]

    # group_allx = csr_matrix(group_allx)
    # group_tx = csr_matrix(group_tx)
    test_index = list(range(len(group_all)))
    test_index = test_index[-10000:]

    # x and allx are same in qq staset since all train instances has their labels
    with open('ind.group.allx', 'wb') as f:
        pickle.dump(group_all, f)
    with open('ind.group.x', 'wb') as f:
        pickle.dump(group_allx, f)
    with open('ind.group.tx', 'wb') as f:
        pickle.dump(group_tx, f)
    with open('ind.group.test.index', 'w') as f:
        for idx in test_index:
            f.write(str(idx)+'\n')



# def make_dataset_small():


make_dataset2()