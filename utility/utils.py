import numpy as np
import tensorflow as tf
from functools import reduce
from operator import mul

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def sparse_sim(s, k):
    """
    sparse the similarity matrix using K-nearest neighbor
    Input: s is N*N matrix, k is the K nearest neighbor
    """
    N, N_ = s.shape
    if N != N_:
        RuntimeError('similarity matrix should be N by N')
    elif np.allclose(s, s.T, atol=1e-8) and k <= N:
        mask = np.zeros(s.shape)
        for i in range(s.shape[0]):
            indx = np.argpartition(s[i, :], -k)[-k:]
            mask[i, indx] = 1
        sparse_mat = np.multiply(s, mask)
    else:
        print('please chech the input of sparse_sim')
    return sparse_mat


def get_lap(S, normalize=False):
    """

    :param S: similarity
    :return: Laplacian L , Degree D
    """
    D_diag = np.sum(S, axis=0)
    L = np.diag(D_diag) - S

    if not normalize:

        return L
    else:
        D_sqrt_diag = np.power(D_diag, -1 / 2)
        L_sym = np.dot(np.dot(np.diag(D_sqrt_diag), L), np.diag(D_sqrt_diag))

        return L_sym


def dimension_reduce(feat):
    """
    feature dimension reduction
    :param feat: train data for PCA
    :return: dimension reduced feature
    """
    from sklearn.decomposition import PCA
    model = PCA(n_components=0.99)
    model.fit(feat)
    return model.transform(feat)


def s2l(s, norm_state=1):
    """
    from similarity to laplacian
    :param s: similarity matrix N by N
    :return: Laplcacian
    """
    D = np.diag(np.sum(s, axis=1))
    if norm_state == 0:
        L = D -s
    elif norm_state == 1:
        adj_tilde = s + 1e-5*np.identity(n=s.shape[0])
        d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
        d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1 / 2)
        d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
        adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
        L = adj_norm
    elif norm_state == 2:
        adj_tilde = s + np.identity(n=s.shape[0])
        d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
        d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1 / 2)
        d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
        adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
        L = adj_norm + np.identity(n=adj_norm.shape[0])
    return L

def cross_modality_sim(s1, s2, method=1):
    """
    define the similarity between different modality
    :param s1: N by N matrix
    :param s2: N by N matrix
    :param method: 1
    :return: the between class similarity 2N by 2N
    """
    if method == 1:
        s11 = s1 @ s1
        s12 = s1 @ s2
        s21 = s2 @ s1
        s22 = s2 @ s2
    else:
        s11 = 0
        s12 = 0
        s21 = 0
        s22 = 0
    s = np.concatenate((np.concatenate((s11, s12), axis=1), np.concatenate((s21, s22), axis=1)), axis=0)
    return s

if __name__ == '__main__':
    S = np.random.rand(3**2).reshape(3, 3)
    S = np.triu(S)
    S += S.T - np.diag(S.diagonal())
    S = sparse_sim(S, k=2)
    L = get_lap(S, normalize=True)
    print(S)
    print(L)
