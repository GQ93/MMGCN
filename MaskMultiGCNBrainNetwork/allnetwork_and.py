import os
import numpy as np
import MaskGCNMultiBrain
from config import configMaskMultiGCNBrain
from Datapreprocess.Datapre import pnc_dataset, get_training_validation_test
import tensorflow as tf
from sklearn.metrics import r2_score
import time
from utility.utils import sparse_sim, cross_modality_sim, s2l


def read_data_sets():
    """
    read_data_set
    :return: data and label
    """
    PNC = pnc_dataset(fmri_log=[True, True, False], meta_log=[True, False])
    Data = PNC.return_roi2roi_network()
    data_emoid = Data['raw_fmri']['emoid']
    data_nback = Data['raw_fmri']['nback']
    label = Data['label']['WRAT']
    N = data_emoid.shape[0]
    N_roi = data_emoid.shape[1]
    data_train = dict()
    data_valid = dict()
    data_test = dict()
    data_train['emoid'] = data_emoid
    data_train['nback'] = data_nback
    label_train = label
    mm = np.mean(label_train)
    label_train -= mm
    print(len(label))
    L = dict()

    s_emoid = np.zeros((N, N_roi, N_roi))
    L_emoid = np.zeros((N, N_roi, N_roi))
    s_nback = np.zeros((N, N_roi, N_roi))
    L_nback = np.zeros((N, N_roi, N_roi))
    for i in range(N):
        s_1 = np.abs(np.corrcoef(data_emoid[i, :, :]))
        s_2 = np.abs(np.corrcoef(data_nback[i, :, :]))
        s_1[np.isnan(s_1)] = 0
        s_1 = sparse_sim(s_1, 20)
        L_1 = s2l(s_1, norm_state=1)
        s_2[np.isnan(s_2)] = 0
        s_2 = sparse_sim(s_2, 20)
        L_2 = s2l(s_2, norm_state=1)
        s_emoid[i, :, :] = s_1
        s_nback[i, :, :] = s_2
        L_emoid[i, :, :] = L_1
        L_nback[i, :, :] = L_2
    L['emoid_train'] = L_emoid
    L['nback_train'] = L_nback

    S_emoid = np.abs(np.corrcoef(np.reshape(s_emoid, (s_emoid.shape[0], -1))))
    S_nback = np.abs(np.corrcoef(np.reshape(s_nback, (s_nback.shape[0], -1))))
    S_emoid[np.isnan(S_emoid)] = 0
    S_nback[np.isnan(S_nback)] = 0
    S_cross_mod = cross_modality_sim(S_emoid, S_nback)
    S_emoid = s2l(S_emoid, norm_state=0)
    S_nback = s2l(S_nback, norm_state=0)
    S_cross_mod = s2l(S_cross_mod, norm_state=0)
    return data_train, label_train, L, S_emoid, S_nback, S_cross_mod

data_train, label_train, L, _, _, _ = read_data_sets()
np.save(r"F:\projects\MultiGCN\result\L_all", L)
