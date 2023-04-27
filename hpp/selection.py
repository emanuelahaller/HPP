"""
Features selection
"""
from __future__ import division
import sys
import math
import numpy as np


def linear_l1(selection, epsilon):
    """
    Aux function for clusterIpfp
    """
    selection = np.squeeze(selection)
    n_features = np.prod(selection.shape)
    ind = np.argsort(selection)
    n_eps = math.floor(1 / epsilon)
    sel_ind = ind[n_features - n_eps:n_features]
    new_selection = np.zeros((n_features, 1), dtype=np.float32)
    new_selection[sel_ind] = epsilon

    return new_selection


def cluster_ipfp(covariance_matrix, k_value):
    """
    Select features
    """
    n_features = covariance_matrix.shape[0]
    epsilon = 1 / np.float32(k_value)
    max_steps = 50

    new_sol = np.ones((n_features, 1), dtype=np.float32)
    new_sol = new_sol / n_features
    new_sol = new_sol / np.sum(new_sol)

    best_score = sys.float_info.min
    n_steps = 0
    sol = new_sol

    while n_steps < max_steps:

        n_steps = n_steps + 1

        score = np.dot(np.dot(new_sol.T, covariance_matrix), new_sol)
        if score > best_score:
            best_score = score
            sol = new_sol

        old_sol = np.copy(new_sol)

        init_sol = np.dot(covariance_matrix, old_sol)

        final_sol = linear_l1(init_sol, epsilon)
        disp_sol = final_sol - old_sol
        prop_k = np.dot(np.dot(disp_sol.T, covariance_matrix), disp_sol)

        if prop_k >= 0:
            new_sol = final_sol
        else:
            aux_c = np.dot(init_sol.T, disp_sol)
            aux_t = min(1, -aux_c / prop_k)
            new_sol = old_sol + aux_t * disp_sol

    return sol
