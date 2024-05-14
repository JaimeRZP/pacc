import numpy as np
import pickle
import pandas as pd
import os
import sacc
import yaml
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 18

def plot_cls(cl_ensembles, wanted_pairs, fmts=['ro', 'bo', 'ko']):
    t_i = np.transpose(wanted_pairs)[0]
    t_j = np.transpose(wanted_pairs)[1]
    unique_t_i = np.unique(t_i)
    unique_t_j = np.unique(t_j)
    l_t_i = len(unique_t_i)
    l_t_j = len(unique_t_j)
    npair = 0

    if (t_j == t_i).all():
        figure, axis = plt.subplots(1, l_t_j, figsize=(5*l_t_i, 5*1))
        for i in range(0, l_t_i):
            proposed_pair = [unique_t_i[i], unique_t_j[i]]
            for k, ensemble in enumerate(cl_ensembles):
                pos = ensemble.pairs.index(proposed_pair)
                ls = ensemble.ls[pos]
                data = ensemble.data[pos]
                err = ensemble.errs[pos]
                axis[i].errorbar(ls, data, yerr=err, fmt=fmts[k], label='Data')
            axis[i].set_title("{}_{}".format(proposed_pair[0],proposed_pair[1]))
            axis[i].set_xscale("log")
            axis[i].set_yscale("log")
            npair += 1
        plt.legend()
        plt.show()
    else:
        figure, axis = plt.subplots(l_t_i, l_t_j, figsize=(5*l_t_i, 5*l_t_j))
        for i in range(0, l_t_i):
            for j in range(0, l_t_j):
                proposed_pair = [unique_t_i[i], unique_t_j[j]]
                if proposed_pair in wanted_pairs:
                    for k, ensemble in enumerate(cl_ensembles):
                        pos = ensemble.pairs.index(proposed_pair)
                        ls = ensemble.ls[pos]
                        data = ensemble.data[pos]
                        err = ensemble.errs[pos]
                        axis[i, j].errorbar(ls, data, yerr=err, fmt=fmts[k], label='Data')
                    axis[i, j].set_title("{}_{}".format(proposed_pair[0],proposed_pair[1]))
                    axis[i, j].set_xscale("log")
                    axis[i, j].set_yscale("log")
                    npair += 1
                else:
                    axis[i, j].axis('off')
        plt.legend()
        plt.show()