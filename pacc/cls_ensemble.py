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

class ClsEnsemble(object):
    def __init__(self, s, y):

        # Apply scale cuts
        self._apply_scale_cuts(s, y)

        self.indices = np.array([])
        self.data = []
        self.ls = []
        self.pairs = []
        for cl in y["order"]:
            t1, t2 = cl["tracers"]
            cls = cl["cls"]
            for cl_name in cls:
                l, c_ell, ind = s.get_ell_cl(
                    cl_name, t1, t2,
                    return_cov=False,
                    return_ind=True)
                self.indices = np.append(self.indices, ind)
                self.data.append(c_ell)
                self.ls.append(l)
                self.pairs.append([t1, t2])
        
        self.indices = self.indices.astype(int)
        self.cov = s.covariance.dense
        self.cov = np.transpose(np.transpose(self.cov[self.indices])[self.indices])
        lengths = [len(l) for l in self.ls]
        self.edges  = np.append(np.array([0]), np.cumsum(lengths))
        errs = np.sqrt(np.diag(self.cov))
        self.errs = self._segment(errs)

    def _apply_scale_cuts(self, s, y):
        indices = np.array([])
        for cl in y["order"]:
            t1, t2 = cl["tracers"]
            cls = cl["cls"]
            if "ell_cuts" in list(cl.keys()):
                lmin, lmax = cl["ell_cuts"]
                for cl_name in cls:
                    ind = s.indices(cl_name, (t1, t2),
                                    ell__gt=lmin, ell__lt=lmax)
                    indices = np.append(indices, ind)
        if len(indices) != 0:
            s.keep_indices(indices)

    def _segment(self, A):
        seg_A = []
        for i in range(0, len(self.edges)-1):
            seg_A.append(A[self.edges[i]:self.edges[i+1]])
        return seg_A