import numpy as np
import sacc
import yaml

class ClsEnsemble(object):
    def __init__(self, s, y, label=None):

        # Apply scale cuts
        self.order = y["order"]
        self.s = self._cut(s)
        self.label = label
        self._get_defaults()

    def _cut(self, s):
        indices = np.array([])
        for cl in self.order:
            t1, t2 = cl["tracers"]
            cls = cl["cls"]
            for cl_name in cls:
                if "ell_cuts" in list(cl.keys()):
                    lmin, lmax = cl["ell_cuts"]
                    ind = s.indices(cl_name, (t1, t2),
                                ell__gt=lmin, ell__lt=lmax)
                else:
                    ind = s.indices(cl_name, (t1, t2))
                indices = np.append(indices, ind)
        if len(indices) != 0:
            indices = indices.astype(int)
            s.keep_indices(indices)
        return s

    def _get_defaults(self):
        self.indices = []
        self.data = []
        self.ls = []
        self.pairs = []
        for cl in self.order:
            t1, t2 = cl["tracers"]
            cls = cl["cls"]
            for cl_name in cls:
                l, c_ell, ind = self.s.get_ell_cl(
                    cl_name, t1, t2,
                    return_cov=False,
                    return_ind=True)
                self.indices.append(ind)
                self.data.append(c_ell)
                self.ls.append(l)
                self.pairs.append([t1, t2])

        self.cov = self.s.covariance.dense
        errs = np.sqrt(np.diag(self.cov))
        self.errs = [errs[ind] for ind in self.indices]

class ClSubSemble(ClsEnsemble):
    def __init__(self, cl_ensemble, pairs):
        poss = [cl_ensemble.pairs.index(pair) for pair in pairs]
        self.order  = [cl_ensemble.order[pos] for pos in poss]
        self.s = self._cut(cl_ensemble.s)
        self._get_defaults()

class ClTheoryEnsemble(ClsEnsemble):
    def __init__(self, cl_ensemble, theory, errs=None, label=None):
        self.order = cl_ensemble.order
        self.s = cl_ensemble.s
        self._get_defaults()
        self.label = label
        self.data = [theory[ind] for ind in self.indices]
        if errs is not None:
            self.errs = errs
        else:
            self.errs = [np.zeros_like(ind) for ind in self.indices]
