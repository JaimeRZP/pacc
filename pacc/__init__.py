import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .cls_ensemble import ClsEnsemble
from .cls_ensemble import ClSubSemble
from .cls_ensemble import ClTheoryEnsemble

def plot_cls(cl_ensembles, wanted_pairs,
             Xi2s=None, alpha=1.0, show_legend=True):
    n_ensembles = len(cl_ensembles)

    if Xi2s is None:
        Xi2s = np.linspace(0, 1, n_ensembles)
    else:
        Xi2s = (Xi2s - np.min(Xi2s)) / (np.max(Xi2s) - np.min(Xi2s))

    colormap = cm.winter
    colors = colormap(np.linspace(0, 1, n_ensembles))

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
                if ensemble.label is None:
                            label = 'Data_{}'.format(k)
                else:
                    label = ensemble.label
                axis[i].errorbar(ls, data, yerr=err,
                                 color=colors[k], fmt="o-",
                                 alpha=alpha,
                                 label=label)
            axis[i].set_title("{}_{}".format(proposed_pair[0], proposed_pair[1]))
            axis[i].set_xscale("log")
            axis[i].set_yscale("log")
            npair += 1
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
                        if ensemble.label is None:
                            label = 'Data_{}'.format(k)
                        else:
                            label = ensemble.label
                        axis[i, j].errorbar(ls, data, yerr=err,
                                            color=colors[k], fmt="o-",
                                            alpha=alpha,
                                            label=label)
                    axis[i, j].set_title("{}_{}".format(proposed_pair[0],proposed_pair[1]))
                    axis[i, j].set_xscale("log")
                    axis[i, j].set_yscale("log")
                    npair += 1
                else:
                    axis[i, j].axis('off')
    if show_legend:
        plt.legend()
    plt.show()

def plot_cov(cle, wanted_pairs, logscale=True):
    clse = ClSubSemble(cle, wanted_pairs)
    edges = [id[0] for id in clse.indices]
    labels = [pair[0]+' x '+pair[1] for pair in wanted_pairs]
    cov = clse.cov
    title='Covariance matrix'
    if logscale:
        title = 'Log10[Abs(covariance matrix)]'
        cov = np.log10(abs(cov))
    plt.imshow(cov)
    plt.xticks(edges, labels, rotation=90)
    plt.yticks(edges, labels)
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_corr(cle, wanted_pairs, logscale=True):
    clse = ClSubSemble(cle, wanted_pairs)
    edges = [id[0] for id in clse.indices]
    labels = [pair[0]+' x '+pair[1] for pair in wanted_pairs]
    cov = clse.cov
    sig = np.sqrt(np.diag(cov))
    corr = clse.cov / np.outer(sig, sig)
    title='correlation matrix'
    if logscale:
        title = 'Log10[Abs(correlation matrix)]'
        corr = np.log10(abs(corr))
    plt.imshow(corr)
    plt.xticks(edges, labels, rotation=90)
    plt.yticks(edges, labels)
    plt.title(title)
    plt.colorbar()
    plt.show()