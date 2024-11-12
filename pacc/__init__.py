import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .cls_ensemble import ClsEnsemble
from .cls_ensemble import ClSubSemble
from .cls_ensemble import ClTheoryEnsemble

def _make_default_config(config):
    keys = config.keys()
    if "Xi2s" not in keys:
        config["Xi2s"] = None
    if "alpha" not in keys:
        config["alpha"] = 1.0
    if "colors" not in keys:
        config["colors"] = None
    if "show_legend" not in keys:
        config["show_legend"] = True
    if "show_colobar" not in keys:
        config["show_colobar"] = False
    return config

def plot_cls(cl_supersembles, wanted_pairs, configs=None):
    if configs is None:
        configs = np.array([{} for _ in range(len(cl_supersembles))])
    first_ensemble = True
    for (cl_ensembles, config) in zip(cl_supersembles, configs):
        config = _make_default_config(config)
        Xi2s = config["Xi2s"]
        alpha = config["alpha"]
        colors = config["colors"]
        show_legend = config["show_legend"]
        show_colobar = config["show_colobar"]
        n_ensembles = len(cl_ensembles)

        if Xi2s is None:
            Xi2s = np.linspace(0, 1, n_ensembles)
        else:
            Xi2s = np.array(Xi2s)

        if colors is None:
            colormap = cm.winter
            norm = mpl.colors.Normalize(vmin=Xi2s.min(), vmax=Xi2s.max())
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
            cmap.set_array([])
            colors = colormap(np.linspace(0, 1, n_ensembles))

        t_i = np.transpose(wanted_pairs)[0]
        t_j = np.transpose(wanted_pairs)[1]
        unique_t_i = np.unique(t_i)
        unique_t_j = np.unique(t_j)
        l_t_i = len(unique_t_i)
        l_t_j = len(unique_t_j)
        npair = 0

        if (t_j == t_i).all():
            if first_ensemble:
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
                axis[i].set_xlim([0.9*ls.min(), 1.1*ls.max()])
                npair += 1
            if show_colobar:
                figure.subplots_adjust(right=0.8)
                cbar_ax = figure.add_axes([0.82, 0.15, 0.01, 0.7])
                figure.colorbar(cmap, cax=cbar_ax, label=r'$\chi^2$')
        else:
            if first_ensemble:
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
                        axis[i, j].set_xlim([0.9*ls.min(), 1.1*ls.max()])
                        npair += 1
                        if i == l_t_i-1:
                            axis[i, j].set_xlabel(r"$\ell$")
                        if j == 0:
                            axis[i, j].set_ylabel(r"$C_\ell$")
                    else:
                        axis[i, j].axis('off')
            if show_colobar:
                figure.subplots_adjust(right=0.8)
                cbar_ax = figure.add_axes([0.82, 0.1, 0.01, l_t_j/35])
                figure.colorbar(cmap, cax=cbar_ax, label=r'$\chi^2$')
        first_ensemble = False
        if show_legend:
            plt.legend()
        first_ensemble = False
    plt.show()
    return figure

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
