import os
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from waterflow import OUTPUT_DIR
from waterflow.utility.conductivityfunctions import soilselector


def quickplot(df, x, y, ax=None, ax_sec=None, xlabel=None, ylabel=None,
              title=None, legend=True, grid=True, save=False, 
              savepath=OUTPUT_DIR, filename=None, **kw):
    ax = ax or plt.gca()
    if ax_sec:
        for col in y:
            ax_sec.plot(df[x], df[col], label=df[col].name, **kw)
            ax_sec.set_xlabel(xlabel or df[x].name)
            ax_sec.set_ylabel(ylabel or ax_sec.get_ylabel())
    else:
        for col in y:
            ax.plot(df[x], df[col], label=df[col].name, **kw)
            ax.set_xlabel(xlabel or df[x].name)
            ax.set_ylabel(ylabel or ax.get_ylabel())

    if legend:
        han, lab = ax.get_legend_handles_labels()
        if ax_sec:
            han2, lab2 = ax_sec.get_legend_handles_labels()
        else:
            han2, lab2 = ([], [])

        ax.legend([*han, *han2], [*lab, *lab2])
    if grid:
        ax.grid()

    ax.set_title(title or ax.get_title())

    if save:
        plotdir = os.path.join(savepath, 'plots')
        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)

        plt.savefig(os.path.join(plotdir, filename))

    return ax, ax_sec


def solverplot(model, save=False, savepath=OUTPUT_DIR, filename=None):
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
    states = model.dft_states
    solver = model.dft_solved_times

    for k, v in states.items():
        ax1.plot(v.states, v.nodes, label=k)

    ax1.set_xlabel('heads (cm)')
    ax1.set_ylabel('distance (cm)')
    ax1.set_title('Hydraulic heads')
    ax1.grid(True)

    ax2.plot(solver['time'], solver['dt'], '.-', color='green')
    ax2.set_xlabel('time (d)')
    ax2.set_ylabel('dt (d)')
    ax2.grid(True)

    ax3.plot(solver['time'], solver['iter'], '.-', color='blue')
    ax3.set_xlabel('time (d)')
    ax3.set_ylabel('iterations (-)')
    ax3.grid(True)

    ax4.plot(solver['time'][1:], np.cumsum(solver['iter'][1:]), '.-',
             color='red')
    ax4.set_xlabel('time (d)')
    ax4.set_ylabel('cumulative dt (d)')
    ax4.grid(True)

    if save:
        plotdir = os.path.join(savepath, 'plots')
        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)

        plt.savefig(os.path.join(plotdir, filename))


def scale(xi, n, min_, max_, invert=False):
    # y = ax + b
    domain = (max_ - min_)
    a = n / domain
    b = -a * min_
    if invert:
        return (xi - b) / a
    return a * xi + b


def solutionspace(soils, indices, pltopts=None, lspace=None, rspace=None, show=True):

    tupsoils, dfs, (smin, smax) = soil = soilselector(soils)

    Nx, Ny, Nz = indices

    if not pltopts:
        pltopts = {'l': ['+', 'brown'], 's': ['o', 'yellow'],
                   'p': ['*', 'green'], 'c': ['^', 'black']}

    # scale conversions
    from_ascale = partial(scale, n=Nx, min_=smin.alpha, max_=smax.alpha)
    from_nscale = partial(scale, n=Ny, min_=smin.n, max_=smax.n)
    from_kscale = partial(scale, n=Nz, min_=smin.ksat, max_=smax.ksat)
    to_ascale = partial(scale, n=Nx, min_=smin.alpha, max_=smax.alpha, invert=True)
    to_nscale = partial(scale, n=Ny, min_=smin.n, max_=smax.n, invert=True)
    to_kscale = partial(scale, n=Nz, min_=smin.ksat, max_=smax.ksat, invert=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axes = [ax1, ax2]

    for ax in axes:
        for soil in tupsoils:
            a = from_ascale(soil.alpha)
            n = from_nscale(soil.n)
            k = from_kscale(soil.ksat)
            ax.scatter([a], [n], [k],
                       marker=pltopts[soil.category][0],
                       color=pltopts[soil.category][1])
            ax.text(a, n, k, soil.soiltype, fontsize=7)

        ax.set_xlabel('alpha')
        ax.set_ylabel('n')
        ax.set_zlabel('ksat')
        ax.set_xticks(list(range(0, Nx)))
        ax.set_yticks(list(range(0, Ny)))
        ax.set_zticks(list(range(0, Nz)))
        ax.set_xticklabels([round(to_ascale(i), 3) for i in range(0, Nx + 1)])
        ax.set_yticklabels([round(to_nscale(i), 3) for i in range(0, Ny + 1)])
        ax.set_zticklabels([round(to_kscale(i), 3) for i in range(0, Nz + 1)])
        ax.set_title(f'Solution space - ({soil.category})')

    c = ['green', 'red']
    for i, space in enumerate([lspace, rspace]):
        if isinstance(space, np.ndarray):
            axes[i].voxels(space, alpha=0.2, color=c[i], edgecolor='k')
        elif isinstance(space, str):
            axes[i].voxels(np.load(space, allow_pickle=True), alpha=0.2,
                           color=c[i], edgecolor='k')
    if show is True:
        plt.show()


if __name__ == '__main__':
    import pandas as pd

    # data
    data = {}
    columns = ['t', 'x', 'y', 'z', 'a', 'b', 'c']
    length = 10
    for i, c in enumerate(columns):
        if i == 0:
            data[c] = np.arange(length)
        elif i <= 3:
            data[c] = np.random.uniform(size=length)
        else:
            data[c] = np.random.randint(50, size=length)
    df = pd.DataFrame(data)

    quickplot(df, 't', 'x')

    fig, ax = plt.subplots()
    quickplot(df, 't', 'x', ax=ax, ylabel='yval')
    ax2 = ax.twinx()
    quickplot(df, 't', 'a', ax=ax, ax_sec=ax2, ylabel='y2val', color='red', grid=False)

    fig, [[ax, ax2], [ax3, ax4]] = plt.subplots(2, 2)
    quickplot(df, 't', ['x', 'y', 'z'], ax=ax, xlabel='time', ylabel='y-vals', legend=False)
    quickplot(df, 't', ['a', 'b', 'c'], ax=ax2, ylabel='y-vals')
    quickplot(df, 'x', ['a', 'b', 'c'], ax=ax3, xlabel='distance')
    quickplot(df, 'x', ['y', 'z'], ax=ax4, ylabel='y-z vals')
    fig.suptitle('My Plot')
    fig.show()

    fig, [ax, ax2] = plt.subplots(2)
    ax_sec = ax.twinx()
    quickplot(df, 't', ['x'], ax=ax, ylabel='y-vals', color='red')
    quickplot(df, 't', ['y'], ax=ax, ax_sec=ax_sec, ylabel='y2-vals')
    quickplot(df, 't', ['z'], ax=ax, ax_sec=ax_sec)
    quickplot(df, 't', ['a'], ax=ax2)
    plt.suptitle('with secondary axis')
    plt.show()

    solutionspace([1, 2, 3], (3, 3, 3))
