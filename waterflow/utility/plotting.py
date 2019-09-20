import matplotlib.pyplot as plt
import numpy as np


def quickplot(data, x, y, title='', xunit='', yunit='', y2=None, y2unit='', grid=True, ls='solid'):
    if isinstance(data, dict):
        fig, ax = plt.subplots()
        for k, v in data.items():
            ax.plot(v[x], v[y], label=k, ls=ls)
            ax.set_xlabel(f'{x} [{xunit}]')
            ax.set_ylabel(f'{y} [{yunit}]')
            ax.grid(grid)
            ax.set_title(title)
            ax.legend()
    else:
        ax = data.plot(x, y, ls=ls)
        if y2:
            ax2 = data.plot(x, y2, ls=ls, secondary_y=True, ax=ax, label=f'{y} [{y2unit}]')
            ax2.set_ylabel(f'{y2} [{y2unit}]')
        ax.set_xlabel(f'{x} [{xunit}]')
        ax.set_ylabel(f'{y} [{yunit}]')
        ax.set_title(title)
        ax.grid(grid)
        ax.legend()


def solverplot(model):
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
