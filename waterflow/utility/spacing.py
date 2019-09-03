''' This package contains spatial discretization functions for one and two
dimensions.'''

import numpy as np
import matplotlib.pyplot as plt


def spacing(nx, Lx, ny=0, Ly=0, linear=True, loc=None, power=None, weight=None):
    axes_args = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)
    if linear:
        return axes_args[0], axes_args[1]
    else:
        # check dimensions
        loc = np.array(loc)
        if len(np.shape(loc)) == 1:
            xloc = loc[np.argsort(loc)]
            yloc = np.empty(0)
        if len(np.shape(loc)) == 2:
            xloc = np.unique(loc[:,0][np.argsort(loc[:,0])])
            yloc = np.unique(loc[:,1][np.argsort(loc[:,1])])
        # start populating the axes
        axes = np.array([np.repeat(0.0, nx), np.repeat(0.0, ny)])
        for iloc, loc in enumerate([xloc, yloc]):
            # dimension is not calculated if axis is empty
            if len(loc) == 0:
                break
            # select new axis
            ax_arg = axes_args[iloc]
            axis = axes[iloc]
            # per positions on the axis
            for pts_i in range(len(loc)):
                p = loc[pts_i]
                # add the center starting point
                axis[p] = axes_args[iloc][p]
                # continue with populating the other positions in range power
                for i in range(1, power + 1):
                    # to right
                    axis[p+i] = axis[p+i-1] + (ax_arg[p+i] - axis[p+i-1]) / weight
                    # to left
                    axis[p-i] = axis[p-i+1] - (axis[p-i+1] - ax_arg[p-i]) / weight
                # fill axis left of the point to zero with linear distance
                if pts_i == 0:
                    fill_left = np.linspace(0, axis[p-power], p - power + 1)
                    axis[:p-power+1] = fill_left
                # fill axis left to previous point with linear distance
                else:
                    fill_left = np.linspace(axis[loc[pts_i-1]+power],
                                            axis[p-power], 
                                            (p-power + 1) - (loc[pts_i-1]+power))
                    axis[loc[pts_i-1]+power:p-power+1] = fill_left
            # fill axis right of the point with linear distance
            fill_right = np.linspace(axis[p+power],
                                     ax_arg[-1],
                                     len(axis) - (p+power))
            axis[p+power:] = fill_right
            # reassign axis at which spacing is completed
            axes[iloc] = axis
    return axes[0], axes[1]


if __name__ == "__main__":
    Lx = 10
    Ly = 10
    nx = 22
    ny = 22
    
    x_spa1, y_spa1 = spacing(nx, Lx, ny, Ly, linear=False,
                           loc=[(6,13)], power = 2, weight = 2)
    
    x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=False,
                           loc=[(6,6), (13,13)], power = 2, weight = 2)
    
    x_spa2, y_spa2 = spacing(nx, Lx)
    
    x_spa3, y_spa3 = spacing(nx, Lx, ny, Ly, linear=False, 
                             loc=[(5,10), (5,5), (14,15)], power = 2, weight = 3)
    
    xlen = [x*0 for x in range(len(x_spa3))]
    ylen = [x*0 for x in range(len(x_spa3))]
    plt.scatter(x_spa3, xlen)
    plt.scatter(ylen, y_spa3)

    

