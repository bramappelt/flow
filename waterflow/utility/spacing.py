''' This package contains functions for the generation of nodal
distributions '''

import numpy as np


def spacing(nx, Lx, ny=0, Ly=0, linear=True, loc=None, power=None, weight=None):
    """ One and two dimensional nodal spacing function.

    Return two arrays that contain a discretization which may be focussed on
    selected locations.

    Parameters
    ----------
    nx : :obj:`int`
        Number of nodes in the x-direction.
    Lx : :obj:`int` of :obj:`float`
        Total length in the x-direction.
    ny : :obj:`int`
        Number of nodes in the y-direction.
    Ly: :obj:`int` of :obj:`float`
        Total length in the y-direction.
    linear : :obj:`bool`, default is True
        Distinguish between linear and non-linear nodal distributions.
    loc : :obj:`list`, default is None
        A list of nodes or coördinate tuples on which to concentrate nodal
        discretization.
    power : :obj:`int`, default is None
        The number of nodes that are influenced by the nodal concentrations
        given at the `loc` parameter.
    weight : :obj:`int` of :obj:`float`
        The degree of nodal concentration around a given `loc`.

    Returns
    -------
    x-dim : numpy.ndarray
        Numpy Array that contains nodal positions in the x-direction.
    y-dim : numpy.ndarray
        Numpy Array that contains nodal positions in the y-direction.

    Notes
    -----
    Make sure that the `power` argument does not collide with
    adjacent nodes or boundary nodes that define the domain.
    Improper use will result in wrongly defined domains.

    Examples
    --------
    One dimensional linear example

    >>> nx, Lx = (11, 10)
    >>> x, _ = spacing(nx, Lx)
    >>> x
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

    One dimensional non-linear example, 1 node around the locations of
    interest are influenced by a factor weight

    >>> x, _ = spacing(nx, Lx, linear=False, loc=[4, 7], power=1, weight=3)
    >>> x
    array([ 0.        ,  1.22222222,  2.44444444,  3.66666667,  4.        ,
            4.33333333,  6.66666667,  7.        ,  7.33333333,  8.66666667,
           10.        ])

    .. plot::

        import matplotlib.pyplot as plt
        from waterflow.utility.spacing import spacing

        fig, ax = plt.subplots(figsize=(9.0, 2))
        fig.suptitle('Nodal discretization of 1D spacing function')
        x, _ = spacing(10, 10, linear=False, loc=[4, 7], power=1, weight=3)
        ax.scatter(x, [0 for i in range(len(x))], marker='*', color='blue')
        ax.set_xlabel('Distance (x)')
        ax.grid()

    Two dimensional non-linear example, 2 nodes around the location of interest
    are influenced by a factor weight.

    >>> ny, Ly = (11, 10)
    >>> x, y = spacing(nx, Lx, ny, Ly, linear=False, loc=[(4, 5)], power=2, weight=4)
    >>> x
    array([ 0.      ,  1.65625 ,  3.3125  ,  3.75    ,  4.      ,  4.25    ,
            4.6875  ,  6.015625,  7.34375 ,  8.671875, 10.      ])
    >>> y
    array([ 0.    ,  1.4375,  2.875 ,  4.3125,  4.75  ,  5.    ,  5.25  ,
            5.6875,  7.125 ,  8.5625, 10.    ])

    .. plot::

        import matplotlib.pyplot as plt
        from waterflow.utility.spacing import spacing

        x, y = spacing(11, 10, 11, 10, linear=False, loc=[(4, 5)], power=2, weight=4)

        fig, ax = plt.subplots(figsize=(9, 9))
        for i in y:
            ax.scatter(x, [i for _ in range(len(x))], marker='*', color='blue')

        ax.set_title('Nodal discretization of 2D spacing function')
        ax.set_xlabel('Distance (x)')
        ax.set_ylabel('Distance (y)')
        ax.grid()

    """
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
            xloc = np.unique(loc[:, 0][np.argsort(loc[:, 0])])
            yloc = np.unique(loc[:, 1][np.argsort(loc[:, 1])])
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
                                            (p-power + 1) - (loc[pts_i-1] + power))
                    axis[loc[pts_i-1]+power:p-power+1] = fill_left
            # fill axis right of the point with linear distance
            fill_right = np.linspace(axis[p+power],
                                     ax_arg[-1],
                                     len(axis) - (p+power))
            axis[p+power:] = fill_right
            # reassign axis at which spacing is completed
            axes[iloc] = axis
    return axes[0], axes[1]


def biasedspacing(numnodes, power, lb=0, rb=1, maxdist=None, length=1):
    """ One dimensional nodal spacing function

    Returns an array that contains a biased nodal distribution in
    which distances increase from left to right.

    Parameters
    ----------
    numnodes : :obj:`int`
        Total number of nodes that is used for the nodal spacing.
    power : :obj:`int` or :obj:`float`
        Degree of nodal shifting to the left of the domain.
    lb : :obj:`int` or :obj:`float`, default is 0
        Left bound of the domain.
    rb : :obj:`int` or :obj:`float`, default = 1
        Right bound of the domain.
    maxdist : :obj:`int` or :obj:`float`, default is None
        Maximum distance allowed between two nodes. The value
        is absolute and does not depend on `length`.
    length : :obj:`int` or :obj:`float`, default is 1
        Multiplier to scale the nodal positions.

    Returns
    -------
    x-dim : numpy.ndarray
        Positions of the nodes.

    Notes
    -----
    The following formula is implemented

    .. math:: S_{i+1} = \\frac{R - S_{i}}{(P * (N - 2 - i))} + S{i}
    .. math:: N - 2 - i > 0

    Examples
    --------
    Linear spacing with scaling

    >>> biasedspacing(11, 1, length=10)
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

    Non-linear spacing without scaling

    >>> biasedspacing(11, 3)
    array([0.        , 0.03703704, 0.07716049, 0.12110523, 0.16993272,
           0.22527054, 0.28983133, 0.36873896, 0.47394913, 0.64929942,
           1.        ])

    Non-linear spacing with scaling and custom domain boundaries

    >>> biasedspacing(11, 2, lb=-1, rb=1, length=10)
    array([-10.        ,  -8.88888889,  -7.70833333,  -6.44345238,
            -5.07316468,  -3.56584821,  -1.87011719,   0.10823568,
             2.58117676,   6.29058838,  10.        ])

    Non-linear spacing with maximum nodal distance limitation

    >>> biasedspacing(11, 5, maxdist=0.4, length=2)
    array([0.        , 0.04815903, 0.09747244, 0.15250051, 0.21499075,
           0.29019617, 0.4       , 0.8       , 1.2       , 1.6       ,
           2.        ])

    .. plot::

        import matplotlib.pyplot as plt
        from waterflow.utility.spacing import spacing, biasedspacing

        fig, [ax1, ax2] = plt.subplots(figsize=(9, 9), nrows=2, ncols=1, sharex=True)
        powers = list(range(1, 7))
        maxdists = [0.10 + 0.03 * i for i in range(len(powers))]
        for i, j in zip(powers, maxdists):
            a1, a2 = biasedspacing(11, i), biasedspacing(11, powers[-1], maxdist=j)
            ax1.scatter(a1, [i for _ in range(len(a1))], marker='*', color='blue')
            ax2.scatter(a2, [j for _ in range(len(a2))], marker='*', color='blue')

        ax1.set_title('Nodal discretization of 1D biasedspacing function')
        ax1.set_yticks(powers)
        ax1.set_ylabel('variable power, no maxdist')
        ax1.grid()
        ax2.set_yticks(maxdists)
        ax2.set_xlabel('Distance (x)')
        ax2.set_ylabel(f'Variable maxdist, fixed power({powers[-1]})')
        ax2.grid()
        plt.show()

    """

    # at least two nodes are needed to define a domain
    if numnodes <= 2:
        return np.array([lb, rb]) * length
    # equal spacing
    if power <= 1:
        return np.linspace(lb, rb, numnodes) * length

    arr = [lb]
    # build discretization iteratively
    for n in range(numnodes - 2, 0, -1):
        i = (rb - lb) / (power * n)
        arr.append(i + arr[-1])
        lb = arr[-1]
    arr.append(rb)
    arr = np.array(arr) * length

    # if maxdist is exceeded, shift nodes proportionally
    if maxdist:
        fraction_prev = 0
        for i in range(numnodes - 2):
            idxl, idxr = numnodes - i - 2, numnodes - i - 1
            dist = arr[idxr] - arr[idxl]
            fraction = (dist - maxdist) / dist
            if fraction >= 0:
                fraction_prev = fraction
                arr[idxl] += fraction * dist
            else:
                diff = arr[2:idxr+1] - arr[1:idxr]
                arr[1:idxr] += diff * fraction_prev
                break
    return arr


if __name__ == "__main__":
    import doctest
    doctest.testmod()
