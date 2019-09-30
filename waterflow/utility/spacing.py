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


def biasedspacing(numnodes, power, length=1):
    """ One dimensional nodal spacing function

    Returns an array which contains a biased nodal distribution.

    Parameters
    ----------
    numnodes : :obj:`int`
        Total number of nodes that is used for the nodal spacing.
    power : :obj:`int`
        Degree of nodal shifting to the left of the domain.
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

    >>> biasedspacing(11, 1, 10)
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

    Non-linear spacing without scaling

    >>> biasedspacing(11, 3)
    array([0.        , 0.03703704, 0.07716049, 0.12110523, 0.16993272,
           0.22527054, 0.28983133, 0.36873896, 0.47394913, 0.64929942,
           1.        ])

    """
    left, right = (0, 1)

    # at least two nodes are needed to define a domain
    if numnodes <= 2:
        return np.array([left, right]) * length
    # equal spacing
    if power <= 1:
        return np.linspace(0, 1, numnodes) * length

    arr = [left]
    # build discretization iteratively
    for n in range(numnodes-2, 0, -1):
        i = (right - left) / (power * n)
        arr.append(i + arr[-1])
        left = arr[-1]
    arr.append(right)
    return np.array(arr) * length


if __name__ == "__main__":
    import doctest
    doctest.testmod()
