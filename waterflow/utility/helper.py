""" Miscellaneous helper functions """

import os
from functools import partial, update_wrapper


def converged(old_states, new_states, threshold):
    """ Check for convergence

    Examples
    --------

    >>> import numpy as np
    >>> from waterflow.utility.helper import converged
    >>> old_states = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> new_states = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> threshold = 0.1
    >>> converged(old_states, new_states, threshold)
    False

    >>> threshold = 0.101
    >>> converged(old_states, new_states, threshold)
    True

    >>> old_states = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> new_states = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> threshold = 0.0
    >>> converged(old_states, new_states, threshold)
    False

    """
    max_abs_change = max(abs(old_states - new_states))
    max_abs_allowed_change = max(abs(threshold * new_states))

    return max_abs_change < max_abs_allowed_change


def initializer(func, *args, **kwargs):
    """ Prepare or reduce a function with default arguments

    Reduce the number of positional arguments in ``func`` or change
    the default values of already set keyword arguments.

    Parameters
    ----------
    *args
        Positional arguments in ``func``.
    **kwargs
        Keyword arguments in ``func``.

    Returns
    -------
    `functools.partial`
        Initialized function.

    Notes
    -----
    This function is implemented as a combination of :func:`functools.partial`
    to adapt the function signature and :func:`functools.update_wrapper` to
    copy the metadata from ``func``.

    Examples
    --------

    >>> from waterflow.utility.helper import initializer
    >>> def f(a, b, c=3):
    ...     return a, b, c
    >>> f(1, 2)
    (1, 2, 3)
    >>> f = initializer(f, c=10)
    >>> f(1, 2)
    (1, 2, 10)
    >>> f = initializer(f, 11)
    >>> f(b=2)
    (11, 2, 10)

    """
    pfunc = partial(func, *args, **kwargs)
    # update initialized function with function's original attributes
    update_wrapper(pfunc, func)
    return pfunc


def newdir(basepath, dirname):
    """ Creates new directory

    Creates a new directory if it does not exist.

    Parameters
    ----------
    basepath : `str`
        Path to the location of the new directory.
    dirname : `str`
        Name of the new directory

    Returns
    -------
    `str`
        Absolute path including the new directory.

    """
    newpath = os.path.join(basepath, dirname)
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    return newpath


if __name__ == '__main__':
    import doctest
    doctest.testmod()
