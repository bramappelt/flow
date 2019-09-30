""" Module with helper functions """

from functools import partial, update_wrapper


def initializer(func, *args, **kwargs):
    """ prepare a function with specific default arguments """
    pfunc = partial(func, *args, **kwargs)
    # update initialized function with function's original attributes
    update_wrapper(pfunc, func)
    return pfunc
