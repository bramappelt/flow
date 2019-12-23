""" Miscellaneous helper functions """

import os
from functools import partial, update_wrapper


def initializer(func, *args, **kwargs):
    """ prepare a function with specific default arguments """
    pfunc = partial(func, *args, **kwargs)
    # update initialized function with function's original attributes
    update_wrapper(pfunc, func)
    return pfunc


def newdir(basepath, dirname):
    """ checks and/or creates new directory """
    newpath = os.path.join(basepath, dirname)
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    return newpath
