""" Several flow equations for (un)saturated flow problems and a
storage change function """


def darcy(x, s, gradient, ksat=1):
    """ Flux function for saturated flow

    Notes
    -----

    .. math ::

        q(x, s, \\frac{\\delta s}{\\delta x}) =
        -\\frac{\\delta s}{\\delta x} * ksat

    """
    return - gradient * ksat


def darcy_s(x, s, gradient, ksat=1):
    """ Flux function for saturated flow

    Notes
    -----

    .. math ::

        q(x, s, \\frac{\\delta s}{\\delta x}) =
        -s * \\frac{\\delta s}{\\delta x} * ksat

    """

    return - gradient * ksat * s


def darcy_k(x, s, gradient, kfun=lambda x: 1):
    """ Flux function for saturated flow

    Notes
    -----

    .. math::

        q(x, s, \\frac{\\delta s}{\\delta x}) =
        -kfun(x) * \\frac{\\delta s}{\\delta x}

    """

    return - kfun(x) * gradient


def darcy_k_s(x, s, gradient, kfun=lambda x: 1):
    """ Flux function for saturated flow

    Notes
    -----

    .. math::

        q(x, s, \\frac{\\delta s}{\\delta x}) =
        -kfun(x) * s * \\frac{\\delta s}{\\delta x}

    """
    return - kfun(x, s) * s * gradient


def richards_equation(x, s, gradient, kfun):
    """ Richards equation, for unsaturated flow

    Notes
    -----

    .. math::

        q(x, s, \\frac{\\delta s}{\\delta x}) =
        -kfun(x, s) * \\left(\\frac{\\delta s}{\\delta x} + 1\\right)

    """
    return -kfun(x, s) * (gradient + 1)


def storage_change(x, s, prevstate, dt, fun=lambda x: 1, S=1):
    """ General storage change function for both saturated and
    unsaturated flow simulations

    Notes
    -----

    .. math::

        q(x, s, prevstate, dt, fun, S) = - S * \\frac{fun(s) - fun(pressure(x))}{dt}

    """
    return - S * (fun(s) - fun(prevstate(x))) / dt


if __name__ == '__main__':
    pass
