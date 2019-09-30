""" This package contains several one dimensional flux functions
and a storage change function """


def fluxfunction(x, s, gradient, ksat=1):
    """ flux function for saturated flow """
    return - gradient * ksat


def fluxfunction_s(x, s, gradient, ksat=1):
    """ flux function for saturated flow """
    return - gradient * ksat * s


def fluxfunction_var_k(x, s, gradient, kfun=lambda x: 1):
    """ flux function for saturated flow """
    return - kfun(x) * gradient


def fluxfunction_var_k_s(x, s, gradient, kfun=lambda x: 1):
    return - kfun(x) * s * gradient


def richards_equation(x, s, gradient, kfun):
    """ Richards equation """
    return -kfun(x, s) * (gradient + 1)


def storage_change(x, s, prevstate, dt, fun=lambda x: 1, S=1):
    """ General storage change function for both saturated and
    unsaturated flow simulations
    """
    return - S * (fun(s) - fun(prevstate(x))) / dt


if __name__ == '__main__':
    pass
