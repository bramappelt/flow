''' This package contains several internal flux functions. '''

def fluxfunction(x, s, gradient, ksat=1):
    return - gradient * ksat

def fluxfunction_s(x, s, gradient, ksat=1):
    return - gradient * ksat * s

def fluxfunction_var_k(x, s, gradient, kfun=lambda x: 1):
    return - kfun(x) * gradient

def fluxfunction_var_k_s(x, s, gradient, kfun=lambda x: 1):
    return - kfun(x) * s * gradient

def richards_equation(x, psi, gradient, kfun):
    return -kfun(psi) * (gradient + 1)
