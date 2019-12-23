""" Conductivity functions and a soil selector function """

import os
from collections import namedtuple

import pandas as pd

from waterflow import DATA_DIR


def soilselector(soils):
    """ Select soil(s) from the Staringreeks """
    staringreeks = pd.read_table(os.path.join(DATA_DIR, "StaringReeks.txt"),
                                 delimiter="\t")
    soildata = staringreeks.iloc[[s-1 for s in soils]]

    # useful for a plotting domain
    minima = namedtuple('min', staringreeks.columns)(*soildata.min())
    maxima = namedtuple('max', staringreeks.columns)(*soildata.max())
    extrema = (minima, maxima)

    # turn row(s) to namedtuple
    rows = list(soildata.itertuples(name='soil', index=False))
    return rows, soildata, extrema


# Van Genuchten (theta)
def VG_theta(theta, theta_r, theta_s, a, n):
    """ h-theta relation as described by Van Genuchten

    Notes
    -----

    .. math::

        m = 1 - \\frac{1}{n}

    .. math::

        h(\\theta) = \\left(\\frac{(\\theta_{s} - \\theta_r) /
        (\\theta - \\theta_{r})^{\\frac{1}{m}} - 1}
        {a^{n}}\\right)^{\\frac{1}{n}}

    """
    m = 1-1/n
    THETA = (theta_s - theta_r) / (theta - theta_r)
    return ((THETA**(1/m) - 1) / a**n)**(1/n)


# Van Genuchten (h)
def VG_pressureh(h, theta_r, theta_s, a, n):
    """ theta-h relation as described by Van Genuchten

    Notes
    -----

    .. math ::

        m = 1-\\frac{1}{n}

    .. math ::

            \\theta(h) =
        \\begin{cases}
            \\theta_{r} + \\frac{\\theta_{s} - \\theta_{r}}{(1+(a*-h)^{n})^m},
            & \\text{if } h < 0\\\\
            \\theta_s, & \\text{otherwise}
        \\end{cases}

    """

    # to theta
    if h >= 0:
        return theta_s
    m = 1-1/n
    return theta_r + (theta_s-theta_r) / (1+(a*-h)**n)**m


# Van Genuchten (x, h)
def VG_conductivity(x, h, ksat, a, n):
    """ k-h relation as described by Van Genuchten

    Notes
    -----

    .. math ::

        m = 1 - \\frac{1}{n}

    .. math ::

            k(h) =
        \\begin{cases}
            \\frac{\\left(1 - (a*-h)^{n-1}*(1+(a*-h)^{n})^{-m}\\right)^{2}}
            {(1 + (a*-h)^{n})^{\\frac{m}{2}}} * ksat, & \\text{if } h<0\\\\
            ksat, & \\text{otherwise}
        \\end{cases}

    """

    if h >= 0:
        return ksat
    m = 1-1/n
    h_up = (1 - (a * -h)**(n-1) * (1 + (a * -h)**n)**-m)**2
    h_down = (1 + (a * -h)**n)**(m / 2)
    return (h_up / h_down) * ksat


if __name__ == "__main__":
    pass
