""" Conductivity functions and a soil selector function """

import os
from collections import namedtuple

import pandas as pd

from waterflow import DATA_DIR


def soilselector(soils):
    """ Select soil(s) from the Staringreeks

    Select soil parameters from one or multiple soils as described in
    The Staringreeks :cite:`Wosten2001`.

    Parameters
    ----------
    soils : `list`
        Number(s) that correspond(s) to a soil in The Staringreeks.

    Returns
    -------
    `list` with `collections.namedtuple` objects
        Parameters of the selected soils in the same order as the input.
    `pandas.core.frame.DataFrame`
        Slice of the complete dataframe with the selected soils.
    `tuple` with `collections.namedtuple` objects
        Holds the parameter extrema of the selected soils.

    Notes
    -----
    The source file from which the soil data is read can be found in
    :py:data:`~waterflow.DATA_DIR`.

    Examples
    --------

    >>> from waterflow.utility.conductivityfunctions import soilselector
    >>> params, df, extrema = soilselector([1, 10, 13])
    >>> s1, s10, s13 = params
    >>> # Print some parameters of the selected soils
    >>> s1.soiltype, s1.name, s1.ksat
    ('B1', 'Non-loamy sand', 23.41)
    >>> s10.name, s10.t_sat, s10.alpha
    ('Light clay', 0.43, 0.0064)
    >>> s13.name, s13.Lambda, s13.n
    ('Loam', -1.4969999999999999, 1.4409999999999998)
    >>> df
       soiltype  t_res  t_sat   ksat   alpha  Lambda      n            name category
    0        B1   0.02   0.43  23.41  0.0234   0.000  1.801  Non-loamy sand        s
    9       B10   0.01   0.43   0.70  0.0064  -3.884  1.210      Light clay        c
    12      B13   0.01   0.42  12.98  0.0084  -1.497  1.441            Loam        l
    >>> extrema[0].ksat
    0.7

    """
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


def VG_theta(theta, theta_r, theta_s, a, n):
    """ Water retention function, :math:h(theta)

    Soil water retention function as described by
    :cite:`VanGenuchten1980`.

    Parameters
    ----------
    theta : `float`
        Water content as a fraction.
    theta_r : `float`
        Residual water content as a fraction.
    theta_s : `float`
        Saturated water content as a fraction.
    a : `float`
        Empirical soil parameter :math:`\\left(\\frac{1}{length}\\right)`.
    n : `float`
        Empirical soil parameter :math:`\\left(-\\right)`.

    Returns
    -------
    `float`
        Soil water potential :math:`\\left(length\\right)`. Note that the sign
        of the value is switched.

    Notes
    -----
    This version of the water retention function is based on the
    Mualem approach with is described by :cite:`Mualem1976` and
    :cite:`VanGenuchten1980`. Under this approach the parameter
    :math:`m` is fixed.

    .. math::
        m = 1 - \\frac{1}{n}

    .. math::
        h(\\theta) = \\left(\\frac{(\\theta_{s} - \\theta_r) /
                     (\\theta - \\theta_{r})^{\\frac{1}{m}} - 1}
                     {a^{n}}\\right)^{\\frac{1}{n}}

    .. note::
        For usage in the flow model,
        :py:class:`~waterflow.flow1d.flowFE1d.Flow1DFE`, the number of
        positional arguments needs to be reduced to one (``theta``)
        and the remaining should be assigned a default value. See the
        examples section.

    Examples
    --------
    >>> from waterflow.utility.conductivityfunctions import soilselector, VG_theta
    >>> from waterflow.utility.helper import initializer
    >>> soil = soilselector([2])
    >>> p = soil[0][0]
    >>> # Without preparation
    >>> round(VG_theta(0.25, p.t_res, p.t_sat, p.alpha, p.n), 4)
    97.3908
    >>> # With preparation
    >>> VG_theta = initializer(VG_theta, theta_r=p.t_res, theta_s=p.t_sat,
    ...                        a=p.alpha, n=p.n)
    >>> round(VG_theta(0.25), 4)
    97.3908

    """
    m = 1-1/n
    THETA = (theta_s - theta_r) / (theta - theta_r)
    return ((THETA**(1/m) - 1) / a**n)**(1/n)


def VG_pressureh(h, theta_r, theta_s, a, n):
    """ Water retention function :math:`theta(h)`

    Soil water retention function as described by
    :cite:`VanGenuchten1980`.

    Parameters
    ----------
    h : `float`
        Soil water potential :math:`\\left(length\\right)`.
    theta_r : `float`
        Residual water content as a fraction.
    theta_s : `float`
        Saturated water content as a fraction.
    a : `float`
        Empirical soil parameter :math:`\\left(\\frac{1}{length}\\right)`.
    n : `float`
        Empirical soil parameter :math:`\\left(-\\right)`.

    Returns
    -------
    `float`
        Moisture content as a fraction.

    Notes
    -----
    This version of the water retention function is based on the
    Mualem approach with is described by :cite:`Mualem1976` and
    :cite:`VanGenuchten1980`. Under this approach the parameter
    :math:`m` is fixed.

    .. math ::
        m = 1-\\frac{1}{n}

    .. math ::
        \\theta(h) = \\begin{cases}
                        \\theta_{r} + \\frac{\\theta_{s} - \\theta_{r}}{(1+(a*-h)^{n})^m},
                        & \\text{if } h < 0\\\\
                        \\theta_s, & \\text{otherwise}
                     \\end{cases}

    .. note::
        For usage in the flow model,
        :py:class:`~waterflow.flow1d.flowFE1d.Flow1DFE`, the number of
        positional arguments needs to be reduced to one (``h``)
        and the remaining should be assigned a default value. See the
        examples section.

    Examples
    --------
    >>> from waterflow.utility.conductivityfunctions import soilselector, VG_pressureh
    >>> from waterflow.utility.helper import initializer
    >>> soil = soilselector([8])
    >>> p = soil[0][0]
    >>> # Without preparation
    >>> round(VG_pressureh(-10**4.2, p.t_res, p.t_sat, p.alpha, p.n), 4)
    0.1079
    >>> # With preparation
    >>> VG_pressureh = initializer(VG_pressureh, theta_r=p.t_res,
    ...                            theta_s=p.t_sat, a=p.alpha, n=p.n)
    >>> round(VG_pressureh(-10**4.2), 4)
    0.1079

    """
    # to theta
    if h >= 0:
        return theta_s
    m = 1-1/n
    return theta_r + (theta_s-theta_r) / (1+(a*-h)**n)**m


def VG_conductivity(x, h, ksat, a, n):
    """ Hydraulic conductivity function

    Unsaturated hydraulic conductivity function as described by
    :cite:`VanGenuchten1980`.

    Parameters
    ----------
    x : `float`
        Positional argument :math:`\\left(length\\right)`.
    h : `float`
        Soil water potential :math:`\\left(length\\right)`.
    ksat : `float`
        Saturated hydraulic conductivity :math:`\\left(\\frac{length}{time}\\right)`.
    a : `float`
        Empirical soil parameter :math:`\\left(\\frac{1}{length}\\right)`.
    n : `float`
        Empirical soil parameter :math:`\\left(-\\right)`.

    Returns
    -------
    `float`
        Hydraulic conductivity :math:`\\left(\\frac{length}{time}\\right)`.

    Notes
    -----
    This is the hydraulic conductivity function with the Mualem approach which
    means that parameter :math:`m` is fixed as described by :cite:`Mualem1976`
    and :cite:`VanGenuchten1980`.

    .. math ::
        m = 1 - \\frac{1}{n}

    .. math ::
        k(h) = \\begin{cases}
                    \\frac{\\left(1 - (a*-h)^{n-1}*(1+(a*-h)^{n})^{-m}\\right)^{2}}
                    {(1 + (a*-h)^{n})^{\\frac{m}{2}}} * ksat, & \\text{if } h<0 \\\\
                    ksat, & \\text{otherwise}
               \\end{cases}

    .. note::
        For usage in the flow model,
        :py:class:`~waterflow.flow1d.flowFE1d.Flow1DFE`, the number of
        positional arguments needs to be reduced to two (``x`` & ``h``).
        Despite independence of argument ``x`` in this function is needs to
        be included in the function signature. The other arguments need to be
        assigned a default value. See the examples section.

    Examples
    --------

    >>> from waterflow.utility.conductivityfunctions import soilselector, VG_conductivity
    >>> from waterflow.utility.helper import initializer
    >>> soil = soilselector([3])
    >>> p = soil[0][0]
    >>> # Without preparation
    >>> round(VG_conductivity(1, -10**2, p.ksat, p.alpha, p.n), 4)
    0.2742
    >>> # With preparation
    >>> VG_conductivity = initializer(VG_conductivity, ksat=p.ksat, a=p.alpha, n=p.n)
    >>> round(VG_conductivity(10, -10**2), 4)
    0.2742

    """
    if h >= 0:
        return ksat
    m = 1-1/n
    h_up = (1 - (a * -h)**(n-1) * (1 + (a * -h)**n)**-m)**2
    h_down = (1 + (a * -h)**n)**(m / 2)
    return (h_up / h_down) * ksat


if __name__ == "__main__":
    import doctest
    doctest.testmod()
