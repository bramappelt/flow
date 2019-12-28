""" Several flow equations for (un)saturated flow problems and a
storage change function """


def darcy(x, s, gradient, kfun=lambda x, s: 1):
    """ Flux function for saturated flow

    Flow equation as first described by :cite:`Darcy1856`.

    Parameters
    ----------
    x : `float`
        Positional argument :math:`\\left(length\\right)`.
    s : `float`
        State of the system :math:`\\left(length\\right)`.
    gradient : `float`
        Gradient :math:`\\frac{\\delta s}{\\delta x}`.
    kfun : `func`, default is :math:`kfun(x,s) = 1`
        Hydraulic conductivity function with signature :math:`kfun(x, s)`
        :math:`\\left(\\frac{length}{time}\\right)`.

    Returns
    -------
    `float`
        Flux value :math:`\\left(\\frac{length}{time}\\right)`.

    Notes
    -----
    See the exact implemention of the :cite:`Darcy1856` function below:

    .. math::
        q(x, s, \\frac{\\delta s}{\\delta x}) =
        -kfun(x, s) * \\frac{\\delta s}{\\delta x}

    .. tip::
        The function :math:`kfun` can return a fixed value if the saturated
        conductivity is needed instead of a hydraulic conductivity function.
        The ``kfun`` argument may look like the following:

            .. math::
                kfun(x, s) = ksat

        Which can be implemented in Python with a lambda function
        (lambda x, s: ksat) or just with a common function.

    Examples
    --------

    >>> from waterflow.utility.fluxfunctions import darcy
    >>> from waterflow.utility.helper import initializer
    >>> # In this case ksat is 1, negative gradient is returned
    >>> darcy(10, 5, 0.01)
    -0.01
    >>> # Prepare the fluxfunction with a different default argument
    >>> def kfun(x, s):
    ...     return -0.001 * x + s**1.2
    >>> darcy = initializer(darcy, kfun=kfun)
    >>> round(darcy(10, 5, 0.01), 4)
    -0.0689

    """
    return - kfun(x, s) * gradient


def darcy_s(x, s, gradient, kfun=lambda x, s: 1):
    """ Flux function for saturated flow

    Flow equation as first described by :cite:`Darcy1856` which
    is altered to include a state dependency.

    Parameters
    ----------
    x : `float`
        Positional argument :math:`\\left(length\\right)`.
    s : `float`
        State of the system :math:`\\left(length\\right)`.
    gradient : `float`
        Gradient :math:`\\frac{\\delta s}{\\delta x}`.
    kfun : `func`, default is :math:`kfun(x,s) = 1`
        Hydraulic conductivity function with signature :math:`kfun(x, s)`
        :math:`\\left(\\frac{length}{time}\\right)`.

    Returns
    -------
    `float`
        Flux value :math:`\\left(\\frac{length}{time}\\right)`.

    Notes
    -----
    See the exact implemention of the adapted function below:

    .. math::
        q(x, s, \\frac{\\delta s}{\\delta x}) =
        -kfun(x, s) * s * \\frac{\\delta s}{\\delta x}

    .. tip::
        The function :math:`kfun` can return a fixed value if the saturated
        conductivity is needed instead of a hydraulic conductivity function.
        The ``kfun`` argument may look like the following:

            .. math::
                kfun(x, s) = ksat

        Which can be implemented in Python with a lambda function
        (lambda x, s: ksat) or just with a common function.

    Examples
    --------

    >>> from waterflow.utility.fluxfunctions import darcy_s
    >>> from waterflow.utility.helper import initializer
    >>> # In this case ksat is 1, negative gradient is returned
    >>> darcy_s(10, 5, 0.01)
    -0.05
    >>> # Prepare the fluxfunction with a different default argument
    >>> def kfun(x, s):
    ...     return -0.001 * x + s**1.2
    >>> darcy_s = initializer(darcy_s, kfun=kfun)
    >>> round(darcy_s(10, 5, 0.01), 4)
    -0.3444

    """
    return - kfun(x, s) * s * gradient


def richards_equation(x, s, gradient, kfun):
    """ Flux function for unsaturated flow

    Flux function for unsaturated flow as described by :cite:`Richards1970`.

    Parameters
    ----------
    x : `float`
        Positional argument :math:`\\left(length\\right)`.
    s : `float`
        State of the system :math:`\\left(length\\right)`.
    gradient : `float`
        Gradient :math:`\\frac{\\delta s}{\\delta x}`.
    kfun : `func`
        Unsaturated hydraulic conductivity function with signature
        :math:`kfun(x, s)` :math:`\\left(\\frac{length}{time}\\right)`.

    Returns
    -------
    `float`
        Flux value :math:`\\left(\\frac{length}{time}\\right)`.

    Notes
    -----
    See the exact implementation of the richards equation :cite:`Richards1970`
    below:

    .. math::
        q(x, s, \\frac{\\delta s}{\\delta x}) =
        -kfun(x, s) * \\left(\\frac{\\delta s}{\\delta x} + 1\\right)

    .. note::
        For usage in the flow model,
        :py:class:`~waterflow.flow1d.flowFE1d.Flow1DFE`, the calling signature
        needs to be reduced to 3 positional arguments. ``kfun`` should be given
        a default value, see the examples section.

    Examples
    --------
    >>> from waterflow.utility.fluxfunctions import richards_equation
    >>> from waterflow.utility.conductivityfunctions import VG_pressureh
    >>> from waterflow.utility.helper import initializer
    >>> # kfun itself needs to be initialized too
    >>> VG_pressureh = initializer(VG_pressureh, theta_s=0.42, a=0.0748, n=1.44)
    >>> # Prepare the unsaturated hydraulic conductivity function
    >>> richards_equation = initializer(richards_equation, kfun=VG_pressureh)
    >>> richards_equation(0, -10**4.2, 0.1)
    -0.462

    """
    return -kfun(x, s) * (gradient + 1)


def storage_change(x, s, prevstate, dt, fun=lambda x: x, S=1.0):
    """ Storage change function

    General storage change function for both saturated and
    unsaturated flow simulations

    Parameters
    ----------
    x : `float`
        Positional argument :math:`\\left(length\\right)`.
    s : `float`
        State of the system :math:`\\left(length\\right)`.
    prevstate : `func`
        Converts a position, :math:`x`, to a state value :math:`s`.
    dt : `float`
        Time step `\\left(time\\right)`.
    fun : `func`, default is :math:`fun(x) = x`
        Convert a state value, :math:`s`, to a moisture content, :math:`\\theta`,
        in case of unsaturated flow.
    S : `float`, default is 1.0
        Sorptivity as a fraction.

    Returns
    -------
    `float`
        Flux value :math:`\\left(\\frac{length}{time}\\right)`.

    Notes
    -----
    Below the exact implementation of the storage change function is shown:

    .. math::
        q(x, s, prevstate, dt) = - S * \\frac{fun(s) - fun(prevstate(x))}{dt}

    With specific arguments the storage change function can be used for
    unsaturated flow problems:

    .. math::
        q(x, s, prevstate, dt) = - \\frac{fun(s) - fun(prevstate(x))}{\\Delta t}

    See the implementation for saturated flow problems below:

    .. math::
        q(x, s, prevstate, dt) = - \\frac{s - prevstate(x)}{\\Delta t}

    :math:`fun` refers to a :math:`theta(h)`-relation of which one is
    defined in :py:func:`~waterflow.utility.conductivityfunctions` and
    :math:`prevstate(x)` calculates the states of the previous time step
    as a function of position. The ready to use function can be found in
    :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.states_to_function()`.

    .. note::
        The Storage change function is a special case of an external flux
        function as described in
        :py:class:`~waterflow.flow1d.flowFE1d.Flow1DFE.add_spatialflux` and
        assumes four positional arguments and any amount of keyword arguments,
        having a default value.

    """
    return - S * (fun(s) - fun(prevstate(x))) / dt


if __name__ == '__main__':
    import doctest
    doctest.testmod()
