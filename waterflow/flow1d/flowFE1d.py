""" One dimensional finite elements flow module """


from inspect import signature
from functools import partial
from copy import deepcopy
import time as Time
import os

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import legendre

from waterflow import OUTPUT_DIR


class Flow1DFE:
    """ Class for solving flow problems numerically

    This class represents an object that can be used to solve
    (un)saturated 1-dimensional flow problems using finite elements.
    To increase the accuracy of numerical solutions the Gaussian
    quadrature method is used for integration approximation.

    Most of the methods applied on the object will change its internal
    state rather than returning a value. The change of the system is
    saved in any of its public attributes.

    Parameters
    ----------
    id_ : `str`
        Name of the model object.

    savepath : `str`, optional
        Directory to which model data will be saved.

    Attributes
    ----------
    id_ : `str`
        Name of the model object.

    savepath: `str`
        Model's save directory.

    systemfluxfunc : `function`
        Holds the selected flux function.

    nodes : `numpy.ndarray`
        Nodal positions at which the system will be solved.

    states : `numpy.ndarray`
        State solutions at the nodal positions as defined in :py:attr:`~nodes`.

    nframe : `numpy.ndarray`
        Two dimensional array that contains the midpoints and the lengths of
        the nodal discretization in its columns respectively.

    lengths : `numpy.ndarray`
        The same data as in the seconds column of :py:attr:`~nframe` but
        in a different representation. This representation has the same length
        as :py:attr:`~nodes` which is more convenient for certain forcing
        calculations at the nodal positions.

    coefmatr : `numpy.ndarray`
        Square jacobian matrix used in the finite elements solution procedure.

    BCs : `dict`
        This contains the system's boundary conditions. The keys that indicate
        the positions are "west" and "east". The corresponding values have the
        following format:

        * (boundary_condition_value, type, domain_index).

    spatflux : `dict`
        Contains the spatial fluxes on the model domain. # !! explain form

    pointflux : `dict`
        Contains the point fluxes on the model domain. # !! explain form

    Spointflux : `dict`
        Contains state dependent point fluxes on the model domain. # !! explain form

    Sspatialflux : `dict`
        Contains state dependent spatial fluxes on the model domain. # !! explain form

    internal_forcing : `dict`
        The internal forcing of the system as calculated with the
        system flux function and saved in :py:attr:`~systemfluxfunc`,
        using the selected Gaussian quadrature :py:attr:`~degree_degree`.

    forcing : `numpy.ndarray`
        pass

    conductivities : `numpy.ndarray`
        pass

    moisture : `numpy.ndarray`
        pass

    fluxes : `numpy.ndarray`
        Fluxes through the :py:attr:`~nodes` are defined to be positive to the
        right.

    isinitial : `bool`, default is True
        No calculations are performed on the input data yet.

    isconverged : `bool`, default is False
        The system has converged to a solution.

    solve_data : `dict`
        Holds the solve information of the system if :py:attr:`~isinitial`
        equals False including the following key-value pairs:

        * solved_objects - A `list` of Flow1DFE objects at solved time steps.

        * time - A `list` of times at which the model states are calculated.

        * dt - A `list` of time step sizes between consecutive model solutions.

        * | iter - A `list` containing the number of iterations needed for
          | consecutive model solutions to converge.

    runtime : `float`
        The total time it takes for :py:meth:`~solve` to find a solution.

    df_states : `pandas.core.frame.DataFrame`
        Current information about the static model solution.

    df_balance : `pandas.core.frame.DataFrame`
        Current static information about the water balance.

    df_balance_summary : `pandas.core.frame.DataFrame`
        Sum of the columns as saved in :py:attr:`~df_balance`.

    dft_solved_times : `pandas.core.frame.DataFrame`
        Dataframe version of :py:attr:`~solve_data`.

    dft_print_times : `pandas.core.frame.DataFrame`
        Objects that contain a solution to the model at specific times,
        calculated with :py:meth:`~transient_dataframify`.

    dft_states : `dict`
        Collection of all :py:attr:`~df_states` dataframes at
        :py:attr:`~solved_times` or at :py:attr:~`print_times` if not `None`.

    dft_nodes : `dict`
        Nodes that are selected in :py:meth:`~transient_dataframify` are
        saved at :py:attr:`~solved_times` or at :py:attr:~`print_times`
        if not `None`.

    dft_balance : `dict`
        Collection of all :py:attr:`~df_balance` dataframes at
        :py:attr:`~solved_times` or at :py:attr:~`print_times` if not `None`.

    dft_balance_summary : `pandas.core.frame.DataFrame`
        Collection of all :py:attr:`~df_balance_summary` dataframes at
        :py:attr:`~solved_times` or at :py:attr:~`print_times` if not `None`.

    _west : `int`
        pass

    _east : `int`
        pass

    _delta : `float`
        pass

    gauss_degree : `str`
        Degree or number of points used in the Gaussian quadrature procedure
        for integral approximation.

    _xgauss : `tuple`
        Roots of Legendre polynomial on the interval [0, 1] for the selected
        :py:attr:~`gauss_degree`.

    _wgauss : `tuple`
        Weights that correspond to :py:attr:`~_xgauss`.

    gaussquad : `list`
        Combination of corresponding values of :py:attr:`~_xgauss` and
        :py:attr:`~wgauss` into a single data structure.

    xintegration : `list`
        Absolute positions of the Gaussian quadrature points in the domain
        :py:attr:`~nodes`. The shape of this list is :py:attr:`~degree` by
        length :py:attr:`~nodes~ minus 1.

    """

    def __init__(self, id_, savepath=OUTPUT_DIR):
        self.id = id_
        self.savepath = savepath
        self.systemfluxfunc = None
        self.nodes = None
        self.states = None
        self.nframe = None
        self.lengths = None
        self.coefmatr = None
        self.BCs = {}
        self.pointflux = {}
        self.spatflux = {}
        self.Spointflux = {}
        self.Sspatflux = {}
        self.internal_forcing = {}
        self.forcing = None
        self.conductivities = []
        self.moisture = []
        self.fluxes = []
        self.stats = {"rmse": [], "mae": []}  # ## ??
        self.isinitial = True
        self.isconverged = False
        self.solve_data = None
        self.runtime = None
        # dataframes
        self.df_states = None
        self.df_balance = None
        self.df_balance_summary = None
        self.dft_solved_times = None
        self.dft_print_times = None
        self.dft_states = None
        self.dft_nodes = None
        self.dft_balance = None
        self.dft_balance_summary = None
        # private attributes
        self._west = None
        self._east = None
        self._delta = 1e-5
        # specific
        self.gauss_degree = 1
        self._xgauss = None
        self._wgauss = None
        self.gaussquad = None
        self.xintegration = None

    def __repr__(self):
        """ Representation of the object as shown to the user """
        return "Flow1DFE(" + str(self.id) + ")"

    def summary(self, show=True):
        """ Description of the object

        Parameters
        ----------
        show : `bool`, default is True
            Print object description to the console

        Notes
        -----

        .. note::

            The dataframe that might be included is based on
            :py:attr:`~df_balance_summary` and is only included if the model
            has been solved for.

        """

        id_ = f"{self.id}"
        if self.nodes is not None:
            len_ = str(self.nodes[-1] - self.nodes[0])
            num_nodes = str(len(self.nodes))
        else:
            len_ = None
            num_nodes = None
        degree = self.gauss_degree
        bcs = [[k, self.BCs[k][0], self.BCs[k][1]] for k in self.BCs.keys()]
        bcs = ["{} value: {} and of type {}".format(*bc) for bc in bcs]
        bcs = ", ".join(i for i in bcs)
        pkeys = list(self.pointflux.keys()) + list(self.Spointflux.keys())
        skeys = list(self.spatflux.keys()) + list(self.Sspatflux.keys())
        pointflux = ", ".join(i for i in pkeys)
        spatflux = ", ".join(i for i in skeys)
        runtime = self.runtime

        k = ['Id', 'System length', 'Number of nodes', 'Gauss degree',
             'BCs', 'Pointflux', 'Spatflux', 'Runtime (s)']
        v = (id_, len_, num_nodes, degree, bcs, pointflux, spatflux, runtime)
        sumstring = ""
        for i, j in zip(k, v):
            if j:
                sumstring += f"{i}: {j}\n"

        if show:
            for s in sumstring.split('\n'):
                print(s)

            try:
                self.calcbalance()
                print(self.df_balance_summary)
            except Exception:
                pass

        self.summarystring = sumstring

    def set_gaussian_quadrature(self, degree=1):
        """ Calculates Gaussian quadrature roots and weights

        The values calculated with this method are stored in the object's
        :py:attr:`~_xgauss` and :py:attr:`~_wgauss` attributes. The absolute
        positions of the Gaussian quadrature points in the :py:attr:`~domain`
        are calculated and saved in ~xintegration`.

        Parameters
        ----------
        degree : `int`, default is 1
            Number of points used in the Gaussian quadrature procedure.

        Notes
        -----

        .. math::
                P_{n}(x) = \\text{Legendre polynomials of degree n}

        .. math::
                w_{i} = \\frac{2}{\\left(\\left(1 - x_{i}^{2}\\right) *
                (P_{n}^{'}(x_{i})^{2}\\right)}


        See :cite:`Strunk1979` for an introduction to stylish blah, blah...

        References
        ----------
        .. bibliography:: bibliography.bib

        """
        # calculate roots of Legendre polynomial for degree n
        legn = legendre(degree)
        roots = np.sort(legn.r)

        # calculate corresponding weights
        weights = 2 / ((1 - roots**2) * (legn.deriv()(roots)) ** 2)

        # shift roots and weights from domain [-1, 1] to [0, 1]
        roots = tuple(roots / 2 + 0.5)
        weights = tuple(weights / 2)

        # Calculate absolute positions of Gaussian quadrature roots
        xintegration = [[] for x in range(len(self.nodes) - 1)]
        for i in range(len(self.nodes) - 1):
            for j in range(len(roots)):
                xij = (self.nodes[i+1] - self.nodes[i]) * roots[j] + \
                       self.nodes[i]
                xintegration[i].append(xij)

        self._xgauss = roots
        self._wgauss = weights
        self.gaussquad = [self._xgauss, self._wgauss]
        self.xintegration = xintegration
        self.gauss_degree = degree

    def _aggregate_forcing(self):
        """ Aggregation of state independent forcing """
        self.forcing = np.repeat(0.0, len(self.nodes))
        # aggregate state independent forcing
        for flux in [self.pointflux, self.spatflux]:
            for key in flux.keys():
                self.forcing += flux[key][-1]

        # add boundaries of type neumann
        for key in self.BCs.keys():
            val, type_, idx = self.BCs[key]
            if type_ == "Neumann":
                self.forcing[idx] += val

    def _internal_forcing(self, calcflux=False, calcbal=False):
        """ Calculate the system's internal forcing

        This is a core method for the numerical finite elements scheme. The
        default behavior is to calculate the system's internal forcing and
        assign the values to :py:attr:`~forcing`.

        Parameters
        ----------
        calcflux : `bool`, default is False
            If ``True``, fluxes through the nodes are calculated and
            saved in :py:attr:`~fluxes`. In this case only the second
            summation in the `notes` section is applied.
        calcbal : `bool`, default = False
            If ``True``, internal forcing is saved in
            :py:attr:`~internal_forcing` instead of :py:attr:`~forcing`.

        Notes
        -----

        This mathematical description of the internal forcing
        calculation describes how the forcing at nodes :math:`i` and
        :math:`i+1` is calculated. To calculate this forcing at the current
        time step :math:`t`, states from the previous time step or initial
        states assigned through :py:meth:`~set_initial_states` are needed,
        this is denoted by :math:`t-1` in the subscripts. These functions are
        applied to all the :py:attr:`~nodes` in the domain. :math:`j` denotes
        the indices of the Gaussian quadrature positions and weights. The
        direction of flow is defined to be positive to the right.

        .. math::
            F_{i,t} = - \\sum_{j=0}^{degree-1} Q(x_{i,j}, s_{j,t-1}, grad_{j,t-1})*weight_{-j-1}

        .. math::
            F_{i+1,t} = \\sum_{j=0}^{degree-1} Q(x_{i,j}, s_{j,t-1}, grad_{j,t-1})*weight_{j}

        * :math:`degree` corresponds to :py:attr:`~gauss_degree`.
        * :math:`weight` is described in :py:attr:`~_wgauss`.
        * :math:`Q(x, s, grad)` is the :py:attr:`~systemfluxfunc`.

        .. centered::
                :math:`Q(x_{i,j}, s_{j,t-1}, grad_{j,t-1}) =`
        .. centered::
                :math:`Q(x_{i,j},s_{i,t-1}*pos_{-j-1}+s_{i+1,t-1}*pos_{j},(s_{i+1,t-1}-s_{i,t-1})/L_{i})`

        * :math:`x` is :py:attr:`~xintegration`.
        * :math:`s` corresponds to :py:attr:`~states`.
        * :math:`pos` is described in :py:attr:`~_xgauss`.
        * :math:`L` are the nodal distances as in :py:attr:`~nframe`.

        .. note::
            If both of the arguments are truthy, the argument which occurs in
            the method signature first has highest precedence. The internal
            forcing is only assigned to the :py:attr:`~forcing` attribute
            when called with default arguments.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility.fluxfunctions import richards_equation
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Strunk1979`
        and prepare the conductivity function with the soil parameters.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)

        Add states for a stationary no flow situation and check the
        :py:attr:`~forcing` attribute for the internal forcing values.

        >>> FE = Flow1DFE("Internal forcing example")
        >>> FE.set_systemfluxfunction(richards_equation, kfun=kfun)
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> FE.set_initial_states([-1 * i for i in range(11)])
        >>> FE.set_gaussian_quadrature(3)
        >>> FE._internal_forcing()
        >>> FE.forcing
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> FE._internal_forcing(calcflux=True)
        >>> FE.fluxes
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        Both arrays should consist of zeros because of the applied
        equilibrium situation with no flow over the boundaries.

        References
        ----------
        .. bibliography:: bibliography.bib

        """
        # internal fluxes from previous iteration
        pos, weight = self.gaussquad
        f = [0.0 for x in range(len(self.nodes))]
        for i in range(len(self.nodes) - 1):
            L = self.nframe[i, 1]
            for idx in range(len(pos)):
                x = self.xintegration[i][idx]
                s = self.states[i] * pos[-idx-1] + self.states[i+1] * pos[idx]
                grad = (self.states[i+1] - self.states[i]) / L
                flux = self.systemfluxfunc(x, s, grad)

                if not calcflux:
                    f[i] -= flux * weight[-idx-1]
                f[i+1] += flux * weight[idx]

        if calcflux:
            self.fluxes = np.array(f)
        else:
            # if balance calculation, don't assign to forcing again
            if not calcbal:
                self.forcing = self.forcing + np.array(f)
            self.internal_forcing["internal_forcing"] = [None, np.array(f)]

    def _statedep_forcing(self):
        # point state dependent forcing
        f = [0.0 for x in range(len(self.nodes))]
        for key in self.Spointflux.keys():
            (Sfunc, (idx_l, idx_r, lfac, rfac)), _ = self.Spointflux[key]
            # calculate state at position by linear interpolation
            dstates = self.states[idx_r] - self.states[idx_l]
            state = self.states[idx_l] + rfac * dstates
            # calculate function value and distribute fluxes accordingly
            value = Sfunc(state)
            f[idx_l] += value * lfac
            f[idx_r] += value * rfac
            self.Spointflux[key][1] = np.array(f)

        # spatial state dependent forcing
        pos, weight = self.gaussquad
        for key in self.Sspatflux.keys():
            f = [0.0 for x in range(len(self.nodes))]
            Sfunc = self.Sspatflux[key][0]
            for i in range(len(self.nodes) - 1):
                L = self.nframe[i, 1]
                for idx in range(len(pos)):
                    x = self.xintegration[i][idx]
                    ds = self.states[i+1] - self.states[i]
                    # linearly interpolate the state value
                    s = self.states[i] + (x - self.nodes[i]) * (ds / L)
                    # assign to nearby nodes according to scheme
                    f[i] += Sfunc(x, s) * weight[-idx-1] * pos[-idx-1] * L
                    f[i+1] += Sfunc(x, s) * weight[idx] * pos[idx] * L

            self.Sspatflux[key][1] = np.array(f)

        # aggregate state dependent forcing
        for flux in [self.Spointflux, self.Sspatflux]:
            for key in flux.keys():
                self.forcing += flux[key][1]

        # add boundaries of type dirichlet
        for key in self.BCs.keys():
            val, type_, idx = self.BCs[key]
            if type_ == "Dirichlet":
                self.forcing[idx] = 0

        # reshape for matrix solver
        self.forcing = np.reshape(self.forcing, (len(self.nodes), 1))

    def _check_boundaries(self):
        """ Check for proper boundary conditions

        The system is checked for singularity. When a boundary condition
        is not set explicitly a natural boundary condition (no flow) is
        set as default.

        Raises
        ------
        numpy.linalg.LinAlgError
            This error is raised if the system has infinitely many solutions
            as a consequence of two Neumann boundary conditions or when none
            are entered.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Check boundaries")
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.add_neumann_BC(-0.3, "west")
        >>> FE.BCs
        {'west': (-0.3, 'Neumann', 0)}
        >>> FE._check_boundaries()
        Traceback (most recent call last):
         ...
        numpy.linalg.LinAlgError: Singular matrix
        >>> FE.add_dirichlet_BC(-100, "west")
        >>> FE.BCs
        {'west': (-100, 'Dirichlet', 0)}
        >>> FE._check_boundaries()
        >>> FE.BCs
        {'west': (-100, 'Dirichlet', 0), 'east': (0, 'Neumann', -1)}

        """
        # no boundary conditions entered
        keys = list(self.BCs.keys())
        if len(keys) == 0:
            raise np.linalg.LinAlgError("Singular matrix")

        # if one boundary is not entered a zero Neumann boundary is set
        if len(keys) == 1:
            val, type_, pos = self.BCs[keys[0]]
            if type_ == "Dirichlet" and pos == 0:
                self.add_neumann_BC(value=0, where="east")
            elif type_ == "Dirichlet" and pos == -1:
                self.add_neumann_BC(value=0, where="west")
            else:
                raise np.linalg.LinAlgError("Singular matrix")

        # both boundaries cannot be of type Neumann
        if len(keys) == 2:
            val0, type_0, pos0 = self.BCs[keys[0]]
            val1, type_1, pos1 = self.BCs[keys[1]]
            if type_0 == type_1 and type_0 == "Neumann":
                raise np.linalg.LinAlgError("Singular matrix")

        # constrain the states to the applied boundary conditions
        for key in keys:
            val, type_, pos = self.BCs[key]
            if type_ == "Dirichlet":
                self.states[pos] = val

    def _calc_theta_k(self):
        if hasattr(self, 'kfun'):
            k = [self.kfun(n, s) for n, s in zip(self.nodes, self.states)]
            self.conductivities = np.array(k)
        if hasattr(self, 'tfun'):
            t = [self.tfun(s) for s in self.states]
            self.moisture = np.array(t)

    def _update_storage_change(self, prevstate, dt):
        ''' feed new previous states function and timestep '''
        storagechange = self.Sspatflux.get('storage_change', None)
        if storagechange:
            storagechange[0] = partial(storagechange[0],
                                       prevstate=self.states_to_function(),
                                       dt=dt)

    def _solve_initial_object(self):
        if self.isinitial:
            self._check_boundaries()
            self.forcing = np.repeat(0, len(self.states))
            self._internal_forcing()
            self._update_storage_change(self.states_to_function(), dt=1)

    def _FE_precalc(self):
        nodes = self.nodes
        middle = []
        nodal_distances = []

        # calculate positions of midpoints and nodal distances
        for i in range(0, len(nodes) - 1):
            middle.append((nodes[i] + nodes[i+1]) / 2)
            nodal_distances.append(nodes[i+1] - nodes[i])

        # calculate the length of each 1d finite element
        length = []
        length.append(middle[0] - nodes[0])
        for i in range(1, len(nodes) - 1):
            length.append(middle[i] - middle[i-1])
        length.append(nodes[-1] - middle[-1])

        # assign to class attributes
        self.nframe = np.array(list((zip(middle, nodal_distances))))
        self.lengths = np.array(length)

    def _CMAT(self, nodes, states):
        systemflux = self.systemfluxfunc
        A = np.zeros((len(nodes), len(nodes)))
        # internal flux
        for i in range(len(nodes) - 1):
            # fixed values for selected element
            L = nodes[i+1] - nodes[i]
            stateleft = states[i] + np.array([0, self._delta, 0])
            stateright = states[i+1] + np.array([0, 0, self._delta])
            grad = (stateright - stateleft) / L

            totflux = np.array([0, 0, 0], dtype=np.float64)
            pos, weight = self.gaussquad
            Svall = 0
            Svalr = 0
            # calculates the selected integral approximation
            for idx in range(len(pos)):
                x = self.xintegration[i][idx]
                state_x = stateleft * pos[-idx-1] + stateright * pos[idx]

                flux = [systemflux(x, s, g) for s, g in zip(state_x, grad)]
                totflux += np.array(flux) * weight[idx]

                # calculates gradients for spatial state dependent fluxes
                for key in self.Sspatflux.keys():
                    Sfunc = self.Sspatflux[key][0]
                    Sval = [Sfunc(x, s) for s in state_x]
                    dsl = (Sval[1] - Sval[0]) / self._delta
                    dsr = (Sval[2] - Sval[0]) / self._delta
                    Svall += pos[-idx-1] * weight[-idx-1] * L * dsl
                    Svalr += pos[idx] * weight[idx] * L * dsr

            dfluxl = (totflux[1] - totflux[0]) / self._delta
            dfluxr = (totflux[2] - totflux[0]) / self._delta

            # assign flux values to the coefficient matrix
            A[i][i] += -dfluxl + Svall
            A[i+1][i] += dfluxl + Svall
            A[i][i+1] += -dfluxr + Svalr
            A[i+1][i+1] += dfluxr + Svalr

        # state dependent point flux
        for key in self.Spointflux.keys():
            (Sfunc, (idx_l, idx_r, lfac, rfac)), _ = self.Spointflux[key]
            # calculate state at position by linear interpolation
            dstates = self.states[idx_r] - self.states[idx_l]
            state = self.states[idx_l] + rfac * dstates
            Ai = Sfunc(state)
            Aiw = Sfunc(state + self._delta)
            Aiiw = (Aiw - Ai) / self._delta
            A[idx_l][idx_l] += Aiiw * lfac
            A[idx_r][idx_r] += Aiiw * rfac

        self.coefmatr = A

    def wrap_bf_linear(self, node, where):
        n = self.nodes

        def basis_function(x):
            if where == "left":
                return (n[node + 1] - x) / (n[node + 1] - n[node])
            elif where == "right":
                return (x - n[node]) / (n[node + 1] - n[node])
        return basis_function

    def set_field1d(self, nodes, degree=1):
        """ Initialize the system's discretization

        :py:attr:`~states` and :py:attr:`~forcing` are initialized with zeros.
        The Gaussian quadrature :py:attr:`~gauss_degree` is set and the
        system's discretization characteristics are calculated with
        :py:meth:`~_FE_precalc`.

        Parameters
        ----------
        nodes : `tuple` or `list` or `numpy.ndarray`
            A tuple of the form (start, end, number of nodes) for a linearly
            spaced domain or a sequence of nodes that contains the nodal
            positions explicitly.

        degree : `int`, default is 1
            Set the Gaussian quadrature degree, this is equivalent to the
            :py:meth:`~set_gaussian_quadrature` method.

        Notes
        -----

        .. warning::
            Make sure that the absolute positions of the nodes increase
            towards the right of the domain.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility.spacing import biasedspacing

        Linear nodal spacing with a non-default Gaussian degree.

        >>> FE = Flow1DFE("Several spacings")
        >>> FE.set_field1d((-10, 0, 11), degree=3)
        >>> FE.nodes
        array([-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.])
        >>> FE._xgauss
        (0.1127016653792583, 0.5, 0.8872983346207417)

        Unstructured nodes using the
        :py:func:`~waterflow.utility.spacing.biasedspacing` function.

        >>> unstructured_nodes = biasedspacing(numnodes=11, power=4, lb=-1, rb=0, maxdist=2, length=10)
        >>> FE.set_field1d(unstructured_nodes)
        >>> FE.nodes
        array([-10.        ,  -9.61405656,  -9.29864794,  -8.94730706,
                -8.54868046,  -8.0844499 ,  -7.12803197,  -6.        ,
                -4.        ,  -2.        ,   0.        ])

        """
        if isinstance(nodes, tuple):
            self.nodes = np.linspace(*nodes)
        else:
            self.nodes = np.array(nodes)

        self.states = np.repeat(0.0, len(self.nodes))
        self.forcing = np.repeat(0.0, len(self.nodes))
        self.set_gaussian_quadrature(degree=degree)
        self._FE_precalc()

    def set_systemfluxfunction(self, function, **kwargs):
        """ Implement the governing flow equation

        The :py:attr:`~systemfluxfunc` is set with the governing flow
        equation.

        Parameters
        ----------
        function : `func`
            Flow equation that takes position, state and gradient as its
            arguments respectively.
        **kwargs : `keyword arguments`
            Extra arguments for the flow equation which are implemented
            as defaults so that the calling signature of the flow equation
            remains the same.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import fluxfunctions as fluxf
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility.helper import initializer

        Implement the Richards equation for unsturated flow, herein the
        Van Genuchten conductivity function is used. :cite:`Strunk1979`.
        Soil 13, 'loam', from De Staringreeks :cite:`Strunk1979` is selected.
        See :py:func:`~waterflow.utility.fluxfunctions.richards_equation` for
        the full definition of the fluxfunction.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)
        >>> richards = fluxf.richards_equation
        >>> FErichard = Flow1DFE("Flow equations")
        >>> FErichard.set_systemfluxfunction(richards, kfun=kfun)

        For saturated flow the Darcy equation with a constant saturated
        conductivity can be used. :cite:`Strunk1979`. See
        :py:func:`~waterflow.utility.fluxfunctions.darcy` for the full
        definition of the fluxfunction.

        >>> darcy = fluxf.darcy
        >>> FEdarcy = Flow1DFE("Flow equations")
        >>> FEdarcy.set_systemfluxfunction(darcy, ksat=s.ksat)

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        def fluxfunction(x, s, gradient):
            return function(x, s, gradient, **kwargs)
        self.systemfluxfunc = fluxfunction

    def set_initial_states(self, states):
        """ Set the initial states

        Although the main purpose of this method is to set the initial states
        it can be used to manipulate the states at any given point in time. The
        states are written to :py:attr:`~states`.

        Parameters
        ----------
        states : `int` or `float` or `list` or `numpy.ndarray`
            Set the states to an uniform value or vary the states with a
            sequence like argument.

        Notes
        -----

        .. note::
            Note that the states can only be set when the discretization of
            the system is known.

        Examples
        --------

        Set the initial states of the system or use the default setting.

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Setting states")
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.states
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> FE.set_initial_states([-1 * i for i in range(11)])
        >>> FE.states
        array([  0.,  -1.,  -2.,  -3.,  -4.,  -5.,  -6.,  -7.,  -8.,  -9., -10.])

        """
        if isinstance(states, int) or isinstance(states, float):
            states = float(states)
            self.states = np.array([states for x in range(len(self.nodes))])
        else:
            self.states = np.array(states, dtype=np.float64)

    def add_dirichlet_BC(self, value, where):
        """ Set boundary condition with fixed state

        The Dirichlet boundary condition is implemented with this
        method. The boundary condition is saved in :py:attr:`~BCs`.

        Parameters
        ----------
        value : `int` or `float`
            State value of the specific boundary
        where : `str`
            Position where the boundary condition will be set. Choose from
            "west", "left", "down", "east", "right" or "up". This argument
            is case insensitive.

        Notes
        -----
        Describe how the boundary condition is implemented. !!!!!!!!!!!!!

        Examples
        --------
        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Dirichlet boundary conditions")
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.BCs
        {}
        >>> FE.add_dirichlet_BC(-100, "Up")
        >>> FE.add_dirichlet_BC(0, "Down")
        >>> FE.BCs
        {'east': (-100, 'Dirichlet', -1), 'west': (0, 'Dirichlet', 0)}

        .. note::
            Note that a new boundary condition will overwrite an existing
            one without a warning.

        """
        if isinstance(value, int) or isinstance(value, float):
            value = [value]
            where = [where]

        for val, pos in zip(value, where):
            if pos.lower() in ["west", "left", "down"]:
                self.BCs["west"] = (val, "Dirichlet", 0)
                self._west = 1
            elif pos.lower() in ["east", "right", "up"]:
                self.BCs["east"] = (val, "Dirichlet", -1)
                self._east = -1

    def add_neumann_BC(self, value, where):
        """ Set boundary condition with fixed flux

        The Neumann boundary condition is implemented with this
        method. The boundary condition is saved in :py:attr:`~BCs`.

        Parameters
        ----------
        value : `int` or `float`
            Flux value of the specific boundary
        where : `str`
            Position where the boundary condition will be set. Choose from
            "west", "left", "down", "east", "right" or "up". This argument
            is case insensitive.

        Notes
        -----
        Describe how the boundary condition is implemented. !!!!!!!!!!!!!
        also note that FE.set_field1d needs to be called for the _east attr

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Neumann boundary conditions")
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.BCs
        {}
        >>> FE.add_neumann_BC(-0.1, "right")
        >>> FE.BCs
        {'east': (-0.1, 'Neumann', -1)}
        >>> FE.add_neumann_BC(-0.5, "up")
        >>> FE.BCs
        {'east': (-0.5, 'Neumann', -1)}
        >>> FE.add_neumann_BC(-0.8, "West")
        >>> FE.BCs
        {'east': (-0.5, 'Neumann', -1), 'west': (-0.8, 'Neumann', 0)}

        .. note::
            Note that a new boundary condition will overwrite an existing
            one without a warning. This method will allow both boundaries
            to be of type Neumann but remember that this won't be useful
            because of the infinite amount of solutions in such a situation.

        """
        if isinstance(value, int) or isinstance(value, float):
            value = [value]
            where = [where]

        for val, pos in zip(value, where):
            if pos.lower() in ["west", "left", "down"]:
                self.BCs["west"] = (val, "Neumann", 0)
                self._west = 0
            elif pos.lower() in ["east", "right", "up"]:
                self.BCs["east"] = (val, "Neumann", -1)
                self._east = len(self.nodes)

    def remove_BC(self, *args):
        """ Remove boundary conditions

        Calling this method with default arguments will clear all boundary
        conditions set. To clear a specific boundary condition the name
        needs to be passed explicitly.

        Parameters
        ----------
        *args : `str`, optional.
            The positional arguments should contain the name of the boundary
            conditions as saved in :py:attr:`~BCs`. This can be "west" or
            "east".

        Raises
        -----
        KeyError
            This exception is raised when ``*args`` contains an invalid
            boundary condition name.

        .. note::
            This is the safe way to remove the boundary conditions because it
            will also handle and reset the :py:attr:`~_west` and
            :py:attr:`~_east` attributes which are associated with the
            implementation of the boundary conditions in the numerical scheme.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Boundary condition removal")
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.add_dirichlet_BC(-100, "up")
        >>> FE.add_neumann_BC(0.0, "down")
        >>> FE.BCs
        {'east': (-100, 'Dirichlet', -1), 'west': (0.0, 'Neumann', 0)}
        >>> FE.remove_BC("right")
        Traceback (most recent call last):
         ...
        KeyError: 'No boundary named right.'
        >>> FE.remove_BC("west")
        >>> FE.BCs
        {'east': (-100, 'Dirichlet', -1)}
        >>> FE.remove_BC()
        >>> FE.BCs
        {}

        """
        if len(args) == 0:
            self.BCs = {}
            self._west = None
            self._east = None
        else:
            for name in args:
                try:
                    self.BCs.pop(name)
                    if name == "west":
                        self._west = None
                    else:
                        self._east = None
                except KeyError as e:
                    raise type(e)("No boundary named " + str(name) + ".")

    def add_pointflux(self, rate, pos, name=None):
        """ Add a pointflux to the system


        Parameters
        ----------

        Notes
        -----
        Describe the calculation of :math:`rfac` and :math:`lfac` here.

        Examples
        --------


        """
        f = [x*0.0 for x in range(len(self.nodes))]
        if isinstance(rate, int) or isinstance(rate, float):
            rate = [rate]
            pos = [pos]

        # add single valued pointflux to corresponding control volume
        if isinstance(rate, list):
            for r, p in zip(rate, pos):
                # find indices left and right of pointflux
                idx_r = np.searchsorted(self.nodes, p)
                idx_l = idx_r - 1
                # calculate contribution of pointflux to neighbouring nodes
                nodedist = self.nodes[idx_r] - self.nodes[idx_l]
                lfactor = 1 - (p - self.nodes[idx_l]) / nodedist
                rfactor = 1 - lfactor
                # assign to the forcing vector
                f[idx_l] += r * lfactor
                f[idx_r] += r * rfactor

            if name:
                self.pointflux[name] = [np.array(f)]
            else:
                name = str(Time.time())
                self.pointflux[name] = [np.array(f)]

        # add state dependent pointflux
        elif callable(rate):
            # find indices left and right of pointflux
            idx_r = np.searchsorted(self.nodes, pos)
            idx_l = idx_r - 1
            # calculate contribution of pointflux to neighbouring nodes
            nodedist = self.nodes[idx_r] - self.nodes[idx_l]
            lfactor = 1 - (pos - self.nodes[idx_l]) / nodedist
            rfactor = 1 - lfactor

            if not name:
                name = rate.__name__

            # create data structure
            self.Spointflux[name] = [(rate, (idx_l, idx_r, lfactor, rfactor)),
                                     np.array(f)]

    def add_spatialflux(self, q, name=None, exact=False):
        if isinstance(q, int) or isinstance(q, float):
            Q = np.repeat(q, len(self.nodes)).astype(np.float64)
        elif isinstance(q, list) or isinstance(q, np.ndarray):
            Q = np.array(q).astype(np.float64)
        elif callable(q):
            Q = q

        if not callable(Q):
            f = Q * self.lengths
            if name:
                self.spatflux[name] = [np.array(f)]
            else:
                # unique key for unnamed forcing
                name = str(Time.time())
                self.spatflux[name] = [np.array(f)]

        else:
            # prepare a domain spaced and zero-ed array
            f = [x*0.0 for x in range(len(self.nodes))]

            # check callable's number of positional arguments
            fparams = signature(Q).parameters
            nargs = sum(i == str(j) for i, j in fparams.items())

            if not name:
                name = Q.__name__

            # if function has one argument > f(x)
            if nargs == 1:
                # linear appoximation integral
                if not exact:
                    pos, weight = self.gaussquad
                    for i in range(len(self.nodes) - 1):
                        # distance between nodes
                        L = self.nframe[i, 1]
                        for idx in range(len(pos)):
                            x = self.xintegration[i][idx]
                            # to left node (no pos negatives?)
                            f[i] += Q(x) * weight[-idx-1] * pos[-idx-1] * L
                            # to right node
                            f[i+1] += Q(x) * weight[idx] * pos[idx] * L

                # only possible if analytical solution exists
                else:
                    # exact integral
                    nodes = self.nodes
                    for i in range(len(self.nodes) - 1):
                        l = self.wrap_bf_linear(i, "left")
                        r = self.wrap_bf_linear(i, "right")

                        def to_left(x):
                            return l(x) * Q(x)

                        def to_right(x):
                            return r(x) * Q(x)

                        # to left node
                        f[i] += quad(to_left, nodes[i], nodes[i+1])[0]
                        # to right node
                        f[i+1] += quad(to_right, nodes[i], nodes[i+1])[0]

                self.spatflux[name] = [np.array(f)]

            # if function has two arguments > f(x,s)
            elif nargs == 2:
                self.Sspatflux[name] = [Q, np.array(f)]

            elif nargs == 3:
                # implement time dependence ??
                pass

            # only valid for the storage change function
            # f(x, s, prevstate, dt)
            elif nargs == 4:
                name = 'storage_change'
                self.Sspatflux[name] = [Q, np.array(f)]

    def remove_pointflux(self, *args):
        # remove all pointfluxes when args is empty
        if len(args) == 0:
            self.pointflux = {}
            self.Spointflux = {}
        else:
            # remove the given pointflux name(s)
            for name in args:
                try:
                    self.pointflux.pop(name)
                except KeyError:
                    try:
                        self.Spointflux.pop(name)
                    except KeyError as e:
                        raise type(e)(str(name) + "is not a pointflux.")

    def remove_spatialflux(self, *args):
        # remove all spatial fluxes when args is empty
        if len(args) == 0:
            self.spatflux = {}
            self.Sspatflux = {}
        else:
            # remove the given spatial flux name(s)
            for name in args:
                try:
                    self.spatflux.pop(name)
                except KeyError:
                    try:
                        self.Sspatflux.pop(name)
                    except KeyError as e:
                        raise type(e)(str(name) + " is not a spatialflux.")

    def states_to_function(self):
        """ Prepare one-dimensional interpolation function

        One-dimensional piecewise linearly interpolated function which returns
        the system's states and is continiously defined on the domain
        :py:attr:`~nodes`.

        Returns
        -------
        `functools.partial`
            Function that calculates the system's states for a given position.

        Notes
        -----
        The main purpose of this method is to allow for Gaussian quadrature
        calculations which require state values at specific positions between
        the system's :py:attr:`~nodes`. This method also provides a tool for
        plotting at arbitrary positions within the domain.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Continiously defined states")
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.set_initial_states([-1 * i for i in range(11)])
        >>> continious_states = FE.states_to_function()
        >>> # On a boundary node
        >>> continious_states(0)
        -10.0
        >>> # In between two nodes
        >>> continious_states(-8.5)
        -1.5
        >>> # multiple results at once
        >>> continious_states([-0.2 * i for i in range(6)])
        array([-10. ,  -9.8,  -9.6,  -9.4,  -9.2,  -9. ])

        .. warning::
            Be aware that the function does not raise an exception but returns
            the value of the nearest boundary when a position outside of the
            domain is given as argument.

        """
        states = self.states.copy()
        # check if west boundary is of type Dirichlet
        if self._west == 1:
            states[0] = self.BCs["west"][0]
        # check is east boundary is of type Dirichlet
        if self._east == -1:
            states[-1] = self.BCs["east"][0]
        # linearly interpolate between states including assigned boundaries
        else:
            return partial(np.interp, xp=self.nodes, fp=states)

    def dt_solve(self, dt, maxiter=500, threshold=1e-3):
        """ solve the system for one specific time step """
        self._check_boundaries()
        west = self._west
        east = self._east

        # if system is transient
        self._update_storage_change(self.states_to_function(), dt)

        itercount = 1
        while itercount <= maxiter:
            self._aggregate_forcing()
            self._internal_forcing()
            self._statedep_forcing()
            self._CMAT(self.nodes, self.states)

            solution = np.linalg.solve(self.coefmatr[west:east, west:east],
                                       -1*self.forcing[west:east]).flatten()

            prevstates = self.states
            curstates = np.copy(prevstates)
            curstates[west:east] += solution
            self.states = curstates

            # update forcing and boundaries at new states
            self._aggregate_forcing()
            self._internal_forcing()
            self._statedep_forcing()

            # check solution for conversion
            max_abs_change = max(abs(prevstates - curstates))
            max_abs_allowed_change = max(abs(threshold * curstates))
            if max_abs_change < max_abs_allowed_change:
                break

            itercount += 1
        else:
            return itercount
        self.isinitial = False
        return itercount

    def solve(self, dt=0.001, dt_min=1e-5, dt_max=0.5, end_time=1, maxiter=500,
              dtitlow=1.5, dtithigh=0.5, itermin=5, itermax=10, threshold=1e-3,
              verbosity=True):
        """ solve the system for a given period of time """

        solved_objs = [deepcopy(self)]
        time_data = [0]
        dt_data = [None]
        iter_data = [None]

        time = dt

        t0 = Time.clock()
        while time <= end_time:
            # solve for given dt
            iters = self.dt_solve(dt, maxiter, threshold)

            # catch cases where maxiter is reached
            if iters > maxiter:
                if dt == dt_min:
                    print(f'Maxiter {iters} at dt_min {dt_min} reached')
                    self.isconverged = False
                    break
                else:
                    print(f'Maxiter {iters} reached, dt {dt} is lowered...')
                    dt *= dtithigh
                    if dt < dt_min:
                        dt = dt_min
                    # revert back to previous model state
                    self.states = solved_objs[-1].states
                    continue

            self.isconverged = True

            if verbosity:
                if self.Sspatflux.get('storage_change', None):
                    fmt = 'Converged at time={} for dt={} with {} iterations'
                    print(fmt.format(time, dt, iters))
                else:
                    print('Converged at {} iterations'.format(iters))

            # break out of loop when system is stationary
            if not self.Sspatflux.get('storage_change', None):
                solved_objs.append(deepcopy(self))
                time_data.append(time)
                dt_data.append(dt)
                iter_data.append(iters)
                break

            # build data record
            solved_objs.append(deepcopy(self))
            time_data.append(time)
            dt_data.append(dt)
            iter_data.append(iters)

            # adapt dt as function of iterations of previous step
            if iters <= itermin:
                dt *= dtitlow
                if dt > dt_max:
                    dt = dt_max
            elif iters >= itermax:
                dt *= dtithigh
                if dt < dt_min:
                    dt = dt_min

            # last time step calculated should be end_time exactly
            remaining_time = end_time - time
            if remaining_time == 0:
                pass
            elif remaining_time < dt:
                dt = remaining_time

            # increment time
            time += dt

        self.runtime = Time.clock() - t0

        # attach all solve data to last created object
        solve_data = {}
        solve_data['solved_objects'] = solved_objs
        solve_data['time'] = time_data
        solve_data['dt'] = dt_data
        solve_data['iter'] = iter_data
        self.solve_data = solve_data

    def calcbalance(self, print_=False, invert=True):
        """ Calculate the water balance for the system

        The water balance values are calculated at every position
        in the domain and will be saved to :py:attr:`~df_balance`.
        The summary of the water balance, which is the sum of all
        relevant columns, is saved in :py:attr:`~df_balance_summary`.

        Parameters
        ----------
        print_ : `bool`, default is ``False``
            Print :py:attr:`~df_balance_summary` to the console.
        invert : `bool`, default is ``True``
            Mirror :py:attr:`~df_balance` w.r.t. the x-axis.

        Notes
        -----
        The external fluxes, :math:`F_{external}`, are the sum of all point
        and spatial fluxes.

        .. centered::
                :math:`F_{external} = \\sum_{j=1}^{n} F_{p_j} +
                \\sum_{j=1}^{m} F_{s_j}`

        The calculation of the point flux, :math:`F_{p_j}`, depends on its
        nature. If the point flux depends on position only, the accumulation is
        straightforward. When the point flux is dependend on the state of the
        the system, the distribution towards the surrounding nodes needs to be
        calculated before values can be accumulated to its total. The
        distribution to the nearest nodes is calculated as follows:

        .. math::
            node_{i} = F_{p_j}(s) * rfac

        .. math::
            node_{i+1} = F_{p_j}(s) * lfac

        where :math:`s` equals the state at the position of the point flux
        which is calculated by linear interpolation. The calculation of the
        fractions :math:`rfac` and :math:`lfac` are described in
        :py:meth:`~add_pointflux`.

        For the calculation of the spatial flux, :math:`F_{s_j}`, a similar
        distinction exists. If the spatial flux is not dependend on state,
        straightforward addition takes place. When there is a state
        dependency, distributions towards the nodes is calculated as
        described in :py:meth:`~_internal_forcing` where
        :math:`F_{s_j}(x, s)` is substituted for the :py:attr:`~systemfluxfunc`
        :math:`Q(x, s, grad)`. This calculation accounts for the selected
        :py:attr:`~gauss_degree` and the state argument :math:`s` is linearly
        interpolated between the neareast nodes.

        The internal forcing in the water balance is the sum of the internal
        forcing as described in :py:meth:`~_internal_forcing` and the external
        forcing.

        .. math::
            F_{total\_internal} = F_{internal} + F_{external}

        The top and bottom values in :math:`F_{total\_internal}` are corrected
        for the flow over the boundaries. The flow over the boundaries is
        calculated as the difference of :math:`F_{total\_internal}` -
        :math:`\\Delta s` at those boundary nodes.

        The net flux is calculated as follows:

        .. math::
            F_{net} = F_{total\_internal} - \\Delta s

        Where :math:`\\Delta s` represents the storage change between
        iterations at every node in the domain.

        .. note::
            Although the storage change :math:`\\Delta s` is entered into the
            model as an external flux, in the calculation of the water balance
            this term is handled as a separate flux which is not included in
            the :math:`F_{external}` term.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility import fluxfunctions as fluxf
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Strunk1979`
        and prepare the conductivity function and theta-h relation with the
        soil parameters. These functions are the arguments to the fluxfunction
        and the storage change function repectively.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> theta_h = initializer(condf.VG_pressureh, theta_r=s.t_res,
        ...                       theta_s=s.t_sat, a=s.alpha, n=s.n)
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)
        >>> storage_change = initializer(fluxf.storage_change, fun=theta_h)

        >>> FE = Flow1DFE("Calculate water balance")
        >>> FE.set_systemfluxfunction(fluxf.richards_equation, kfun=kfun)
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> FE.add_dirichlet_BC(0.0, 'west')
        >>> # Constant boundary flow of 0.3 cm/d out of the system
        >>> FE.add_neumann_BC(-0.3, 'east')
        >>> # theta_h function needs to be added manually to be included in the water balance
        >>> FE.tfun = theta_h
        >>> # Extraction of 0.001 cm/d over the complete domain
        >>> FE.add_spatialflux(-0.001, 'extraction')
        >>> # Add storage change function
        >>> FE.add_spatialflux(storage_change)
        >>> # Solve the system for one time step (dt=0.01 d)
        >>> iters = FE.dt_solve(dt=0.01)
        >>> FE.calcbalance()
        >>> FE.df_balance
            nodes  spat-extraction  storage_change  internal  all-spatial  all-points  all-external           net    fluxes
        0     0.0          -0.0005        0.144730 -0.144730      -0.0005         0.0       -0.0005  0.000000e+00  0.155770
        1    -1.0          -0.0010        0.266555 -0.266555      -0.0010         0.0       -0.0010 -4.663325e-12 -0.109784
        2    -2.0          -0.0010        0.222706 -0.222706      -0.0010         0.0       -0.0010 -2.797818e-12 -0.331490
        3    -3.0          -0.0010        0.182558 -0.182558      -0.0010         0.0       -0.0010 -6.783463e-14 -0.513048
        4    -4.0          -0.0010        0.145754 -0.145754      -0.0010         0.0       -0.0010  1.635247e-12 -0.657801
        5    -5.0          -0.0010        0.112096 -0.112096      -0.0010         0.0       -0.0010  1.979528e-12 -0.768897
        6    -6.0          -0.0010        0.081552 -0.081552      -0.0010         0.0       -0.0010  1.637579e-12 -0.849449
        7    -7.0          -0.0010        0.054287 -0.054287      -0.0010         0.0       -0.0010  1.292744e-12 -0.902736
        8    -8.0          -0.0010        0.030765 -0.030765      -0.0010         0.0       -0.0010  1.127098e-12 -0.932501
        9    -9.0          -0.0010        0.012078 -0.012078      -0.0010         0.0       -0.0010  1.573075e-12 -0.943580
        10  -10.0          -0.0005        0.002068 -0.002068      -0.0005         0.0       -0.0005  0.000000e+00 -0.945148
        >>> FE.df_balance_summary
        spat-extraction   -1.000000e-02
        storage_change     1.255148e+00
        internal          -1.255148e+00
        all-spatial       -1.000000e-02
        all-points         0.000000e+00
        all-external      -1.000000e-02
        net                1.716294e-12
        dtype: float64

        References
        ----------
        .. bibliography:: bibliography.bib

        """
        data = {}
        # internal fluxes
        self._internal_forcing(calcbal=True)
        self._internal_forcing(calcflux=True)
        internalfluxes = self.internal_forcing['internal_forcing'][-1]

        # add all point fluxes
        pnt = np.repeat(0.0, len(self.nodes))
        pointfluxes = {**self.pointflux, **self.Spointflux}
        for key in pointfluxes.keys():
            # if state dependent, calculate new forcing for new states
            if key in self.Spointflux.keys():
                f = [0.0 for x in range(len(self.nodes))]
                (Sfunc, (idx_l, idx_r, lfac, rfac)), _ = self.Spointflux[key]
                # calculate state at position by linear interpolation
                dstates = self.states[idx_r] - self.states[idx_l]
                state = self.states[idx_l] + rfac * dstates
                # calculate function value and distribute fluxes accordingly
                value = Sfunc(state)
                f[idx_l] += value * lfac
                f[idx_r] += value * rfac
                pnt += np.array(f)
            # if constant/position dependent then no need for a new calculation
            else:
                pnt += pointfluxes[key][-1]
            data.update({"pnt-"+str(key): pointfluxes[key][-1]})

        # add all spatial fluxes
        spat = np.repeat(0.0, len(self.nodes))
        spatialfluxes = {**self.spatflux, **self.Sspatflux}
        for key in spatialfluxes.keys():
            # if state dependent, calculate new forcing for new states
            if key in self.Sspatflux.keys():
                pos, weight = self.gaussquad
                f = [0.0 for x in range(len(self.nodes))]
                Sfunc = self.Sspatflux[key][0]
                for i in range(len(self.nodes) - 1):
                    L = self.nframe[i, 1]
                    for idx in range(len(pos)):
                        x = self.xintegration[i][idx]
                        ds = self.states[i+1] - self.states[i]
                        # linearly interpolate the state value
                        s = self.states[i] + (x - self.nodes[i]) * (ds / L)
                        # assign to nearby nodes according to scheme
                        f[i] += Sfunc(x, s) * weight[-idx-1] * pos[-idx-1] * L
                        f[i+1] += Sfunc(x, s) * weight[idx] * pos[idx] * L
                spat += np.array(f)
            # if constant/position dependent then no need for a new calculation
            else:
                spat += spatialfluxes[key][-1]
            data.update({"spat-"+str(key): spatialfluxes[key][-1]})

        # storage between iterations, if exists
        try:
            storage_change = data.pop("spat-storage_change")
        except KeyError:
            storage_change = np.repeat(0.0, len(self.nodes))

        # remove storage change from external spatial forcings
        spat = spat - storage_change

        # internal balance
        internalfluxes = internalfluxes + pnt + spat

        # boundaries
        lbound = internalfluxes[0] + storage_change[0]
        rbound = internalfluxes[-1] + storage_change[-1]

        # correct for boundary fluxes
        internalfluxes[0] -= lbound
        internalfluxes[-1] -= rbound
        self.fluxes[0] -= lbound

        # net flow
        net = internalfluxes + storage_change

        # dump waterbalance & summary to dataframe
        data.update({'storage_change': storage_change,
                     'internal': internalfluxes, 'all-spatial': spat,
                     'all-points': pnt, 'all-external': pnt + spat,
                     'net': net, 'fluxes': self.fluxes})

        df_balance = pd.DataFrame(data)
        df_balance.insert(0, 'nodes', self.nodes)

        if invert:
            df_balance = df_balance.iloc[::-1].reset_index(drop=True)

        self.df_balance = df_balance
        df_balance_summary = df_balance.sum().transpose().drop(['nodes', 'fluxes'])
        self.df_balance_summary = df_balance_summary

        if print_:
            print(self.df_balance_summary)

    def dataframeify(self, invert):
        """ Write current static model to dataframe

        Save the current model results to :py:attr:`~df_states`.

        Parameters
        ----------
        invert : `bool`
            Mirror :py:attr:`~df_states` w.r.t. the x-axis.

        Notes
        -----
        At least, lengths, nodes, states and the internal forcing are written
        to the dataframe.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility.fluxfunctions import richards_equation
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Strunk1979`
        and prepare the conductivity function with the soil parameters.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)

        Add states for a stationary no flow situation and check the
        :py:attr:`~forcing` attribute for the internal forcing values.

        >>> FE = Flow1DFE("Internal forcing example")
        >>> FE.set_systemfluxfunction(richards_equation, kfun=kfun)
        >>> FE.set_field1d(nodes=(-10, 0, 11))

        References
        ----------
        .. bibliography:: bibliography.bib

        """
        self._calc_theta_k()
        self._solve_initial_object()

        columns = ['lengths', 'nodes', 'states', 'moisture',
                   'conductivities', 'pointflux', 'Spointflux',
                   'spatflux', 'Sspatflux', 'internal_forcing']
        data = {}
        for idx, name in enumerate(columns):
            if 0 <= idx < 3:
                data[name] = self.__dict__.get(name)
            elif 3 <= idx < 5:
                values = self.__dict__.get(name)
                if list(values):
                    data[name] = values
            else:
                for k, v in self.__dict__.get(name).items():
                    data[k] = v[-1]

        df = pd.DataFrame(data)
        if invert:
            df = df.iloc[::-1].reset_index(drop=True)
        self.df_states = df

    def transient_dataframeify(self, print_times=None, include_maxima=True,
                               nodes=None, invert=True):
        """ Bundle all static dataframes to a single data structure



        Parameters
        ----------
        print_times : `int`, `list` or `numpy.ndarray`
            pass
        include_maxima : `bool`
            pass
        nodes : `list` or `numpy.ndarray`
            pass
        invert : `bool`
            Mirror the built dataframes w.r.t. the x-axis.

        Notes
        -----

        Examples
        --------


        """
        # times at which the model has been solved
        self.dft_solved_times = pd.DataFrame(data=self.solve_data)

        # solve model at specific print times
        if print_times:
            st = self.solve_data['time']
            solved_obj = self.solve_data['solved_objects']

            # print_times can be a sequence or a scalar
            if isinstance(print_times, (list, np.ndarray)):
                pt = np.array(print_times)
                pt = np.delete(pt, np.argwhere(pt < min(st)))
                pt = np.delete(pt, np.argwhere(pt > max(st)))
                if include_maxima:
                    pt = np.insert(pt, 0, min(st))
                    pt = np.append(pt, max(st))
            else:
                pt = np.linspace(min(st), max(st), print_times)
            pt.sort()
            pt = np.unique(pt)

            # Find solved model that is nearest in terms of time
            # solve the model for the new time from here
            new_obj = []
            for t, i in zip(pt, np.searchsorted(st, pt)):
                # print time already equals known state
                if t == st[i]:
                    new_obj.append(deepcopy(solved_obj[i]))
                # calculate model state at new time
                else:
                    dts = t - st[i - 1]
                    obj = deepcopy(solved_obj[i - 1])
                    obj.dt_solve(dts)
                    new_obj.append(obj)

            # dump model states at print times to dataframe
            data = {'solved_objects': new_obj, 'time': pt}
            dft_print_times = pd.DataFrame(data=data)
            self.dft_print_times = dft_print_times

        # if no specific print times remove dataframe
        else:
            self.dft_print_times = None

        # if available, use print times instead of solved_times
        if self.dft_print_times is not None:
            timedf = self.dft_print_times
        else:
            timedf = self.dft_solved_times

        # build all dataframes
        dft_states = {}
        dft_balance = {}
        dft_nodes = {}
        dft_bal_sum = pd.DataFrame()
        for row in timedf.itertuples(index=False):
            obj, t = row.solved_objects, row.time

            # states dataframe
            obj.dataframeify(invert=invert)
            dft_states.update({t: obj.df_states})

            # balance dataframes
            obj.calcbalance(invert=invert)
            dft_balance.update({t: obj.df_balance})
            dft_bal_sum = dft_bal_sum.append(obj.df_balance_summary,
                                             ignore_index=True)

            # track specific nodes (if present)
            if nodes:
                if not dft_nodes:
                    dft_nodes = dict((k, np.array([])) for k in nodes)
                df = obj.df_states
                c = df.columns
                for node in nodes:
                    nrow = df[df['nodes'] == node].to_numpy()
                    if len(dft_nodes[node]):
                        dft_nodes[node] = np.vstack((dft_nodes[node], nrow))
                    else:
                        dft_nodes[node] = nrow

        # nodes dataframe (if present)
        if nodes:
            d = {}
            for k, v in dft_nodes.items():
                d[k] = pd.DataFrame(v, columns=c)
                d[k].insert(0, timedf.time.name, timedf.time)
            self.dft_nodes = d

        self.dft_states = dft_states
        self.dft_balance = dft_balance
        dft_bal_sum.insert(0, timedf.time.name, timedf.time)
        self.dft_balance_summary = dft_bal_sum

    def save(self, savepath=None, dirname=None):
        """ Save model metadata and dataframes to disk

        Parameters
        ----------
        savepath: :obj:`str`, default is `OUTDIR`
            A base path to which runs will be saved.
        dirname : :obj:`str`, default is a chronological name
            Name of save directory that is appended to savepath.

        Notes
        -----

        Examples
        --------

        """
        savepath = savepath or self.savepath
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

        # save directory
        if not dirname:
            # chronological name
            dirname = Time.strftime('%d%b%Y_%H%M%S', Time.gmtime(Time.time()))
            runpath = os.path.join(savepath, dirname)
            os.mkdir(runpath)
        else:
            # given name
            runpath = os.path.join(savepath, dirname)
            if not os.path.isdir(runpath):
                os.mkdir(runpath)

        for k, v in self.__dict__.items():
            # transient dataframes only
            if k.startswith('dft_'):
                fname = f"{k}.xlsx"
                save_file = os.path.join(runpath, fname)

                if isinstance(v, pd.core.frame.DataFrame):
                    with pd.ExcelWriter(save_file) as fw:
                        v.to_excel(fw, sheet_name=k)

                elif isinstance(v, dict):
                    with pd.ExcelWriter(save_file) as fw:
                        for k, df in v.items():
                            df.to_excel(fw, sheet_name=str(k))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
