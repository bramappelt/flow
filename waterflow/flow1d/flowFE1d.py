""" One dimensional finite elements flow module """


from inspect import signature
from functools import partial
from copy import deepcopy
import time as Time
import os

import numpy as np
import pandas as pd
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

    savepath : `str`, default is :py:data:`~waterflow.OUTPUT_DIR`
        Directory to which model runs will be saved.

    Attributes
    ----------
    id_ : `str`
        Name of the model object.

    savepath: `str`, default is :py:data:`~waterflow.OUTPUT_DIR`
        Directory to which model runs will be saved.

    systemfluxfunc : `function`
        Holds the selected flux function.

    nodes : `numpy.ndarray`
        Nodal positions at which the system will be solved. Matrix of dimension
        :math:`[1 \\times N]`.

    states : `numpy.ndarray`
        State solutions at the nodal positions as defined in :py:attr:`~nodes`.
        Matrix of dimension :math:`[1 \\times N]`.

    seg_lengths : `numpy.ndarray`
        Lengths of segments between the nodes in the shape
        :math:`[1 \\times N - 1]`.

    lengths : `numpy.ndarray`
        The same data as in :py:attr:`~seg_lengths` but in a different
        representation. This representation has the shape
        :math:`[1 \\times N]`. This is more convenient for static
        state independent spatial forcing calculations.

    coefmatr : `numpy.ndarray`
        Square jacobian matrix used in the finite elements solution procedure.
        The exact dimension of the matrix is :math:`N \\times N`.

    BCs : `dict`
        This contains the system's boundary conditions. The keys that indicate
        the positions are "west" and "east". The corresponding values have the
        following format:

        * (boundary_condition_value, type, domain_index).

    pointflux : `dict`
        Contains the scalar point fluxes on the model domain. The key : value
        pairs in the dictionary have the following format:

            'Flux name' : [:math:`F_{local}` of shape :math:`[1 \\times N]`]

    Spointflux : `dict`
        Contains state dependent point flux functions on the model domain. The
        key : value pairs in the dictionary have the following format:

            'Flux name' : [(:math:`P`, (:math:`x_{l}`, :math:`x_{r}`, :math:`lfac`, :math:`rfac`)), :math:`F_{local}` of shape :math:`[1 \\times N]`]

    spatflux : `dict`
        Contains the spatial fluxes on the model domain. Both the scalar and
        the calculated position dependent spatial flux function values. The
        key : value pairs in the dictionary have the following format:

            name : [:math:`F_{local}` of shape :math:`[1 \\times N]`]

    Sspatflux : `dict`
        Contains state dependent spatial fluxes on the model domain. The key :
        value pairs in the dictionary have the following format:

            name : [:math:`S`, :math:`F_{local}` of shape :math:`[1 \\times N]`]

    internal_forcing : `dict`
        The internal forcing of the system as calculated with
        :py:attr:`~_internal_forcing`, using the selected Gaussian quadrature
        :py:attr:`~gauss_degree`.

    forcing : `numpy.ndarray`
        All the forcing fluxes applied to the system including the storage
        change forcing. This is the matrix that will be used for the
        Newton-Raphson solving procedure. The dimension of this matrix is
        :math:`[1 \\times N]`.

    conductivities : `numpy.ndarray`
        Hydraulic conductivities at the nodal positions, :py:attr:`~nodes`.
        These values are calculated with the conductivity function as
        given in :py:attr:`~systemfluxfunc`. Matrix of dimension
        :math:`[1 \\times N]`.

    moisture : `numpy.ndarray`
        Moisture contents at the nodal positions, :py:attr:`~nodes`. These
        values are calculated with a function that should be assigned to the
        model explicitly, having :py:attr:`~tfun` as attribute name. Matrix of
        dimension :math:`[1 \\times N]`.

    fluxes : `numpy.ndarray`
        Fluxes through the :py:attr:`~nodes`, defined to be positive to the
        right. Matrix of dimension :math:`[1 \\times N]`.

    isinitial : `bool`, default is True
        First object that contains initial input which has not been solved for
        yet. This attribute is set to ``False`` when the model object has been
        solved for.

    isconverged : `bool`, default is False
        The system has converged to a solution.

    solve_data : `dict`
        Holds the solve information of the system including the following
        key : value pairs:

        * solved_objects - A `list` of Flow1DFE objects at solved time steps.

        * time - A `list` of times at which the model states are calculated.

        * dt - A `list` of time step sizes between consecutive model solutions.

        * | iter - A `list` containing the number of iterations needed for
          | consecutive model solutions to converge.

    runtime : `float`
        The total time (s) it takes for :py:meth:`~solve` to find a solution.

    df_states : `pandas.core.frame.DataFrame`
        Current information about the static model solution.

    df_balance : `pandas.core.frame.DataFrame`
        Current static information about the water balance.

    df_balance_summary : `pandas.core.frame.DataFrame`
        Sum of the (relevant) columns as saved in :py:attr:`~df_balance`.

    dft_solved_times : `pandas.core.frame.DataFrame`
        Dataframe version of :py:attr:`~solve_data`.

    dft_print_times : `pandas.core.frame.DataFrame`
        A version of :py:attr:`~dft_solved_times` but at specifically
        chosen print times. This attribute is calculated with
        :py:meth:`~transient_dataframeify` giving it a
        value for ``print_times``, otherwise the default
        :py:attr:`~dft_solved_times` will be used.

    dft_states : `dict`
        Collection of all :py:attr:`~df_states` dataframes for the times in
        :py:attr:`~dft_solved_times` or at :py:attr:`~dft_print_times` if
        specific print times were given.

    dft_nodes : `dict`
        Nodes that are selected in :py:meth:`~transient_dataframeify` are
        saved at :py:attr:`~dft_solved_times` or at :py:attr:`~dft_print_times`
        if specific print times were given.

    dft_balance : `dict`
        Collection of all :py:attr:`~df_balance` dataframes at
        :py:attr:`~dft_solved_times` or at :py:attr:`~dft_print_times` if
        specific print times were given.

    dft_balance_summary : `pandas.core.frame.DataFrame`
        Collection of all :py:attr:`~df_balance_summary` dataframes at
        :py:attr:`~dft_solved_times` or at :py:attr:`~dft_print_times` if
        specific print times were given.

    _west : `int`
        Internal value that differentiates between a Dirichlet or Neumann
        boundary condition on the western side of the domain.

    _east : `int`
        Internal value that differentiates between a Dirichlet or Neumann
        boundary condition on the eastern side of the domain.

    _delta : `float`
        Fixed value used for the finite displacement in the derivatives of the
        jacobian matrix, :py:attr:`~coefmatr`. This value may be changed
        manually for extremeley steep gradients.

    gauss_degree : `int`
        Degree or number of points used in the Gaussian quadrature procedure
        for integral approximation.

    _xgauss : `tuple`
        Roots of Legendre polynomial on the interval :math:`[0, 1]` for the
        selected :py:attr:`~gauss_degree`.

    _wgauss : `tuple`
        Weights that correspond to the positions in :py:attr:`~_xgauss`.

    gaussquad : `list`
        Combination of corresponding values of :py:attr:`~_xgauss` and
        :py:attr:`~wgauss` into a single data structure.

    xintegration : `list`
        Absolute positions of the Gaussian quadrature points in the domain
        :py:attr:`~nodes`. The shape of this list is
        :math:`[\\Lambda \\times N-1]`.

    summarystring : `str`
        Model information obtained by :py:meth:`~summary`.

    """
    def __init__(self, id_, savepath=OUTPUT_DIR):
        self.id_ = id_
        self.savepath = savepath
        self.systemfluxfunc = None
        self.nodes = None
        self.states = None
        self.seg_lenghts = None
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
        self.isinitial = True
        self.isconverged = False
        self.solve_data = None
        self.runtime = None
        self.summarystring = ""
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
        # Gauss specific
        self.gauss_degree = 1
        self._xgauss = None
        self._wgauss = None
        self.gaussquad = None
        self.xintegration = None

    def __repr__(self):
        """ Representation of the object as shown to the user """
        return "Flow1DFE(" + str(self.id_) + ")"

    def summary(self, show=True, save=False, path=None):
        """ Description of the object

        Subsequent calls will update the model description if
        adaptations to the model were made.

        Parameters
        ----------
        show : `bool`, default is True
            Print object description to the console.
        save : `bool`, default is False
            Save object description to disk.
        path : `str`, default is None
            Full path of a directory to which will be saved. This
            argument is mandatory if ``save=True``.

        Notes
        -----
        The description of the model is saved as :py:attr:`~id_` with a`.txt`
        extension. The string version of the description is also available via
        :py:attr:`~summarystring`.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility import fluxfunctions as fluxf
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Wosten2001`
        and prepare the conductivity function and theta-h relation with the
        soil parameters. These functions are the arguments to the fluxfunction
        and the storage change function respectively.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> theta_h = initializer(condf.VG_pressureh, theta_r=s.t_res,
        ...                       theta_s=s.t_sat, a=s.alpha, n=s.n)
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)
        >>> storage_change = initializer(fluxf.storage_change, fun=theta_h)

        >>> FE = Flow1DFE("static df_states dataframe")
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> FE.set_systemfluxfunction(fluxf.richards_equation, kfun=kfun)
        >>> FE.add_dirichlet_BC(0.0, 'west')
        >>> # Constant boundary flow of 0.3 cm/d out of the system
        >>> FE.add_neumann_BC(-0.3, 'east')
        >>> # theta_h add manually to be included in the dataframe
        >>> FE.tfun = theta_h
        >>> # add spatial flux
        >>> FE.add_spatialflux(-0.001, 'extraction')
        >>> # Add storage change function
        >>> FE.add_spatialflux(storage_change)
        >>> # Solve the system for one time step (dt=0.01 d)
        >>> iters = FE.dt_solve(dt=0.01)
        >>> FE.summary()
        Id: static df_states dataframe
        System length: 10.0
        Number of nodes: 11
        Gauss degree: 1
        kfun: VG_conductivity
        tfun: VG_pressureh
        BCs: west value: 0.0 and of type Dirichlet, east value: -0.3 and of type Neumann
        Spatflux: extraction, storage_change
        <BLANKLINE>
        spat-extraction   -1.000000e-02
        storage_change     1.255148e+00
        internal          -1.255148e+00
        all-spatial       -1.000000e-02
        all-points         0.000000e+00
        all-external      -1.000000e-02
        net                1.716294e-12

        """
        # build key : value pairs where data is available
        id_ = f"{self.id_}"
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

        if hasattr(self, 'kfun'):
            kfun = self.kfun.__name__
        else:
            kfun = None

        if hasattr(self, 'tfun'):
            tfun = self.tfun.__name__
        else:
            tfun = None

        k = ['Id', 'System length', 'Number of nodes', 'Gauss degree',
             'kfun', 'tfun', 'BCs', 'Pointflux', 'Spatflux', 'Runtime (s)']
        v = (id_, len_, num_nodes, degree, kfun, tfun, bcs, pointflux,
             spatflux, runtime)

        # build summary string
        sumstring = ""
        for i, j in zip(k, v):
            if j:
                sumstring += f"{i}: {j}\n"

        try:
            self.calcbalance()
            sumstring += '\n' + self.df_balance_summary.to_string()
        except Exception:
            pass

        # print to console
        if show:
            for s in sumstring.split('\n'):
                print(s)

        # save to disk
        if save:
            if not os.path.isdir(path):
                os.mkdir(path)

            fname = f"{self.id_}.txt"
            with open(os.path.join(path, fname), "w") as fw:
                fw.write(sumstring)

        self.summarystring = sumstring

    def set_gaussian_quadrature(self, degree=1):
        """ Calculates Gaussian quadrature roots and weights

        The values calculated with this method are stored in the object's
        :py:attr:`~_xgauss` and :py:attr:`~_wgauss` attributes. The absolute
        positions of the Gaussian quadrature points in the domain are
        calculated and saved in :py:attr:`~xintegration`.

        Parameters
        ----------
        degree : `int`, default is 1
            Number of points used in the Gaussian quadrature procedure.

        Notes
        -----
        The integration points :math:`p_{\\lambda}` for the Gaussian
        quadrature method are obtained by finding the roots of the Legendre
        polynomial of degree :math:`\\Lambda`.

        .. math::
            P_{\\Lambda}(p) = \\text{Legendre polynomials of degree $\\Lambda$}

        The corresponding weights :math:`w_{\\lambda}` are calculated with the
        following closed form equation.

        .. math::
            w_{\\lambda} = \\frac{2}{\\left(\\left(1 - p_{\\lambda}^{2}\\right) *
            P_{\\Lambda}^{'}(p_{\\lambda})^{2}\\right)}

        A full description of the theory behind this Gaussain quadrature method
        is documented in :cite:`Abramowitz1972`.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Gaussian quadrature")
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.set_gaussian_quadrature(2)
        >>> # positions
        >>> FE._xgauss
        (0.21132486540518708, 0.7886751345948129)
        >>> # weights
        >>> FE._wgauss
        (0.4999999999999999, 0.5000000000000002)
        >>> # check shape of xintegration
        >>> np.array(FE.xintegration).shape
        (10, 2)

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
        """ Aggregation of state independent forcing

        The state independent forcing saved in :py:attr:`~pointflux` and
        :py:attr:`~spatflux` is accumulated into one matrix :math:`F_{forcing}`
        and is saved as :py:attr:`~forcing` in the object. In the case of a
        Neumann boundary condition, the value of this flux is added to
        either the left or the right side of the domain.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Calculate water balance")
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> FE.add_dirichlet_BC(0.0, 'west')
        >>> # Constant boundary flow of 0.3 cm/d out of the system
        >>> FE.add_neumann_BC(-0.3, 'east')
        >>> # Add spatial extraction of -0.001 cm/d
        >>> FE.add_spatialflux(-0.001, 'extraction')
        >>> # add a point extraction of -0.05 cm/d at 5.5 cm depth
        >>> FE.add_pointflux(-0.05, -5.5, 'sink')
        >>> # Aggregate state independent forcing and account for Neumann BC
        >>> FE._aggregate_forcing()
        >>> FE.forcing
        array([-0.0005, -0.001 , -0.001 , -0.001 , -0.026 , -0.026 , -0.001 ,
               -0.001 , -0.001 , -0.001 , -0.3005])

        """
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
        accumulate the values to :math:`F_{forcing}` which is saved in the
        model as :py:attr:`~forcing`.

        Parameters
        ----------
        calcflux : `bool`, default is False
            If ``True``, fluxes through the nodes are calculated and
            saved in :py:attr:`~fluxes`.
        calcbal : `bool`, default = False
            If ``True``, internal forcing is saved in
            :py:attr:`~internal_forcing` instead of :py:attr:`~forcing`.

        Notes
        -----

        This mathematical description of the internal forcing
        calculation describes how the forcing at the nodes is calculated.
        The direction of flow is defined to be positive to the right.
        The equation below describes how the forcing at a specific node can
        be calculated, taking the Gaussian quadrature procedure into account.

        .. math::
            F_{i} = \\sum_{\\lambda=1}^{\\Lambda} Q(X_{i,\\lambda}, s_{i,\\lambda}+\\delta x, grad_{i}) * w_{\\lambda}

        Argument :math:`X_{i, \\lambda}` represents the absolute position
        of the Gaussian quadrature point as saved in :py:attr:`~xintegration`.
        The state argument :math:`s_{i,\\lambda}` at this specific Gaussian
        quadrature point is calculated as follows:

        .. math::
            s_{i,\\lambda} = s_{i} * (1-p_{\\lambda}) + s_{i+1} * p_{\\lambda}

        The third argument, :math:`grad_{i}`, is the gradient of the state
        between the nodes of the current segment and is calculated as
        shown below:

        .. math::
            grad_{i} = \\frac{s_{i+1} - s{i}}{L_{i}}

        All internal fluxes are collected in :math:`F_{internal}`.

        .. math::
            F_{internal} = \\begin{bmatrix}
                                -F_{i}              \\\\
                                F_{i} - F_{i+1}    \\\\
                                F_{i+1} - F_{i+2}  \\\\
                                \\vdots            \\\\
                                F_{N-1} - F_{N}    \\\\
                                F_{N}
                           \\end{bmatrix}

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

        Select soil 13, 'loam', from De Staringreeks :cite:`Wosten2001`
        and prepare the conductivity function with the soil parameters.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)

        Add states for a stationary no flow situation and check the
        :py:attr:`~forcing` attribute for the internal forcing values.

        >>> FE = Flow1DFE("Internal forcing example")
        >>> FE.set_systemfluxfunction(richards_equation, kfun=kfun)
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> FE.set_initial_states([-i for i in range(11)])
        >>> FE.set_gaussian_quadrature(3)
        >>> FE._internal_forcing()
        >>> FE.forcing
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> FE._internal_forcing(calcflux=True)
        >>> FE.fluxes
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        Both arrays should consist of zeros because of the applied
        equilibrium situation with no flow over the boundaries.

        """
        # internal fluxes from previous iteration
        pos, weight = self.gaussquad
        f = [0.0 for x in range(len(self.nodes))]
        for i in range(len(self.nodes) - 1):
            L = self.seg_lengths[i]
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
        """ Calculation of state dependent forcing values

        Values for the state dependent forcing functions saved in
        :py:attr:`~Sspatflux` and :py:attr:`~Spointflux` are calculated.
        The local forcing matrices :math:`F_{local}` are populated and the
        total of both is accumulated to the global forcing matrix
        :math:`F_{forcing}`. Forcing values at the position of a Dirichlet
        boundary condition are set to zero. This method is called internally
        by :py:meth:`~dt_solve` to prepare for the next iteration.

        Notes
        -----
        Calculations of the individual spatial fluxes :math:`j` are saved in
        the local forcing matrix :math:`F_{local}`. The following two
        components need to be calculated over the complete domain of the
        system for the specific spatialflux. The Gaussian quadrature degree
        :math:`\\Lambda` is accounted for.

        .. math::
            F_{l_{i}} = \\sum_{\\lambda=1}^{\\Lambda} S(X_{i, \\lambda}, s_{i,\\lambda})_{j} * (1-p_{\\lambda}) * w_{\\lambda} * L_{i}

        .. math::
            F_{r_{i}} = \\sum_{\\lambda=1}^{\\Lambda} S(X_{i, \\lambda}, s_{i,\\lambda})_{j} * p_{\\lambda} * w_{\\lambda} * L_{i}

        where the state argument :math:`s_{i,j}` is calculated by linear
        interpolation as shown below:

        .. math::
            s_{i,\\lambda} = s_{i} + (X_{i,\\lambda} - x_{i}) * \\frac{s_{i+1} - s_{i}}{L_{i}}

        Calculations of the individual point fluxes :math:`k` are also saved
        in its local forcing matrix :math:`F_{local}`. The following two
        components need to be calculated at the position of the specific
        pointflux:

        .. math::
            F_{l_{i}} = P(s_{i,k})_{k} * lfac_{k}

        .. math::
            F_{r_{i}} = P(s_{i,k})_{k} * rfac_{k}

        where the state argument :math:`s_{i,k}` is calculated by linear
        interpolation as shown below:

        .. math::
            s_{i,k} = s_{i} + rfac_{k} * (s_{i+1} - s_{i})

        The local forcing matrix is populated as follows, this scheme
        is used for both the pointflux and the spatialflux calculation.

        .. math::
            F_{local} = \\begin{bmatrix}
                          F_{l_{i}}                  \\\\
                          F_{r_{i}} + F_{l_{i+1}}    \\\\
                          F_{r_{i+1}} + F_{l_{i+2}}  \\\\
                          \\vdots                    \\\\
                          F_{r_{N-1}} + F_{l_{N}}  \\\\
                          F_{r_{N}}
                        \\end{bmatrix}

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("state dependent fluxes")
        >>> FE.set_field1d((-10, 0, 11))
        >>> # Set initial states other than all zeros
        >>> FE.set_initial_states([-i for i in range(11)])

        >>> # Define a state dependent pointflux function
        >>> def Spflux(s):
        ...     return abs(np.sin(s)) * -0.1
        >>> FE.add_pointflux(Spflux, -3.1)

        >>> # Define a state dependent spatialflux function
        >>> def linear_s_extraction(x, s):
        ...     return -0.001 * abs(x) - 0.001 * s
        >>> FE.add_spatialflux(linear_s_extraction)

        >>> # Calculcate state dependent forcing (for next iteration)
        >>> FE._statedep_forcing()
        >>> # Local forcing matrix of Spflux
        >>> FE.Spointflux['Spflux'][1]
        array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                0.        , -0.0057844 , -0.05205958,  0.        ,  0.        ,
                0.        ])
        >>> # Local forcing matrix of 'linear_s_extraction'
        >>> FE.Sspatflux['linear_s_extraction'][1]
        array([-0.0045, -0.008 , -0.006 , -0.004 , -0.002 ,  0.    ,  0.002 ,
                0.004 ,  0.006 ,  0.008 ,  0.0045])

        """
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
                L = self.seg_lengths[i]
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
        is not explicitly set, a natural boundary condition (no flow) is
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
        """ Conductivities and moisture contents

        Calculation of conductivities and moisture contents in the system.
        The results are saved in :py:attr:`~conductivities` and
        :py:attr:`~moisture` and will be included in :py:attr:`~df_states`.

        Notes
        -----
        The conductivity function is part of the :py:attr:`~systemfluxfunc`.
        The moisture content function should be assigned to the model
        explicitly. See the example below.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility import fluxfunctions as fluxf
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Wosten2001`
        and prepare the conductivity function and theta-h relation with the
        soil parameters. These functions are the arguments to the fluxfunction
        and the storage change function repectively.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> theta_h = initializer(condf.VG_pressureh, theta_r=s.t_res,
        ...                       theta_s=s.t_sat, a=s.alpha, n=s.n)
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)

        >>> FE = Flow1DFE("Calculate water balance")
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> # The conductivity function is added as argument to the systemfluxfunction
        >>> FE.set_systemfluxfunction(fluxf.richards_equation, kfun=kfun)
        >>> FE._calc_theta_k()
        >>> FE.conductivities
        array([12.98, 12.98, 12.98, 12.98, 12.98, 12.98, 12.98, 12.98, 12.98,
               12.98, 12.98])
        >>> # Omitting a function will skip the calculation (e.g. in case of saturated flow)
        >>> FE.moisture
        []
        >>> # Create an equilibrium situation
        >>> FE.set_initial_states([-i for i in range(10)])
        >>> # Now add the moisture content function explicitly
        >>> FE.tfun = theta_h
        >>> # Calculate again
        >>> FE._calc_theta_k()
        >>> FE.conductivities
        array([12.98      , 10.01657823,  9.05018386,  8.36428965,  7.8189874 ,
                7.36164639,  6.96580186,  6.61596642,  6.30216943,  6.01755319])
        >>> FE.moisture
        array([0.42      , 0.41987202, 0.41965291, 0.41937832, 0.41906052,
               0.41870661, 0.41832137, 0.41790836, 0.41747038, 0.4170097 ])

        """
        if hasattr(self, 'kfun'):
            k = [self.kfun(n, s) for n, s in zip(self.nodes, self.states)]
            self.conductivities = np.array(k)
        if hasattr(self, 'tfun'):
            t = [self.tfun(s) for s in self.states]
            self.moisture = np.array(t)

    def _update_storage_change(self, prevstate, dt):
        """ Update states function and time step size of storage change function

        The storage change function is implemented as a spatial state
        dependent flux function. Therefore, it needs to match its mandatory
        function signature as described in :py:meth:`~add_spatialflux`. This
        method updates the ``prevstate`` and ``dt`` arguments by changing these
        default arguments of the storage change function so that the calling
        signature remains the same.

        Parameters
        ----------
        prevstate : `func`
            Function that calculates the system's states :math:`s` for a given
            position :math:`x`.
        dt : `float` or `int`
            Time step over which the storage change will be calculated.

        Notes
        -----
        The storage change function, if present, is stored in
        :py:attr:`~Sspatflux`. The calling signature may look as follows:

        ..  math::
            storage\\textunderscore change(x, s, prevstate, dt, fun=lambda\\text{ }x: 1, S=1)

        The :math:`prevstate` and :math:`dt` argument need to be updated for
        every new time step and are set as default values so the storage
        change function can be called with the signature demanded by
        :py:meth:`~add_spatialflux`. See
        :py:func:`~waterflow.utility.fluxfunctions.storage_change` for the
        complete definition of the storage change function.

        """
        storagechange = self.Sspatflux.get('storage_change', None)
        if storagechange:
            storagechange[0] = partial(storagechange[0],
                                       prevstate=self.states_to_function(),
                                       dt=dt)

    def _solve_initial_object(self):
        """ Calculation on first model input

        Calculations on the input data for the first object, used as
        first entry in any of the model's dataframes.

        Notes
        -----
        If :py:attr:`~isinitial` equals ``True`` the boundaries of the model
        are checked, the internal forcing is calculated and the storage change
        function is updated, if present.

        .. note::
            The values calculated by this method and included in the model's
            dataframes as first entries not need to make any sense because the
            user can set any unrealistic combination of initial input.

        """
        if self.isinitial:
            self._check_boundaries()
            self.forcing = np.repeat(0, len(self.states))
            self._internal_forcing()
            self._update_storage_change(self.states_to_function(), dt=1)

    def _FE_precalc(self):
        """ Discretization lengths

        Calculate the values for :py:attr:`~seg_lengths` and
        :py:attr:`~lengths`.

        Notes
        -----
        The lengths of the segments between the nodes are calculated as
        follows. The number of segments is always one less than the number of
        nodes in the system.

        .. math::
            L_{i} = x_{i+1} - x_{i} \\text{ for } i=1,2,\\dotsc,N-1

        To assign a length to every node in the system, a different approach
        has been used. Except for the boundary cases, the differences of the
        midpoints between the nodes has been taken as a segment length. See the
        exact definition:

        .. math::
            nL_{i} =
                \\begin{cases}
                \\frac{x_{i}+x_{i+1}}{2} - x_{i}                  & \\text{ for } i=1 \\\\
                \\frac{x_{i+1}-x_{i-1}}{2}                        & \\text{ for } i=2,3,\\dotsc,N-1 \\\\
                x_{i}-\\frac{x_{i-1}+x_{i}}{2}                    & \\text{ for } i=N
                \\end{cases}

        Multiplication of :math:`nL_{i}` with scalar or sequence like
        spatial fluxes result in a forcing array that has the shape
        :math:`[1 \\times N]` which is convenient for direct addition to the
        global :py:attr:`~forcing` matrix.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Flow equations")
        >>> # FE_precalc() is called implicitly by set_field1d()
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.seg_lengths
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> FE.lengths
        array([0.5, 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0.5])

        """
        slen = np.repeat(0.0, len(self.nodes) - 1)
        nlen = np.repeat(0.0, len(self.nodes))

        # calculate both arrays in one loop & catch boundary cases
        for i in range(len(self.nodes)):
            if i == 0:
                nlen[i] = (self.nodes[i] + self.nodes[i+1]) / 2 - self.nodes[i]
            elif i == len(self.nodes) - 1:
                nlen[i] = self.nodes[i] - (self.nodes[i-1] + self.nodes[i]) / 2
            else:
                nlen[i] = (self.nodes[i+1] - self.nodes[i-1]) / 2

            if i != len(self.nodes) - 1:
                slen[i] = self.nodes[i+1] - self.nodes[i]

        # assign to class attributes
        self.seg_lengths = slen
        self.lengths = nlen

    def _CMAT(self, nodes, states):
        """ Build the jacobian matrix

        Build the complete jacobian matrix according to the finite elements
        scheme for the selected degree :math:`\\Lambda`. The jacobian matrix
        :math:`A` is assigned to :py:attr:`~coefmatr`.

        Parameters
        ----------
        nodes : `numpy.ndarray`
            Array that contains the nodal positions.
        states : `numpy.ndarray`
            Array that contains the states at the nodal positions.

        Notes
        -----
        The jacobian matrix :math:`A` is build from three individual parts.

        .. math::
            A = A_{sys} + A_{spat} + A_{point}

        * :math:`A_{sys}`, derivatives of the :py:attr:`~systemfluxfunc`, :math:`Q`.
        * :math:`A_{spat}`, derivatives of functions in :py:attr:`~Sspatflux`, :math:`S`.
        * :math:`A_{point}`, derivatives of functions in :py:attr:`~Spointflux`, :math:`P`.

        **System's flow equation jacobian**

        The derivatives of the systemflux function, :math:`Q`, are collected
        in :math:`A_{sys}`. The equation below describes how the
        derivatives are calculated in which the Gaussian quadrature scheme
        is accounted for.

        .. math::
            \\frac{\\delta Q}{\\delta x}_{i} = \\sum_{\\lambda}^{\\Lambda} \\frac{Q(X_{i,\\lambda}, s_{i,\\lambda}+\\delta x, grad_{i}) - Q(X_{i,\\lambda}, s_{i,\\lambda}, grad_{i})}{\\delta x} * w_{\\lambda}

        Argument :math:`X_{i, \\lambda}` represents the absolute position
        of the Gaussian quadrature point as saved in :py:attr:`~xintegration`.
        The state argument :math:`s_{i,\\lambda}` at this specific Gaussian
        quadrature point is calculated as follows:

        .. math::
            s_{i,\\lambda} = s_{i} * (1-p_{\\lambda}) + s_{i+1} * p_{\\lambda}

        The third argument, :math:`grad_{i}`, is the gradient of the state
        between the nodes of the current segment and is calculated as
        shown below:

        .. math::
            grad_{i} = \\frac{s_{i+1} - s_{i}}{L_{i}}

        :math:`A_{sys}` presents the structure of the sparse jacobian matrix
        in which the derivatives of the systemflux function :math:`Q` are
        saved.

        .. math::
            A_{sys} = \\begin{bmatrix}
                            -\\frac{\\delta Q}{\\delta x}_{i} & -\\frac{\\delta Q}{\\delta x}_{i} &  &  &  & \\\\
                            \\frac{\\delta Q}{\\delta x}_{i} & \\frac{\\delta Q}{\\delta x}_{i} -\\frac{\\delta Q}{\\delta x}_{i+1} & -\\frac{\\delta Q}{\\delta x}_{i+1} &  & & \\\\
                             & \\frac{\\delta Q}{\\delta x}_{i+1} & \\frac{\\delta Q}{\\delta x}_{i+1} -\\frac{\\delta Q}{\\delta x}_{i+2} & \\ddots & & \\\\
                             &  & \\ddots & \\ddots & & \\ddots & \\\\
                             &  &  & \\ddots &  & \\frac{\\delta Q}{\\delta x}_{N-1} -\\frac{\\delta Q}{\\delta x}_{N} & -\\frac{\\delta Q}{\\delta x}_{N} \\\\
                             &  &  &  &  & \\frac{\\delta Q}{\\delta x}_{N} & \\frac{\\delta Q}{\\delta x}_{N}
                    \\end{bmatrix}

        **Spatial state dependent jacobian**

        The derivatives of the state dependent spatial fluxes :math:`S`, if
        present, are collected in :math:`A_{spat}`. The calculation of these
        derivatives is described by the equations below. The values are summed
        at every segment in the model domain, taking into account the selected
        Gaussian quadrature scheme.

        .. math::
            \\sum \\frac{\\delta S_{l}}{\\delta x}_{i} = \\sum_{\\lambda=1}^{\\Lambda} \\sum_{j=1}^{n} \\frac{S(X_{i,\\lambda}, s_{i,\\lambda} + \\delta x)_{j} - S(X_{i,\\lambda}, s_{i,\\lambda})_{j}}{\\delta x} * L_{i} * (1-p_{\\lambda}) * w_{\\lambda}

        .. math::
            \\sum \\frac{\\delta S_{r}}{\\delta x}_{i} = \\sum_{\\lambda=1}^{\\Lambda} \\sum_{j=1}^{n} \\frac{S(X_{i,\\lambda}, s_{i,\\lambda} + \\delta x)_{j} - S(X_{i,\\lambda}, s_{i,\\lambda})_{j}}{\\delta x} * L_{i} * p_{\\lambda} * w_{\\lambda}

        Argument :math:`X_{i, \\lambda}` represents the absolute position
        of the Gaussian quadrature point as saved in :py:attr:`~xintegration`.
        The state argument :math:`s_{i,\\lambda}` at this specific Gaussian
        quadrature point is calculated as follows:

        .. math::
            s_{i,\\lambda} = s_{i} * (1-p_{\\lambda}) + s_{i+1} * p_{\\lambda}

        The structure of the sparse jacobian matrix, :math:`A_{spat}`, of the
        sum of all state dependent spatialflux function derivatives is shown
        below:

        .. math::
            A_{spat} = \\begin{bmatrix}
                            \\Sigma \\frac{\\delta S_{l}}{\\delta x}_{i} & \\Sigma \\frac{\\delta S_{r}}{\\delta x}_{i} &  &  &  & \\\\
                            \\Sigma \\frac{\\delta S_{l}}{\\delta x}_{i} & \\Sigma \\frac{\\delta S_{r}}{\\delta x}_{i} + \\Sigma \\frac{\\delta S_{l}}{\\delta x}_{i+1} & \\Sigma \\frac{\\delta S_{r}}{\\delta x}_{i+1} &  & & \\\\
                              & \\Sigma \\frac{\\delta S_{l}}{\\delta x}_{i+1} & \\Sigma \\frac{\\delta S_{r}}{\\delta x}_{i+1} + \\Sigma \\frac{\\delta S_{l}}{\\delta x}_{i+2} & \\ddots & & \\\\
                              &  & \\ddots & \\ddots & & \\ddots & \\\\
                              &  &  & \\ddots &  & \\Sigma \\frac{\\delta S_{r}}{\\delta x}_{N-1} + \\Sigma \\frac{\\delta S_{l}}{\\delta x}_{N} & \\Sigma \\frac{\\delta S_{r}}{\\delta x}_{N} \\\\
                              &  &  &  &  & \\Sigma \\frac{\\delta S_{l}}{\\delta x}_{N} & \\Sigma \\frac{\\delta S_{r}}{\\delta x}_{N}
                       \\end{bmatrix}

        **Point state dependent jacobian**

        The derivatives of the state dependent point fluxes, :math:`P`, if
        present, are collected in :math:`A_{point}`. The derivates are
        calculated for the specific functions, distributed to the two nearest
        nodes and summed to a total at this specific position. This is
        described by the equations below:

        .. math::
            \\sum \\frac{\\delta P_{l}}{\\delta x}_{i} = \\sum_{k=1}^{m} \\frac{P(s_{i,k} + \\delta x)_{k} + P(s_{i,k})_{k}}{\\delta x} * rfac_{k}

        .. math::
            \\sum \\frac{\\delta P_{r}}{\\delta x}_{i} = \\sum_{k=1}^{m} \\frac{P(s_{i,k} + \\delta x)_{k} + P(s_{i,k})_{k}}{\\delta x} * lfac_{k}

        The state argument, :math:`s_{i,k}`, is calculated by linear
        interpolation between the nearest nodes.

        .. math::
            s_{i,k} = s_{i} + rfac_{k} * (s_{i+1} - s_{i})

        The structure of the jacobian matrix, :math:`A_{point}`, of the sum of
        all state dependent pointflux function derivatives is shown below:

        .. math::
            A_{point} = \\begin{bmatrix}
                            \\Sigma \\frac{\\delta P_{l}}{\\delta x}_{i} &  &  &  &  & \\\\
                            & \\Sigma \\frac{\\delta P_{r}}{\\delta x}_{i} + \\Sigma \\frac{\\delta P_{l}}{\\delta x}_{i+1} &  &  &  &  \\\\
                            &  & \\Sigma \\frac{\\delta P_{r}}{\\delta x}_{i+1} + \\Sigma \\frac{\\delta P_{l}}{\\delta x}_{i+2} &  &  &  \\\\
                            &  &  & \\ddots &  &  & \\\\
                            &  &  &  &  & \\Sigma \\frac{\\delta P_{r}}{\\delta x}_{N-1} + \\Sigma \\frac{\\delta P_{l}}{\\delta x}_{N}  &  \\\\
                            &  &  &  &  &  & \\Sigma \\frac{\\delta P_{r}}{\\delta x}_{N}
                        \\end{bmatrix}

        """
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
                # position and state gaussian integration
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

                    # distribution of flux according to gaussian integration
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
            sfunc_s = Sfunc(state)
            sfunc_sd = Sfunc(state + self._delta)
            dfunc = (sfunc_sd - sfunc_s) / self._delta
            A[idx_l][idx_l] += dfunc * lfac
            A[idx_r][idx_r] += dfunc * rfac

        self.coefmatr = A

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
            Make sure that the positions of the ``nodes`` increase towards the
            right of the domain.

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
            Flow equation that takes position, state and gradient of state as
            its arguments respectively.
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
        Van Genuchten conductivity function is used, :cite:`VanGenuchten1980`.
        Soil 13, 'loam', from De Staringreeks :cite:`Wosten2001` is selected.
        See :py:func:`~waterflow.utility.fluxfunctions.richards_equation` for
        the full definition of the fluxfunction.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)
        >>> richards = fluxf.richards_equation
        >>> FErichard = Flow1DFE("Flow equations")
        >>> FErichard.set_systemfluxfunction(richards, kfun=kfun)

        For saturated flow the Darcy equation with a constant saturated
        conductivity can be used :cite:`Darcy1856`. See
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
        >>> FE.set_initial_states([-i for i in range(11)])
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
        In which attrs ?

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
        *args : `str`, optional
            The positional arguments should contain the name of the boundary
            conditions as saved in :py:attr:`~BCs`. This can be "west" or
            "east".

        Raises
        -----
        KeyError
            This exception is raised when ``*args`` contains an invalid
            boundary condition name which does not occur in the system.

        Notes
        -----

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

        Pointflux values and pointflux functions are accepted. Multiple
        point fluxes can be combined in a list. Scalar pointflux values
        are accumulated to its local matrix :math:`F` at the correct indices.
        The state dependent pointflux functions are prepared and saved for
        calculation in a context where :math:`s` is available.

        Parameters
        ----------
        rate : `float`, `int`, `list` or `func`
            *   Scalar pointflux value(s).
            *   Pointflux function of the form :math:`P(s)`.
        pos : `float`, `int` or `list`
            Position(s) of the pointflux value(s)/function.
        name : `str`, default is None
            Name of the pointflux. If omitted, a unique key is
            generated for a scalar pointflux or the string version
            of the pointflux function is used.

        Notes
        -----
        :math:`node_{r}` is the index of the right node that is nearest
        the position of the pointflux and is calculated by
        :py:meth:`~numpy.searchsorted`. :math:`node_{l} = node_{r} - 1`,
        which is the left node that is most near the position of the pointflux.

        :math:`lfac` and :math:`rfac` represent the fractions of the pointflux
        that contribute to the nearest left and right node repectively.

        .. math::
            lfac = 1 - (pos - x_{node_{l}}) / (x_{node_{r}} - x_{node_{l}})

        .. math::
            rfac = 1 - lfac

        In case of a scalar pointflux the distributed values are assigned to
        its local forcing array at the correct nodal positions and saved in
        :py:attr:`~pointflux`. See the formulas below:

        .. math::
            F_{node_{l}} = rate * lfac

        .. math::
            F_{node_{r}} = rate * rfac

        For the state dependent pointflux function the calculated values of
        :math:`node_{l}`, :math:`node_{r}`, :math:`lfac`, and :math:`rfac` are
        saved in :py:attr:`~Spointflux` in addition to the state dependent
        pointflux function itself and an empty local forcing array :math:`F`.

        .. warning::
            Multiple functions, unlike scalar point fluxes, should be
            implemented separately and cannot be combined in a list argument.

        Examples
        --------
        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Point fluxes")
        >>> FE.set_field1d((-10, 0, 11))

        >>> # Add a scalar pointflux
        >>> FE.add_pointflux(-0.003, -5.5, 'pflux')
        >>> FE.pointflux
        {'pflux': [array([ 0.    ,  0.    ,  0.    ,  0.    , -0.0015, -0.0015,  0.    ,
                0.    ,  0.    ,  0.    ,  0.    ])]}

        >>> # Define a state dependent point flux function
        >>> def Spflux(s):
        ...     return abs(np.sin(s)) * -0.1
        >>> FE.add_pointflux(Spflux, -3.1)
        >>> FE.Spointflux #doctest: +ELLIPSIS
        {'Spflux': [(<function Spflux at 0x...>, (6, 7, 0.10000000000000009, 0.8999999999999999)), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]}

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

    def add_spatialflux(self, q, name=None):
        """ Add a spatialflux to the system

        Spatial fluxes of several types are accepted by this method.
        Direct calculation of the forcing values is performed where
        possible. State dependent forcing functions will be
        calculated in a different context. The storage change function
        for the simulation of a transient system should be implemented
        here.

        Parameters
        ----------
        q : `int`, `float`, `list` or `func`
            * Scalar spatial flux value applied over the complete domain.
            * Sequence of different spatial flux values for every nodal
              position.
            * Positional and/or state dependent spatial flux function.
            * Storage change function for a transient simulation.
        name : `str`, default is None
            Name of the spatialflux. If omitted, a unique key is
            generated for a scalar/sequence like spatialflux or the string
            version of the spatialflux function is used.

        Notes
        -----
        When ``q`` is a scalar or sequence like argument the local forcing
        array :math:`F_{local}` is calculated by multiplication with the
        corresponding lengths and will be saved in :py:attr:`~spatflux`.

        .. math::
            F_{local} = q * nL

        If ``q`` is a function of position, :math:`S(x)`, the flux is
        calculated per segment. Below the exact definition of this calculation
        is presented, taking into account the Gaussian quadrature degree
        :math:`\\Lambda`.

        .. math::
            F_{l_{i}} = \\sum_{\\lambda =1}^{\\Lambda} S(X_{i, \\lambda}) * (1-p_{\\lambda}) * w_{\\lambda} * L_{i}

        .. math::
            F_{r_{i}} = \\sum_{\\lambda =1}^{\\Lambda} S(X_{i, \\lambda}) * p_{\\lambda} * w_{\\lambda} * L_{i}

        After the calculation of the distribution towards the nearest nodes the
        local forcing matrix :math:`F_{local}` is populated and will be saved
        in :py:attr:`~spatflux`.

        .. math::
            F_{local} = \\begin{bmatrix}
                            F_{l_{i}}                   \\\\
                            F_{r_{i}} + F_{l_{i+1}}     \\\\
                            F_{r_{i+1}} + F_{l_{i+2}}   \\\\
                            \\vdots                     \\\\
                            F_{r_{N-1}} + F_{l_{N}}     \\\\
                            F_{r_{N}}
                        \\end{bmatrix}

        In the case of ``q`` being a function of position and state,
        :math:`S(x, s)`, the function will be assigned to :py:attr:`~Sspatflux`
        for later processing in a context where :math:`s` is available.

        ``q`` can have four arguments, :math:`S(x, s, prevstate, dt)`. This is
        a special case reserved for the storage change function.
        The function signature may have keyword arguments but they need to
        be optional having a default value. See a possible definition of
        a storage change function,
        :py:func:`~waterflow.utility.fluxfunctions.storage_change`, that can
        be used for both saturated and unsaturated conditions depending on its
        default keyword arguments. The storage change function is saved in
        :py:attr:`~Sspatflux` and will carry the default name 'storage_change',
        see the last example.

        .. note::
            For spatial state dependent flux functions the function signature
            will always look like :math:`S(x, s)` whether there is a
            positional dependency or not. This is needed to distinguish between
            the :math:`x` and :math:`s` arguments.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility.fluxfunctions import storage_change
        >>> FE = Flow1DFE("spatial fluxes")
        >>> FE.set_field1d((-10, 0, 11))

        >>> # Add a scalar spatialflux
        >>> FE.add_spatialflux(-0.001, 'Root extraction')
        >>> FE.spatflux
        {'Root extraction': [array([-0.0005, -0.001 , -0.001 , -0.001 , -0.001 , -0.001 , -0.001 ,
               -0.001 , -0.001 , -0.001 , -0.0005])]}

        >>> # Add position dependent spatial flux function
        >>> def linear_extraction(x):
        ...     return -0.001 * abs(x)
        >>> FE.add_spatialflux(linear_extraction)
        >>> FE.spatflux
        {'Root extraction': [array([-0.0005, -0.001 , -0.001 , -0.001 , -0.001 , -0.001 , -0.001 ,
               -0.001 , -0.001 , -0.001 , -0.0005])], 'linear_extraction': [array([-0.00475, -0.009  , -0.008  , -0.007  , -0.006  , -0.005  ,
               -0.004  , -0.003  , -0.002  , -0.001  , -0.00025])]}

        >>> # Add position and state dependent spatial flux function
        >>> def linear_s_extraction(x, s):
        ...     return -0.001 * x - 0.001 * s
        >>> FE.add_spatialflux(linear_s_extraction, 'Sfunc')
        >>> # Note that the local forcing array will always be empty and is only initialized
        >>> FE.Sspatflux #doctest: +ELLIPSIS
        {'Sfunc': [<function linear_s_extraction at 0x...>, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]}

        >>> # The storage change function is imported and passed to the model
        >>> # note that the name argument is ignored.
        >>> FE.add_spatialflux(storage_change, name='My_storage_change')
        >>> FE.Sspatflux #doctest: +ELLIPSIS
        {'Sfunc': [<function linear_s_extraction at 0x...>, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])], 'storage_change': [<function storage_change at 0x...>, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]}

        """
        if isinstance(q, int) or isinstance(q, float):
            Q = np.repeat(q, len(self.nodes)).astype(np.float64)
        elif isinstance(q, list) or isinstance(q, np.ndarray):
            Q = np.array(q).astype(np.float64)
        elif callable(q):
            Q = q

        if not callable(Q):
            f = Q * self.lengths

            if not name:
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
                pos, weight = self.gaussquad
                for i in range(len(self.nodes) - 1):
                    # distance between nodes
                    L = self.seg_lengths[i]
                    for idx in range(len(pos)):
                        x = self.xintegration[i][idx]
                        # to left node (no pos negatives?)
                        f[i] += Q(x) * weight[-idx-1] * pos[-idx-1] * L
                        # to right node
                        f[i+1] += Q(x) * weight[idx] * pos[idx] * L
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
        """ Remove point fluxes

        Remove all or a specific point fluxes. When this method is called
        with default arguments all point fluxes will be removed from both
        :py:attr:`~pointflux` and :py:attr:`~Spointflux`.

        Parameters
        ----------
        *args : `str`, optional
            Name(s) of specific point fluxes.

        Raises
        ------
        KeyError
            Will be raised when the name of the pointflux does not exists.

        Examples
        --------
        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Point flux removal")
        >>> FE.set_field1d((-10, 0, 11))

        >>> # Add a scalar pointflux
        >>> FE.add_pointflux(-0.001, -3.3, 'Point1')
        >>> # Add an other
        >>> FE.add_pointflux(-0.002, -5.5, 'Point2')
        >>> # Add a third
        >>> FE.add_pointflux(-0.003, -7.7, 'Point3')
        >>> # Check for all the currently available pointflux names
        >>> FE.pointflux.keys()
        dict_keys(['Point1', 'Point2', 'Point3'])
        >>> def Spflux(s):
        ...     return abs(np.sin(s)) * -0.1
        >>> FE.add_pointflux(Spflux, -6.6)
        >>> FE.Spointflux.keys()
        dict_keys(['Spflux'])

        >>> # Specific removal of the point fluxes
        >>> FE.remove_pointflux('Point1', 'Point3')
        >>> FE.pointflux.keys()
        dict_keys(['Point2'])
        >>> # Use an incorrect name
        >>> FE.remove_pointflux("Point4") #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        KeyError: "'Point4' is not a pointflux."
        >>> # Remove the remaining point fluxes
        >>> FE.remove_pointflux()
        >>> FE.pointflux
        {}
        >>> FE.Spointflux
        {}

        """
        # remove all point fluxes
        if len(args) == 0:
            self.pointflux = {}
            self.Spointflux = {}
        else:
            # remove the specific pointflux name(s)
            for name in args:
                try:
                    self.pointflux.pop(name)
                except KeyError:
                    try:
                        self.Spointflux.pop(name)
                    except KeyError as e:
                        # Only raise exception at top of call stack
                        raise KeyError('{} is not a pointflux.'.format(e)) from None

    def remove_spatialflux(self, *args):
        """ Remove spatial fluxes

        Remove all or a specific spatial fluxes. When this method is called
        with default arguments all spatial fluxes will be removed from both
        :py:attr:`~spatflux` and :py:attr:`~Sspatflux`.

        Parameters
        ----------
        *args : `str`, optional
            Name(s) of specific spatial fluxes.

        Raises
        ------
        KeyError
            Will be raised when the name of the spatialflux does not exists.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility.fluxfunctions import storage_change
        >>> FE = Flow1DFE("Spatial flux removal")
        >>> FE.set_field1d((-10, 0, 11))

        >>> # Add a scalar spatialflux
        >>> FE.add_spatialflux(-0.001, 'Spat1')
        >>> # Add an other
        >>> FE.add_spatialflux(-0.002, 'Spat2')
        >>> # Add a third
        >>> FE.add_spatialflux(-0.003, 'Spat3')
        >>> # Check for all the currently available spatialflux names
        >>> FE.spatflux.keys()
        dict_keys(['Spat1', 'Spat2', 'Spat3'])
        >>> # Add the storage change function
        >>> FE.add_spatialflux(storage_change)
        >>> # Check which spatial state dependent flux function is saved
        >>> FE.Sspatflux.keys()
        dict_keys(['storage_change'])

        >>> # Specific removal of the spatial fluxes
        >>> FE.remove_spatialflux("Spat1", "Spat2")
        >>> FE.spatflux.keys()
        dict_keys(['Spat3'])
        >>> # Use an incorrect name
        >>> FE.remove_spatialflux("Spat4") #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        KeyError: "'Spat4' is not a spatialflux."
        >>> # Remove the remaining spatial fluxes
        >>> FE.remove_spatialflux()
        >>> FE.spatflux
        {}
        >>> FE.Sspatflux
        {}

        """
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
                        # Only raise exception at top of call stack
                        raise KeyError('{} is not a spatialflux.'.format(e)) from None

    def states_to_function(self):
        """ Prepare one-dimensional interpolation function

        One-dimensional piecewise linearly interpolated function which returns
        the system's states and is continiously defined on the domain
        :py:attr:`~nodes`.

        Returns
        -------
        `functools.partial`
            Function that calculates the system's states for a given position.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> FE = Flow1DFE("Continiously defined states")
        >>> FE.set_field1d((-10, 0, 11))
        >>> FE.set_initial_states([-i for i in range(11)])
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
        """ Solve the system for one specific time step

        Performs the Newton-Raphson method, :cite:`Newton1964`, to find a
        solution to the system of equations.

        Parameters
        ----------
        dt : `int` or `float`
            Time step which will be solved for.
        maxiter : `int`
            Maximum number of iterations in which the system should
            converge to a solution.
        threshold : `float`, default is 1e-3
            Threshold for conversion, the system has converged when the
            definition below is satisfied:

            .. math::
                max(abs(s_{t-1}-s_{t})) < max(abs(threshold * s_{t}))

        Notes
        -----
        .. note::
            This method may also be used for stationary systems in which no
            time step value is given. Any value of ``dt`` can be passed
            because of the method being independent from this argument in
            such a case.

        .. warning::
            Time step ``dt`` should not be to large. In :py:meth:`~solve`
            a more quantitative description of time step selection is
            described.

        **Procedure**

        1.  Check for proper boundary conditions (:py:meth:`~_check_boundaries`).
        2.  Update states function and time step size of storage change
            function if system is transient (:py:meth:`~_update_storage_change`).
        3.  Check if current iteration does not exceed ``maxiter``, otherwise return.
        4.  Collect all forcing in :py:attr:`~forcing`.

            a.  Aggregation of state independent forcing (:py:meth:`~_aggregate_forcing`).
            b.  Calculate the system’s internal forcing (:py:meth:`~_internal_forcing`).
            c.  Calculation of state dependent forcing values (:py:meth:`~_statedep_forcing`).

        5.  Build the jacobian matrix (:py:meth:`~_CMAT`).
        6.  Newton-Raphson iteration :cite:`Newton1964`.

            a.  :math:`A * y + F_{forcing} = 0` is solved for :math:`y` (:py:func:`~numpy.linalg.solve`).
            b.  :math:`y` is accumulated to :py:attr:`~states`.

        7. Check for convergence ``threshold``, if not satisfied proceed with
           next iteration from step 3.

        Returns
        -------
        `int`
            Number of iterations until system convergence or ``maxiter`` if
            the system has not been converged.

        Examples
        --------
        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility import fluxfunctions as fluxf
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Wosten2001`
        and prepare the conductivity function and theta-h relation with the
        soil parameters. These functions are the arguments to the fluxfunction
        and the storage change function repectively.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> theta_h = initializer(condf.VG_pressureh, theta_r=s.t_res,
        ...                       theta_s=s.t_sat, a=s.alpha, n=s.n)
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)
        >>> storage_change = initializer(fluxf.storage_change, fun=theta_h)

        >>> FE = Flow1DFE("Solve for one time step")
        >>> FE.set_systemfluxfunction(fluxf.richards_equation, kfun=kfun)
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> FE.add_dirichlet_BC(0.0, 'west')
        >>> # Constant boundary flow of 0.3 cm/d out of the system
        >>> FE.add_neumann_BC(-0.3, 'east')
        >>> # Solve the stationary system, independent or dt.
        >>> iterations = FE.dt_solve(dt=0.0)
        >>> iterations
        3
        >>> FE.states
        array([  0.        ,  -1.02794815,  -2.05972769,  -3.09448033,
                -4.13189871,  -5.17183029,  -6.21419211,  -7.25893958,
                -8.30605208,  -9.35552524, -10.40736646])

        >>> # Add storage change function to make the system transient
        >>> FE.add_spatialflux(storage_change)
        >>> # Change eastern boundary to drive a change in the system
        >>> FE.add_neumann_BC(-0.4, 'east')
        >>> # solve for 0.01 days
        >>> iterations = FE.dt_solve(dt=0.01)
        >>> iterations
        2
        >>> FE.states
        array([  0.        ,  -1.0353111 ,  -2.07549983,  -3.1195156 ,
                -4.16702367,  -5.21790586,  -6.27215519,  -7.32983969,
                -8.39108809,  -9.45608494, -10.52507089])

        """
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
        """ Solve the system for an arbitrary period of time

        The outer loop that progresses the model over time by calling
        :py:meth:`~dt_solve` sequentially until ``end_time`` is reached.
        Each sequential call to :py:meth:`~dt_solve` will produce a new
        solved object that holds the current model states at the specific time.
        This data will be saved to :py:attr:`~solve_data`. The total time that
        it takes for this method to converge to a solution at ``end_time`` is
        saved in :py:attr:`~runtime`.

        Parameters
        ----------
        dt : `int` or `float`
            Initial time step.
        dt_min : `int` or `float`, default is 1e-5
            Minimum time step size for which will be solved.
        dt_max : `int` or `float`, default is 0.5
            Maximum time step size for which will be solved.
        end_time : `int` or `float`, default is 1
            Total period of time for which will be solved.
        maxiter : `int`, default is 500
            Maximum number of iterations in which the system should
            converge to a solution.
        dtitlow : `float`, default is 1.5, :math:`>1`
            Multiplier for increasing ``dt`` for the next time step.
        dtithigh : `float`, default is 0.5, :math:`\\langle 0,1 \\rangle`
            Multiplier for decreasing ``dt`` for the next time step.
        itermin : `int`, default is 5, :math:`>2`
            Maximum number of iterations at which the time step ``dt``
            will be multiplied with its increasing multiplier ``dtitlow``
            for the next time step.
        itermax : `int`, default is 10, :math:`>3`
            Minimum number of iterations at which the time step ``dt``
            will be multiplied with its decreasing multiplier ``dtithigh``
            for the next time step.

            .. warning::
                ``itermin`` < ``itermin``. See the notes below for a
                precise description of the variable time step selection
                procedure.

        threshold : `float`, default is 1e-3
            Threshold for conversion, the system has converged when the
            definition below is satisfied:

            .. math::
                max(abs(s_{t-1}-s_{t})) < max(abs(threshold * s_{t}))

        verbosity : `bool`, default is True
            Be descriptive about the solve process.

        Notes
        -----

        .. note::
            This method may also be used for stationary systems in which no
            time step value is given. The default value for ``dt`` can be
            passed because of the method being independent from this argument
            in such a case.

        **Variable time step selection**

        The number of iterations, Niter, returned by :py:meth:`~dt_solve` will
        decide how the time step ``dt`` will be altered for the next call to
        :py:meth:`~dt_solve`. See the description below:

        1.  If Niter > ``maxiter``, then ``dt`` * ``dtithigh``.
        2.  If Niter < ``maxiter``, then check for the following:

                A.  If Niter <= ``itermin``, then ``dt`` * ``dtitlow``.
                B.  If Niter >= ``itermax``, then ``dt`` * ``dtithigh``.
                C.  If ``itermin`` < Niter < ``itermax``, then ``dt``.

        Note that ``dt`` cannot become smaller than ``dt_min`` or larger
        than ``dt_max``.

            .. warning::
                The size of the last time step is not determined by this
                procedure but will be calculated as difference between the
                current time and the ``end_time`` so the model will
                have its last solution on its ``end_time`` exactly.

        Examples
        --------
        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility import fluxfunctions as fluxf
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Wosten2001`
        and prepare the conductivity function and theta-h relation with the
        soil parameters. These functions are the arguments to the fluxfunction
        and the storage change function repectively.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> theta_h = initializer(condf.VG_pressureh, theta_r=s.t_res,
        ...                       theta_s=s.t_sat, a=s.alpha, n=s.n)
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)
        >>> storage_change = initializer(fluxf.storage_change, fun=theta_h)

        >>> FE = Flow1DFE("Solve for a period of time")
        >>> FE.set_systemfluxfunction(fluxf.richards_equation, kfun=kfun)
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> FE.add_dirichlet_BC(0.0, 'west')
        >>> FE.add_neumann_BC(-0.3, 'east')
        >>> FE.add_spatialflux(-0.001, 'Extraction')
        >>> FE.solve(verbosity=False)
        >>> FE.solve_data.keys()
        dict_keys(['solved_objects', 'time', 'dt', 'iter'])
        >>> # Solving the stationary system will call dt_solve only once. The
        >>> # result is the initial object and the solved one.
        >>> FE.solve_data['solved_objects']
        [Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time)]

        >>> # Make the system transient
        >>> FE.add_spatialflux(storage_change)
        >>> # Change the boundary to drive a change in the currently
        >>> # stationary solution.
        >>> FE.add_neumann_BC(-0.4, 'east')
        >>> FE.solve(verbosity=False, end_time=0.1)
        >>> # This will now return a sequence of solved objects at specific times.
        >>> FE.solve_data['solved_objects']
        [Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time), Flow1DFE(Solve for a period of time)]

        """
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

        .. math::
            F_{external} = \\sum_{j=1}^{n} S_{j} + \\sum_{k=1}^{m} P_{k}

        The calculation of the pointflux, :math:`P_{k}`, depends on its
        nature. If the pointflux depends on position only, the accumulation is
        straightforward. When the pointflux is dependent on the state of the
        the system, the distribution towards the surrounding nodes needs to be
        calculated before values can be accumulated. The distribution to the
        nearest nodes is calculated as follows:

        .. math::
            node_{i} = P_{k}(s) * lfac

        .. math::
            node_{i+1} = P_{k}(s) * rfac

        where :math:`s` equals the state at the position of the pointflux
        which is calculated by linear interpolation between the nearest nodes.
        The calculation of the fractions, that distribute the flux,
        :math:`rfac` and :math:`lfac` are described in
        :py:meth:`~add_pointflux`.

        For the calculation of the spatial flux, :math:`S_{j}`, a similar
        distinction exists. If the spatial flux is not dependend on state,
        straightforward addition takes place. When there is a state
        dependency, distributions towards the nodes is calculated as
        described in :py:meth:`~_statedep_forcing`. This calculation accounts
        for the selected :py:attr:`~gauss_degree` and the state argument
        :math:`s` is linearly interpolated between the neareast nodes.

        The total forcing in the water balance is the sum of the internal
        forcing as described in :py:meth:`~_internal_forcing` and the external
        forcing.

        .. math::
            F_{total} = F_{internal} + F_{external}

        The top and bottom values in :math:`F_{total}` are corrected
        for the flow over the boundaries. The flow over the boundaries is
        calculated as the difference of :math:`F_{total}` -
        :math:`\\Delta s` at those boundary nodes.

        The net flux is calculated as follows:

        .. math::
            F_{net} = F_{total} - \\Delta s

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

        Select soil 13, 'loam', from De Staringreeks :cite:`Wosten2001`
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
                    L = self.seg_lengths[i]
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
        """ Write internal model values to a dataframe

        Save the static model results to :py:attr:`~df_states`.

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
        >>> from waterflow.utility import fluxfunctions as fluxf
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Wosten2001`
        and prepare the conductivity function and theta-h relation with the
        soil parameters. These functions are the arguments to the fluxfunction
        and the storage change function repectively. Note that the theta-h
        relation needs to be added to the model explicitly.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> theta_h = initializer(condf.VG_pressureh, theta_r=s.t_res,
        ...                       theta_s=s.t_sat, a=s.alpha, n=s.n)
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)
        >>> storage_change = initializer(fluxf.storage_change, fun=theta_h)

        >>> FE = Flow1DFE("static df_states dataframe")
        >>> FE.set_systemfluxfunction(fluxf.richards_equation, kfun=kfun)
        >>> FE.set_field1d(nodes=(-10, 0, 11))
        >>> FE.add_dirichlet_BC(0.0, 'west')
        >>> # Constant boundary flow of 0.3 cm/d out of the system
        >>> FE.add_neumann_BC(-0.3, 'east')
        >>> # theta_h add manually to be included in the dataframe
        >>> FE.tfun = theta_h
        >>> # Add storage change function
        >>> FE.add_spatialflux(storage_change)
        >>> # Solve the system for one time step (dt=0.01 d)
        >>> iters = FE.dt_solve(dt=0.01)
        >>> FE.dataframeify(invert=True)
        >>> FE.df_states
            lengths  nodes    states  moisture  conductivities  storage_change  internal_forcing
        0       0.5    0.0 -9.304032  0.416865        5.936014        0.144618          0.155382
        1       1.0   -1.0 -8.278454  0.417344        6.220225        0.266340         -0.266340
        2       1.0   -2.0 -7.295883  0.417781        6.519720        0.222509         -0.222509
        3       1.0   -3.0 -6.345835  0.418182        6.840203        0.182383         -0.182383
        4       1.0   -4.0 -5.419426  0.418549        7.189191        0.145605         -0.145605
        5       1.0   -5.0 -4.509093  0.418885        7.577188        0.111975         -0.111975
        6       1.0   -6.0 -3.608378  0.419190        8.020009        0.081460         -0.081460
        7       1.0   -7.0 -2.711757  0.419462        8.544065        0.054224         -0.054224
        8       1.0   -8.0 -1.814469  0.419698        9.201041        0.030729         -0.030729
        9       1.0   -9.0 -0.912220  0.419888       10.126956        0.012064         -0.012064
        10      0.5  -10.0  0.000000  0.420000       12.980000        0.002066          0.951906

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
        """ Combine the static dataframes to a transient collection

        This method will build the following dataframes:
        :py:meth:`~dft_solved_times`, :py:attr:`dft_states`,
        :py:attr:`dft_balance`, :py:attr:`dft_balance_summary`.

        Generation of :py:attr:`dft_print_times` and :py:attr:`dft_nodes`
        depends on the ``print_times`` and ``nodes`` arguments.

        Parameters
        ----------
        print_times : `int`, `list` or `numpy.ndarray`, default is None
            Number of linearly spaced print times, or sequence of specific
            print times.
        include_maxima : `bool`, default is ``True``
            Include both endpoints in the dataframe.
        nodes : `list` or `numpy.ndarray`, default is None
            Positional values of the nodes that will be tracked over time.
        invert : `bool`, default is ``True``
            Mirror the built dataframes w.r.t. the x-axis.

        Notes
        -----
        The default behaviour is to generate the dataframes for the times at
        which the model has been solved. These times are selected by the
        :py:meth:`~solve` method and saved in :py:attr:`~dft_solved_times`.
        When ``print_times != None``, the collection of the static dataframes
        will be built at the new print times. This requires the model to
        calculate new model objects. The calculations are done from the nearest
        known object that was solved for in terms of time
        (:py:attr:`~dft_solved_times`) and will be saved in
        :py:attr:`~dft_print_times`.

        Examples
        --------

        >>> from waterflow.flow1d.flowFE1d import Flow1DFE
        >>> from waterflow.utility import conductivityfunctions as condf
        >>> from waterflow.utility import fluxfunctions as fluxf
        >>> from waterflow.utility.helper import initializer

        Select soil 13, 'loam', from De Staringreeks :cite:`Wosten2001`
        and prepare the conductivity function and theta-h relation with the
        soil parameters. These functions are the arguments to the fluxfunction
        and the storage change function repectively.

        >>> s, *_ = condf.soilselector([13])[0]
        >>> theta_h = initializer(condf.VG_pressureh, theta_r=s.t_res,
        ...                       theta_s=s.t_sat, a=s.alpha, n=s.n)
        >>> kfun = initializer(condf.VG_conductivity, ksat=s.ksat, a=s.alpha, n=s.n)
        >>> storage_change = initializer(fluxf.storage_change, fun=theta_h)

        >>> FE = Flow1DFE("All transient dataframes")
        >>> FE.set_systemfluxfunction(fluxf.richards_equation, kfun=kfun)
        >>> FE.set_field1d(nodes=(-100, 0, 11))
        >>> FE.add_dirichlet_BC(0.0, 'west')
        >>> # Constant boundary flow of 0.3 cm/d out of the system
        >>> FE.add_neumann_BC(-0.3, 'east')
        >>> # theta_h add manually to be included in the dataframe
        >>> FE.tfun = theta_h
        >>> # Extraction of 0.001 cm/d over the complete domain
        >>> FE.add_spatialflux(-0.001, 'extraction')
        >>> # Add storage change function
        >>> FE.add_spatialflux(storage_change)
        >>> FE.solve(end_time=5, verbosity=False)
        >>> FE.transient_dataframeify(nodes=[0, -2, -5, -8, -10])
        >>> # returns None because it is empty
        >>> FE.dft_print_times

        >>> FE.dft_solved_times.head()
                               solved_objects     time       dt  iter
        0  Flow1DFE(All transient dataframes)  0.00000      NaN   NaN
        1  Flow1DFE(All transient dataframes)  0.00100  0.00100   7.0
        2  Flow1DFE(All transient dataframes)  0.00200  0.00100   3.0
        3  Flow1DFE(All transient dataframes)  0.00350  0.00150   3.0
        4  Flow1DFE(All transient dataframes)  0.00575  0.00225   3.0

        >>> # access transient data from the top node
        >>> FE.dft_nodes[0]
                time  lengths  nodes      states  moisture  conductivities  extraction  storage_change  internal_forcing
        0   0.000000      5.0    0.0    0.000000  0.420000       12.980000      -0.005        0.000000        -12.980000
        1   0.001000      5.0    0.0   -6.230996  0.418228        6.881334      -0.005        4.413886         -4.108886
        2   0.002000      5.0    0.0   -8.525052  0.417231        6.149441      -0.005        3.381237         -3.076237
        3   0.003500      5.0    0.0  -10.769689  0.416145        5.570882      -0.005        2.713307         -2.408307
        4   0.005750      5.0    0.0  -13.147704  0.414894        5.058867      -0.005        2.213596         -1.908596
        5   0.009125      5.0    0.0  -15.759044  0.413423        4.583379      -0.005        1.816135         -1.511135
        6   0.014188      5.0    0.0  -18.682582  0.411673        4.131618      -0.005        1.490460         -1.185460
        7   0.021781      5.0    0.0  -21.992779  0.409584        3.698174      -0.005        1.219763         -0.914763
        8   0.033172      5.0    0.0  -25.766774  0.407089        3.281456      -0.005        0.993457         -0.688457
        9   0.050258      5.0    0.0  -30.088421  0.404117        2.882094      -0.005        0.804074         -0.499074
        10  0.075887      5.0    0.0  -35.050660  0.400591        2.502107      -0.005        0.645821         -0.340821
        11  0.114330      5.0    0.0  -40.755672  0.396435        2.144426      -0.005        0.513876         -0.208876
        12  0.171995      5.0    0.0  -47.311047  0.391582        1.812568      -0.005        0.404104         -0.099104
        13  0.258493      5.0    0.0  -54.818914  0.385985        1.510322      -0.005        0.312974         -0.007974
        14  0.388239      5.0    0.0  -63.353249  0.379645        1.241351      -0.005        0.237546          0.067454
        15  0.582859      5.0    0.0  -72.918234  0.372640        1.008723      -0.005        0.175458          0.129542
        16  0.874788      5.0    0.0  -83.379449  0.365167        0.814411      -0.005        0.124874          0.180126
        17  1.312682      5.0    0.0  -94.367200  0.357578        0.658911      -0.005        0.084417          0.220583
        18  1.812682      5.0    0.0 -103.335192  0.351603        0.559141      -0.005        0.058055          0.246945
        19  2.312682      5.0    0.0 -109.947478  0.347330        0.497676      -0.005        0.041411          0.263589
        20  2.812682      5.0    0.0 -114.891068  0.344208        0.457272      -0.005        0.030168          0.274832
        21  3.312682      5.0    0.0 -118.614160  0.341899        0.429578      -0.005        0.022269          0.282730
        22  3.812682      5.0    0.0 -121.429531  0.340177        0.410046      -0.005        0.016582          0.288417
        23  4.312682      5.0    0.0 -123.563493  0.338885        0.395994      -0.005        0.012421          0.292579
        24  4.812682      5.0    0.0 -125.183260  0.337912        0.385736      -0.005        0.009343          0.295657
        25  5.000000      5.0    0.0 -125.727039  0.337587        0.382369      -0.005        0.008330          0.296670

        >>> # Revert to initial model state
        >>> FE.set_initial_states(0.0)
        >>> FE.solve(end_time=5, verbosity=False)
        >>> # 5 linearly spaced print times
        >>> FE.transient_dataframeify(nodes=[0, -2, -5, -8, -10], print_times=5)
        >>> FE.dft_nodes[0]
           time  lengths  nodes      states  moisture  conductivities  extraction  storage_change  internal_forcing
        0  0.00      5.0    0.0    0.000000  0.420000       12.980000      -0.005        0.008330          0.296670
        1  1.25      5.0    0.0  -93.203885  0.358368        0.673469      -0.005        0.088281          0.216719
        2  2.50      5.0    0.0 -112.163382  0.345923        0.479024      -0.005        0.036331          0.268668
        3  3.75      5.0    0.0 -121.156173  0.340343        0.411892      -0.005        0.017130          0.287870
        4  5.00      5.0    0.0 -125.727039  0.337587        0.382369      -0.005        0.008330          0.296670

        >>> FE.dft_print_times
                               solved_objects  time
        0  Flow1DFE(All transient dataframes)  0.00
        1  Flow1DFE(All transient dataframes)  1.25
        2  Flow1DFE(All transient dataframes)  2.50
        3  Flow1DFE(All transient dataframes)  3.75
        4  Flow1DFE(All transient dataframes)  5.00

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
        """ Save model data to disk

        The transient dataframes created with
        :py:meth:`~transient_dataframeify` and the model summary as created
        with :py:meth:`~summary` will be saved to disk by this method.

        Parameters
        ----------
        savepath: `str`, default is :py:attr:`~savepath`
            A base path to which runs will be saved.
        dirname : `str`, default is a chronological name
            Name of save directory that is appended to savepath where
            the data of the current model run will be stored.

        Notes
        -----
        All dataframes of the form ``dft_<name>``, if populated,
        are written to disk in a `.xlsx` extension. The model summary
        is saved as :py:attr:`~id_` with a `.txt` extension.

        .. warning::
            Data that already exists in the target directory will
            be overwritten with new data. Prevent this by selecting a
            new directory name or set ``dirname=None`` to automatically
            generate a chronological directory name which is always unique.

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

        # write transient dataframes
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

        # write model summary
        self.summary(show=False, save=True, path=runpath)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
