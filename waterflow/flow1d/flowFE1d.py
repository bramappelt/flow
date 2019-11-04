""" One dimensional finite elements flow module """


from inspect import signature
from functools import partial
from copy import deepcopy
import time as Time
import os

import numpy as np
import pandas as pd
from scipy.integrate import quad

from waterflow import OUTPUT_DIR


'''
documentation !!
'''


class Flow1DFE:
    """ Class for solving flow problems numerically

    This class represents an object that can be used to solve
    (un)saturated 1-dimensional flow problems using finite elements.
    To increase the accuracy of numerical solutions the Gaussian
    Quadrature method is used for integration.

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
        Name of the model object as passed to the class constructor.
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
        This defines the system's boundary conditions.
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
        system flux function as saved in :py:attr:`~systemfluxfunc`,
        using the selected Gaussian Quadrature :py:attr:`~scheme`.
    forcing : `numpy.ndarray`
        pass
    conductivities : `numpy.ndarray`
        pass
    moisture : `numpy.ndarray`
        pass
    fluxes : `numpy.ndarray`
        pass
    isinitial : `bool`, default is True
        pass
    isconverged : `bool`, default is False
        pass
    scheme : `str`
        pass

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
        self.dft_balance = None
        self.dft_balance_summary = None
        self.dft_nodes = None
        # private attributes
        self._west = None
        self._east = None
        self._delta = 1e-5

        # specific
        self.scheme = "linear"
        self._schemes = ["linear", "quadratic", "cubic", "quartic", "quintic"]
        self._xgaus = None
        self._wgaus = None
        self.gausquad = None
        self.xintegration = None
        self._calcgaussianquadrature()

    def __repr__(self):
        return "Flow1DFE(" + str(self.id) + ")"

    def summary(self, show=True):
        id_ = f"{self.id}"
        if self.nodes is not None:
            len_ = str(self.nodes[-1] - self.nodes[0])
            num_nodes = str(len(self.nodes))
        else:
            len_ = None
            num_nodes = None
        scheme = self.scheme
        bcs = [[k, self.BCs[k][0], self.BCs[k][1]] for k in self.BCs.keys()]
        bcs = ["{} value: {} and of type {}".format(*bc) for bc in bcs]
        bcs = ", ".join(i for i in bcs)
        pkeys = list(self.pointflux.keys()) + list(self.Spointflux.keys())
        skeys = list(self.spatflux.keys()) + list(self.Sspatflux.keys())
        pointflux = ", ".join(i for i in pkeys)
        spatflux = ", ".join(i for i in skeys)
        runtime = self.runtime

        k = ['Id', 'System lenght', 'Number of nodes', 'Scheme',
             'BCs', 'Pointflux', 'Spatflux', 'Runtime (s)']
        v = (id_, len_, num_nodes, scheme, bcs, pointflux, spatflux, runtime)
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

    def _calcgaussianquadrature(self):
        ''' Calculate gaussian quadrature points '''
        pos = []
        weights = []
        for i in range(1, len(self._schemes) + 1):
            # calculate gaussian quadrature points for degree i
            p, w = np.polynomial.legendre.leggauss(i)
            # shift points from domain [-1, 1] to [0, 1]
            pos.append(tuple(np.array(p) / 2 + 0.5))
            weights.append(tuple(np.array(w) / 2))
        # assign data to class attributes
        self._xgaus = tuple(pos)
        self._wgaus = tuple(weights)
        self.gausquad = dict(zip(self._schemes, zip(pos, weights)))

    def _aggregate_forcing(self):
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
        """ Internal fluxes for model convergence,
        flux calculation or balance calculation. Only one calculation
        at a time, when both arguments are truthy, the first argument
        in the method signature is calculated.
        """
        # internal fluxes from previous iteration
        pos, weight = self.gausquad[self.scheme]
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
            # if balance calculation, dont assign to forcing again
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
        pos, weight = self.gausquad[self.scheme]
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
        # check whether the system is circular or not
        # no boundary conditions entered
        keys = list(self.BCs.keys())
        if len(keys) == 0:
            raise np.linalg.LinAlgError("Singular matrix")
        # If one boundary is not entered a zero neumann boundary is set
        if len(keys) == 1:
            val, type_, pos = self.BCs[keys[0]]
            if type_ == "Dirichlet" and pos == 0:
                self.add_neumann_BC(value=0, where="east")
            elif type_ == "Dirichlet" and pos == -1:
                self.add_neumann_BC(value=0, where="west")
            else:
                raise np.linalg.LinAlgError("Singular matrix")
        # both boundaries cannot be of type neumann
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

    def _FE_precalc(self, nodes):
        # calculate the x positions of each integration point between nodes
        xintegration = [[] for x in range(len(nodes) - 1)]
        pos, weight = self.gausquad[self.scheme]
        for i in range(len(nodes) - 1):
            for j in range(len(pos)):
                xij = nodes[i] + (nodes[i+1] - nodes[i]) * pos[j]
                xintegration[i].append(xij)
        self.xintegration = xintegration

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
            pos, weight = self.gausquad[self.scheme]
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

    def set_field1d(self, **kwargs):
        ''' Define the nodal structure '''
        for key in kwargs.keys():
            if key == "linear":
                self.nodes = np.linspace(0, kwargs[key][0], kwargs[key][1])
                self.states = np.repeat(0, len(self.nodes))
                self._FE_precalc(self.nodes)
                break
            elif key == "array":
                self.nodes = np.array(kwargs[key])
                self.states = np.repeat(0, len(self.nodes))
                self._FE_precalc(self.nodes)
                break

    def set_scheme(self, scheme):
        if scheme.lower() in self._schemes:
            self.scheme = scheme.lower()
        else:
            raise ValueError(f"Select any of these schemes: {self._schemes}")

    def set_systemfluxfunction(self, function, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        def fluxfunction(x, s, gradient):
            return function(x, s, gradient, **kwargs)
        self.systemfluxfunc = fluxfunction

    def set_initial_states(self, states):
        if isinstance(states, int) or isinstance(states, float):
            states = float(states)
            self.states = np.array([states for x in range(len(self.nodes))])
        else:
            self.states = np.array(states, dtype=np.float64)

    def add_dirichlet_BC(self, value, where):
        if isinstance(value, int) or isinstance(value, float):
            value = [value]
            where = [where]

        for val, pos in zip(value, where):
            if pos.lower() in "western":
                self.BCs["west"] = (val, "Dirichlet", 0)
                self._west = 1
            elif pos.lower() in "eastern":
                self.BCs["east"] = (val, "Dirichlet", -1)
                self._east = -1

    def add_neumann_BC(self, value, where):
        if isinstance(value, int) or isinstance(value, float):
            value = [value]
            where = [where]

        for val, pos in zip(value, where):
            if pos.lower() in "western":
                self.BCs["west"] = (val, "Neumann", 0)
                self._west = 0
            elif pos.lower() in "eastern":
                self.BCs["east"] = (val, "Neumann", -1)
                self._east = len(self.nodes)

    def remove_BC(self, *args):
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
                    pos, weight = self.gausquad[self.scheme]
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
        ''' gives a continuous function of states in the domain '''
        circular = True
        states = self.states.copy()
        # check if west boundary is of type Dirichlet
        if self._west == 1:
            states[0] = self.BCs["west"][0]
            circular = False
        # check is east boundary is of type Dirichlet
        if self._east == -1:
            states[-1] = self.BCs["east"][0]
            circular = False
        # if none entered or both of type Neumann, return None
        if circular:
            print("Define the boundary conditions first")
            return None
        # linearly interpolate between states including assigned boundaries
        else:
            return partial(np.interp, xp=self.nodes, fp=states)

    def dt_solve(self, dt, maxiter=500, threshold=1e-3):
        ''' solve the system for one specific time step '''
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
        ''' solve the system for a given period of time '''

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
                pos, weight = self.gausquad[self.scheme]
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

        # flow over boundaries
        fill = np.repeat(0.0, (len(self.nodes) - 1,))
        leftb = np.insert(fill, 0, internalfluxes[0] + (spat[0] + pnt[0])) * -1
        rightb = np.append(fill, internalfluxes[-1] + (spat[-1] + pnt[-1])) * -1

        # net flow
        net = pnt + spat + internalfluxes
        net[0] += leftb[0]
        net[-1] += rightb[-1]

        # fluxes
        self.fluxes[0] = leftb[0]
        self.fluxes[-1] = -rightb[-1]

        # dump waterbalance & summary to dataframe
        data.update({'internal': internalfluxes, 'all-spatial': spat,
                     'all-points': pnt, 'all-external': pnt + spat,
                     'lbound': leftb, 'rbound': rightb, 'net': net,
                     'fluxes': self.fluxes})

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
        ''' write current static model to dataframe '''
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
        ''' write transient solve data to dataframe '''

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
        """ Save model metadata and dataframes

        Parameters
        ----------
        savepath: :obj:`str`, default is `OUTDIR`
            A base path to which runs will be saved.
        dirname : :obj:`str`, default is a chronological name
            Name of save directory that is appended to savepath.

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
