from copy import deepcopy
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from inspect import signature
from functools import partial

from waterflow.utility.spacing import spacing
from waterflow.utility.statistics import RMSE, MAE
from waterflow.utility.forcingfunctions import polynomial

'''
documentation !!
'''


class Flow1DFE(object):
    def __init__(self, id_):
        self.id = id_
        self.systemfluxfunc = None
        self.nodes = None
        self.states = None
        self.nframe = None
        self.lengths = None
        self.coefmatr = None
        self.BCs = {}
        self.spatflux = {}
        self.pointflux = {}
        self.internal_forcing = {}
        self.Spointflux = {}
        self.Sspatflux = {}
        self.forcing = None
        self.scheme = "linear"
        self.conductivities = []
        self.moisture = []
        self.stats = {"rmse" : [], "mae" : []}
        self.balance = {}
        # private attributes
        self._west = None
        self._east = None
        self._delta = 1e-5

        # specific
        self._schemes = ["linear", "quadratic", "cubic", "quartic", "quintic"]
        self._xgaus = None
        self._wgaus = None
        self.gausquad = None
        self.xintegration = None

        self._calcgaussianquadrature()

    def __repr__(self):
        return "Flow1DFE("+ str(self.id) +")"

    def __str__(self):
        id_ = "{}".format(self.id)
        if self.nodes is not None:
            len_ = "System lenght: {}".format(self.nodes[-1] - self.nodes[0])
            num_nodes = "Number of nodes: {}".format(len(self.nodes))
        else:
            len_ = None
            num_nodes = None
        bcs = [[k, self.BCs[k][0], self.BCs[k][1]] for k in self.BCs.keys()]
        bcs = ["{} value: {} and of type {}".format(*bc) for bc in bcs]
        bcs = ", ".join(i for i in bcs)
        pkeys = list(self.pointflux.keys()) + list(self.Spointflux.keys())
        skeys = list(self.spatflux.keys()) + list(self.Sspatflux.keys())
        pointflux = ", ".join(i for i in pkeys)
        spatflux = ", ".join(i for i in skeys)

        if 'net' in self.balance:
            netbalance = sum(self.balance['net'])
        else:
            netbalance = "Not yet solved"

        return "id: {}\n{}\n{}\nBCs: {}\nPointflux: {}\nSpatial flux: {}\
               \nNet balance: {}\n".format(id_, len_, num_nodes, bcs, pointflux,
               spatflux, netbalance)

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

    def _internal_forcing(self, calcbal=False):
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
                f[i] -= flux * weight[-idx-1]
                f[i+1] += flux * weight[idx]

        self.internal_forcing["internal_forcing"] = [None, np.array(f)]
        # if balance calculation, dont assign to forcing again
        if not calcbal:
            self.forcing = self.forcing + np.array(f)

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

        # calculate the lenght of each 1d finite element
        length = []
        length.append(middle[0] - nodes[0])
        for i in range(1, len(nodes) - 1):
            length.append(middle[i] - middle[i-1])
        length.append(nodes[-1] - middle[-1])

        # assign to class attributes
        self.nframe = np.array(list((zip(middle, nodal_distances))))
        self.lengths = np.array(length)

    def _CMAT(self, nodes, states): # !!!! 
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
        self.scheme = scheme

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
            self.BCs  = {}
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
                name = str(time.time())
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
                name = str(time.time())
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
                            # to left node
                            f[i] +=  Q(x) * weight[-idx-1] * pos[-idx-1] * L    ### no pos negatives??
                            # to right node
                            f[i+1] += Q(x) * weight[idx] * pos[idx] * L
                
                # only possible if analytical solution exists
                else:
                    # exact integral
                    nodes = self.nodes
                    for i in range(len(self.nodes) - 1):
                        l = self.wrap_bf_linear(i,  "left")
                        r = self.wrap_bf_linear(i, "right")

                        def to_left(x):
                            return l(x) * Q(x)

                        def to_right(x):
                            return r(x) * Q(x)

                        # to left node
                        f[i] += quad(to_left, nodes[i],nodes[i+1])[0]
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
        # if none entered of both of type Neumann, return None
        if circular:
            print("Define the boundary conditions first")
            return None
        # linearly interpolate between states including assigned boundaries
        else:
            return partial(np.interp, xp = self.nodes, fp = states)

    def dt_solve(self, dt, maxiter=500, threshold=1e-3):
        ''' solve the system for one specific time step '''
        self._check_boundaries()
        west = self._west
        east = self._east

        if self.Sspatflux.get('storage_change', None):
            # adapt storage change function for time step and previous states
            storage_change = self.Sspatflux['storage_change'][0]
            previous_states = self.states_to_function()
            storage_change = partial(storage_change, prevstate=previous_states, dt=dt)
            self.Sspatflux['storage_change'][0] = storage_change

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
        
        self._calc_theta_k()
        return itercount

    def solve(self, dt_min=0.01, dt_max=0.5, end_time=1, maxiter=500, threshold=1e-3, verbosity=True):
        ''' solve the system for a given period of time '''
        
        solved_objs = [deepcopy(self)]
        time_data = [0]
        dt_data = [None]
        iter_data = [None]

        dt = dt_min
        time = dt

        while time <= end_time:
            # solve for given dt
            iters = self.dt_solve(dt, maxiter, threshold)

            if verbosity:
                if self.Sspatflux.get('storage_change', None):
                    fmt = 'Converged at time={} for dt={} with {} iterations'
                    print(fmt.format(time, dt, iters))
                else:
                    print('Converged at {} iterations'.format(iters))

            if not self.Sspatflux.get('storage_change', None):
                solved_objs.append(deepcopy(self))
                time_data.append(time) 
                dt_data.append(dt)
                iter_data.append(iters)
                break

            if dt == dt_min and iters == maxiter:
                print('Maxiter at dt_min reached...')
                break

            # build data record
            solved_objs.append(deepcopy(self))
            time_data.append(time)
            dt_data.append(dt)
            iter_data.append(iters)

            # adapt time step as function of iterations
            if iters <= 5:
                dt *= 1.5
                if dt > dt_max:
                    dt = dt_max
            elif iters >= 10:
                dt *= 0.5
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
    
        # attach all solve data to last created object
        solve_data = {}
        solve_data['solved_objects'] = solved_objs
        solve_data['time'] = time_data
        solve_data['dt'] = dt_data 
        solve_data['iter'] = iter_data
        self.solve_data = solve_data

    def calcbalance(self, print_=False):
        # internal fluxes
        self._internal_forcing(calcbal=True)
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
            self.balance["pnt-"+str(key)] = pointfluxes[key][-1]

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
            self.balance["spat-"+str(key)] = spatialfluxes[key][-1]

        # flow over boundaries
        leftb = internalfluxes[0] + (spat[0] + pnt[0])
        rightb = internalfluxes[-1] + (spat[-1] + pnt[-1])
        bnd = (leftb + rightb)

        # net flow
        net = pnt + spat
        net[0] -= leftb
        net[-1] -= rightb

        self.balance['internal'] = internalfluxes
        self.balance['all-spatial'] = spat
        self.balance['all-points'] = pnt
        self.balance['all-external'] = pnt + spat
        self.balance['lbound'] = leftb
        self.balance['rbound'] = rightb
        self.balance['bounds'] = bnd
        self.balance['net'] = net

        if print_:
            basis_terms = ["internal", "all-spatial", "all-points",
                           "all-external", "lbound", "rbound", "bounds",
                           "net"]
            for key in self.balance.keys():
                if key not in basis_terms:
                    print(key, sum(self.balance[key]))

            for key in basis_terms:
                try:
                    print(key, sum(self.balance[key]))
                except:
                    try:
                        print(key, self.balance[key])
                    except Exception as e:
                        raise type(e)

    def dataframeify(self, invert):
        columns = ['lengths', 'nodes', 'states', 'moisture',
                   'conductivities','pointflux', 'Spointflux',
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
        return df

    def save(self, print_time, invert=True, dirname=None, savepath='C:/Users/bramb/Documents/thesis/output'):
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

        # new dir for current run
        if not dirname:
            dirname = time.strftime('%d%b%Y_%H%M%S', time.gmtime(time.time()))
            runpath = os.path.join(savepath, dirname)
            os.mkdir(runpath)
        else:
            runpath = os.path.join(savepath, dirname)
            if not os.path.isdir(runpath):
                os.mkdir(runpath)

        # time data to file
        timefile = os.path.join(runpath, 'time.xlsx')
        transient_data = pd.DataFrame(data=self.solve_data)
        transient_data.to_excel(timefile, engine='xlsxwriter')

        statefile = os.path.join(runpath, 'states.xlsx')
        with pd.ExcelWriter(statefile) as fw:
            for row in transient_data.itertuples(index=False):
                obj, t = row.solved_objects, row.time

                # call a function that 'dataframeifies' the object
                dft = pd.DataFrame(obj.dataframeify(invert=invert))
                dft.to_excel(fw, sheet_name=str(t))
