import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from inspect import signature
from functools import partial

from waterflow.utility.spacing import spacing
from waterflow.utility.statistics import RMSE, MAE
from waterflow.utility.forcingfunctions import polynomial

'''
Documentation !!
'''

class Flow1DFV(object):
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
        self.Sspatflux = {}
        self.Spointflux = {}
        self.forcing = None
        self.stats = {"rmse" : [], "mae" : []}
        self.balance = {}
        # private attributes
        self._west = None
        self._east = None
        self._delta = 1e-5
        
    def __repr__(self):
        return "Flow1DFV("+ str(self.id) +")"
        
    def __str__(self):
        id = "{}".format(self.id)
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
               \nNet balance: {}\n".format(id, len_, num_nodes, bcs, pointflux,
               spatflux, netbalance)
        
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
        f = [0.0 for x in range(len(self.nodes))]
        for i in range(len(self.nodes) - 1):
            middle = self.nframe[i][0]
            state = (self.states[i+1] + self.states[i]) / 2
            grad = (self.states[i+1] - self.states[i]) / self.nframe[i][1]
            f[i] -= self.systemfluxfunc(middle, state, grad)
            f[i+1] += self.systemfluxfunc(middle, state, grad)
        
        self.internal_forcing["internal_forcing"] = [None, np.array(f)]
        # if balance calculation, dont assign to forcing again
        if not calcbal:
            self.forcing = self.forcing + np.array(f)
    
    def _statedep_forcing(self):         
        # point state dependent forcing
        f = [0.0 for x in range(len(self.nodes))]
        for key in self.Spointflux.keys():
            (Sfunc, idx), value = self.Spointflux[key]
            value = Sfunc(self.states[idx])
            f[idx] += value
            self.Spointflux[key][1] = np.array(f)
            
        # spatial state dependent forcing
        f = [0.0 for x in range(len(self.nodes))]
        for key in self.Sspatflux.keys():
            for i in range(len(self.nodes)):
                Sfunc = self.Sspatflux[key][0]
                f[i] = Sfunc(self.nodes[i], self.states[i]) * self.lengths[i]
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

    def _FV_precalc(self, nodes):
        """ Assigns middle points between the nodes and the distance between 
        the nodes to the nframe class attribute. Secondly, the lenght of each 
        finite volumes element is assigned to the lengths attribute"""
        middle = []
        nodal_distances = []

        # calculate positions of midpoints and nodal distances
        for i in range(0, len(nodes) - 1):
            middle.append((nodes[i] + nodes[i+1]) / 2)
            nodal_distances.append(nodes[i+1] - nodes[i])
        
        # calculate the lenght of each 1d volume element
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
            xmid = self.nframe[::,0][i]
            L = self.nframe[::,1][i]
        
            stateleft = states[i] + np.array([0, self._delta, 0])
            stateright = states[i+1] + np.array([0, 0, self._delta])
            
            grad = (stateright - stateleft) / L
            smid = (stateright + stateleft) / 2
            flux = [systemflux(xmid, s, g) for s, g in zip(smid, grad)]
            
            dfluxl = (flux[1] - flux[0])/ self._delta
            dfluxr = (flux[2] - flux[0])/ self._delta
            
            # assign flux values to the coefficient matrix
            A[i][i] += -dfluxl
            A[i+1][i] += dfluxl
            A[i][i+1] += -dfluxr
            A[i+1][i+1] += dfluxr
            
        # state dependent point flux
        for key in self.Spointflux.keys():
            (Sfunc, idx), value = self.Spointflux[key]
            Ai = Sfunc(self.states[idx])
            Aiw = Sfunc(self.states[idx] + self._delta)
            Aiiw = (Aiw - Ai) / self._delta
            A[idx][idx] += Aiiw
            
        # state dependent spatial flux
        for key in self.Sspatflux.keys():
            for i in range(len(self.nodes)):
                Sfunc = self.Sspatflux[key][0]
                Ai = Sfunc(self.nodes[i], self.states[i])
                Aiw= Sfunc(self.nodes[i], self.states[i] + self._delta)
                Aiiw = (Aiw - Ai) * self.lengths[i] / self._delta
                A[i][i] += Aiiw

        self.coefmatr = A
            
    def set_field1d(self, **kwargs):
        ''' Define the nodal structure '''
        for key in kwargs.keys():
            if key == "linear":
                self.nodes = np.linspace(0, kwargs[key][0], kwargs[key][1])
                self.states = np.repeat(1.0, len(self.nodes))
                self._FV_precalc(self.nodes)
                break
            elif key == "array":
                self.nodes = np.array(kwargs[key])
                self.states = np.repeat(1.0, len(self.nodes))
                self._FV_precalc(self.nodes)
                break
            
    def set_systemfluxfunction(self, function, **kwargs):
        def fluxfunction(x, s, gradient, **kwargs):
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
                
    def add_pointflux(self, rate, pos, name): #!!!!
        # add single valued pointflux to corresponding control volume
        f = [x*0.0 for x in range(len(self.nodes))]
        if isinstance(rate, int) or isinstance(rate, float):
            rate = [rate]
            pos = [pos]
        if isinstance(rate, list):
            for r, p in zip(rate, pos):
                f[np.abs(self.nodes - p).argmin()] += r
            self.pointflux[name] = [np.array(f)]
        
        # add state dependent pointflux
        elif callable(rate):
            idx = abs(self.nodes - pos).argmin()
            self.Spointflux[name] = [(rate, idx), np.array(f)]
            
    def add_spatialflux(self, q, name, exact=False): #!!!!
        if isinstance(q, int) or isinstance(q, float):
            Q = np.repeat(q, len(self.nodes)).astype(np.float64)
        elif isinstance(q, list) or isinstance(q, np.ndarray):
            Q = np.array(q).astype(np.float64)
        elif callable(q):
            Q = q
            
        # If single value or array, add to forcing fluxes
        if not callable(Q):
            f = Q * self.lengths
            self.spatflux[name] = [np.array(f)]
        
        # if function, check its number of arguments
        else:
            f = [x*0.0 for x in range(len(self.nodes))]
            
            # if function has one argument > f(x)
            if len(signature(Q).parameters) == 1:
                # linear approximation integral
                if not exact:
                    for i in range(len(self.nodes)):
                        f[i] = Q(self.nodes[i]) * self.lengths[i]
                # exact integral
                elif exact:
                    int_l = self.nodes[0]
                    int_r = self.nodes[0] + self.lengths[0]
                    for i in range(len(self.nodes)):
                        f[i] = quad(Q, int_l, int_r)[0]
                        int_l = int_r
                        int_r += self.lengths[(i+1) % len(self.nodes)]
                                                    # trick to avoid IndexError
                self.spatflux[name] = [np.array(f)]
            
            # if function has two arguments > f(x,s)
            if len(signature(Q).parameters) == 2:
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
        circular = True
        states = self.states.copy()
        # check if west boundary is of type Dirichlet
        if self._west == 1:
            states[0] = self.BCs["west"][0]
            circular = False
        # check is east bounary is of type Dirichlet
        if self._east == -1:
            states[-1] = self.BCs["east"][0]
            circular = False
        # if none entered of both of type Neumann, return None
        if circular:
            print("Define the boundary conditions first")
            return None
        # return function including the proper state boundaries
        else:
            # return states at nearest node including assigned boundaries
            def state_nearest_node(x):
                return states[np.abs(self.nodes - x).argmin()]
            return state_nearest_node

    def solve(self, maxiter=1000, rmse_threshold=1e-16, mae_threshold=1e-16):
        self._check_boundaries()
        west = self._west
        east = self._east
        self.stats = {"rmse" : [], "mae" : []}
        iter_step = 1
        while iter_step <= maxiter:
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
            
            self._aggregate_forcing()
            self._internal_forcing()
            self._statedep_forcing()
            
            rmse = RMSE(prevstates, curstates)
            mae = MAE(prevstates, curstates)
            
            if rmse < rmse_threshold:
                #self.states = newstates
                self.stats["rmse"].append(rmse)
                self.stats["mae"].append(mae)
                print("Small rmse", iter_step)
                break
            
            if mae < mae_threshold:
                self.stats["rmse"].append(rmse)
                self.stats["mae"].append(mae)
                print("Small mae", iter_step)
                break
            
            self.stats["rmse"].append(rmse)
            self.stats["mae"].append(mae)
            
            if iter_step == maxiter:
                print("max iter reached", iter_step)
            
            iter_step += 1
        
    def calcbalance(self, print_=False):
        # internal fluxes
        self._internal_forcing(calcbal=True)
        internalfluxes = self.internal_forcing['internal_forcing'][-1]
        
        # add all point fluxes
        pnt = np.repeat(0.0, len(self.nodes))
        pointfluxes = {**self.pointflux, **self.Spointflux}
        for key in pointfluxes.keys():
            pnt += pointfluxes[key][-1]
            self.balance["pnt-"+str(key)] = pointfluxes[key][-1]
        
        # add all spatial fluxes
        spat = np.repeat(0.0, len(self.nodes))
        spatialfluxes = {**self.spatflux, **self.Sspatflux}
        for key in spatialfluxes.keys():
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
        
        # assign balance arrays to class attr dict
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
                           "all-external", "lbound", "rbound", "bounds", "net"]
            # print self entered forcing components first
            for key in self.balance.keys():
                if key not in basis_terms:
                    print(key, sum(self.balance[key]))
            # print basis balance terms
            for key in basis_terms:
                try:
                    print(key, sum(self.balance[key]))
                except:
                    try:
                        print(key, self.balance[key])
                    except Exception as e:
                        raise type(e)
        
    
if __name__ == "__main__":
    L = 20
    nx = 11
    domain = [L, nx]
    S = 1.0
    dt = 1.0
    runs = 10
    
    ksat = 1.5
    def kfun(x):
        return ksat + 0.0065*x

############################### FLUXFUNCTIONS #################################

    def fluxfunction(x, s, gradient):
        return -1 * gradient * ksat
    
    def fluxfunction_s(x, s, gradient):
        return -1 * gradient * ksat * s
    
    def fluxfunction_var_k(x, s, gradient):
        return - kfun(x) * gradient
    
    def fluxfunction_var_k_s(x, s, gradient):
        return - kfun(x) * s * gradient

############################### POINT FLUXES ##################################
    
    Spointflux = partial(np.interp, xp=[4.9, 4.95, 5.005], 
                        fp=[-0.02, -0.029, -0.034])
    
############################### SPATIAL FLUXES ################################
    
    def rainfun(x):
        return 1.98e-5*x**3 - 7.34e-4 * x**2 + 5.36e-3 * x
    
    def rainfunc(x):
        return x*0.001 + 0.001
    
    a2, x2, b2, rainfun2 = polynomial([[0, 0.001], [5, 0.003], [10, 0.005], 
                                    [15, 0.003], [20, 0.002]])
    
    def storage_change(x, s):
        return -S*(s - prevstate(x)) / dt
    
    def stateposfunc(x, s):
        return (-2 + np.sin(x) + s) / 1000
    
################################# STRUCTURED ##################################
    
    FV = Flow1DFV("structured")
    FV.set_field1d(linear=domain)
    FV.set_systemfluxfunction(fluxfunction)
    FV.set_initial_states(4.90)
    
    FV.add_dirichlet_BC(5.0, "west")
    #FV.add_dirichlet_BC(5.01, "east")
    FV.add_neumann_BC(0.01, "east")
    
    FV.add_pointflux([-0.027, -0.015], [4.0, 8.0], "well!")
    FV.add_pointflux(-0.02, 10, "well1")
    FV.add_pointflux(Spointflux, 6.0, "Swell!")
    
    FV.add_spatialflux(0.003, "recharge")    
    FV.add_spatialflux(rainfunc, "rainfunc")
    FV.add_spatialflux(rainfun2, "rainfun2", exact=False)
    #FV.add_spatialflux(rainfun, "rainfun")
    FV.add_spatialflux(stateposfunc, "stateposfunc")
    
    FV.add_spatialflux(storage_change, "sc")
    for i in range(runs):
        prevstate = FV.states_to_function()
        FV.solve(rmse_threshold=1e-10)
        plt.plot(FV.nodes, FV.states, "-o", color="green", label="FV")
    
    #plt.plot(FV.nodes, FV.states, color="black", label="FV")
    print(sum(FV.states))
    print("")
    FV.calcbalance()
    print("")
    print(FV)
        
################################ UNSTRUCTURED #################################

    FVu = Flow1DFV("unstructured")
    xsp, _ = spacing(nx, L, linear=False, loc=[4, 7], power=2, weight=10)
    FVu.set_field1d(array=xsp)
    FVu.set_systemfluxfunction(fluxfunction)
    FVu.set_initial_states(4.90)
    
    FVu.add_dirichlet_BC(5, "west")
    FVu.add_dirichlet_BC(5.01, "east")
    #FVu.add_neumann_BC(0.01, "east")
    
    FVu.add_pointflux([-0.027, -0.015], [5.0, 7.8], "well!")
    FVu.add_pointflux(Spointflux, 6.2, "Swell!")
    
    FVu.add_spatialflux(rainfunc, "rainfunc")
    FVu.add_spatialflux(0.003, "recharge")
    FVu.add_spatialflux(stateposfunc, "stateposfunc")
    
    FVu.add_spatialflux(storage_change, "sc")
    for i in range(runs):
        prevstate = FVu.states_to_function()
        FVu.solve(rmse_threshold=1e-10)
        plt.plot(FVu.nodes, FVu.states, "-o", color="magenta", label="FVu")
    
    #plt.plot(FVu.nodes, FVu.states, color="black", label="FVu")
    print(sum(FVu.states))
    print("")
    FVu.calcbalance()
    print("")
    print(FVu)
    
    plt.title('Finite Volumes')
