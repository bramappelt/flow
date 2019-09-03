import numpy as np
import matplotlib.pyplot as plt

from waterflow.utility.statistics import RMSE, MAE


class Flow1DFD(object):
    def __init__(self, id_):
        self.id = id_
        self.systemfluxfunc = None
        self.nodes = None
        self.dx = None
        self.scheme = None
        self.states = None
        self.coefmatr = None
        self.coefmatrCN = None
        self.BCs = {}
        self.spatflux = {}
        self.pointflux = {}
        self.forcing = None
        self.accumulated_forcing = None
        self.stats = {"rmse" : [], "mae" : []}
        
    def __repr__(self):
        return "Flow1DFD("+str(self.id)+")"
        
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
        spatflux = ", ".join(i for i in self.spatflux.keys())
        pointflux = ", ".join(i for i in self.pointflux.keys())
        return "id: {}\n{}\n{}\nBCs: {}\nSpatial flux: {}\nPointflux: {}\
               \n".format(id, len_, num_nodes, bcs, spatflux, pointflux)
        
    def _aggregate_forcing(self):
        self.forcing = np.repeat(0.0, len(self.nodes))
        for flux in [self.spatflux, self.pointflux]:
            for key in flux.keys():
                self.forcing += flux[key]
        self.forcing[0] = 0
        self.forcing[-1] = 0
        self.accumulated_forcing = self.forcing.copy()
        
    def _aggregate_boundaries(self, mat="1"):
        coefmatrs = {"1" : self.coefmatr, "2" : self.coefmatrCN}
        for key in self.BCs.keys():
            val, type_, idx = self.BCs[key]
            if type_ == "Dirichlet":
                coefmatrs[mat][idx] = 0
                coefmatrs[mat][idx][idx] = 1
                self.states[idx] = val
            if type_ == "Neumann":
                coefmatrs[mat][idx] = 0
                coefmatrs[mat][idx][idx] = 1
                self.states[idx] = self.states[idx+int((-1)**idx)] + val
                
        keys = list(self.BCs.keys())
        if len(keys) == 0:
            raise np.linalg.LinAlgError("Singular matrix")
        
        if len(keys) == 1:
            bound = self.BCs[keys[0]]
            if bound[1] == "Dirichlet" and bound[2] == 0:
                coefmatrs[mat][-1] = 0
                coefmatrs[mat][-1][-1] = 1
                self.states[-1] = self.states[-2]
            elif bound[1] == "Dirichlet" and bound[2] == -1:
                coefmatrs[mat][0] = 0
                coefmatrs[mat][0][0] = 1
                self.states[0] = self.states[1]
            else:
                raise np.linalg.LinAlgError("Singular matrix") 
        
        if len(keys) == 2 and self.BCs[keys[0]][1] == "Neumann":
            if self.BCs[keys[0]][1] == self.BCs[keys[1]][1]:
                raise np.linalg.LinAlgError("Singular matrix")
    
    def _CMAT(self, nodes, states):
        dx2 = self.dx**2
        systemflux = self.systemfluxfunc
        A = np.zeros((len(nodes), len(nodes)))
        
        if self.scheme == "FTCS":
            for i in range(1, len(nodes) - 1):
                left = systemflux(nodes[i-1], states[i-1], -1)
                mid = systemflux(nodes[i], states[i], -1)
                right = systemflux(nodes[i+1], states[i+1], -1)
                scheme = [1*left/dx2, 1-(2*mid/dx2), 1*right/dx2]
                A[i, i-1] = scheme[0]
                A[i, i] = scheme[1]
                A[i, i+1] = scheme[2]
            
            self.coefmatr = A
            return A
                
        elif self.scheme == "BTCS":
            for i in range(1, len(nodes) - 1):
                left = systemflux(nodes[i-1], states[i-1], -1) / self.dx**2
                mid = systemflux(nodes[i], states[i], -1) / self.dx**2
                right = systemflux(nodes[i+1], states[i+1], -1) / self.dx**2
                scheme = [-1*left, 1+2*mid, -1*right]
                A[i, i-1] = scheme[0]
                A[i, i] = scheme[1]
                A[i, i+1] = scheme[2]

            self.coefmatr = A
            return A
        
        elif self.scheme == "CN":
            for i in range(1, len(nodes) - 1):
                left = systemflux(nodes[i-1], states[i-1], -1) / self.dx**2
                mid = systemflux(nodes[i], states[i], -1) / self.dx**2
                right = systemflux(nodes[i+1], states[i+1], -1) / self.dx**2
                scheme1 = [-1*left, 2+2*mid, -1*right]
                A[i, i-1] = scheme1[0]
                A[i, i] = scheme1[1]
                A[i, i+1] = scheme1[2]
                
            B = np.zeros((len(nodes), len(nodes)))
            for i in range(1, len(nodes) - 1):
                left = systemflux(nodes[i-1], states[i-1], -1) / self.dx**2
                mid = systemflux(nodes[i], states[i], -1) / self.dx**2
                right = systemflux(nodes[i+1], states[i+1], -1) / self.dx**2
                scheme2 = [1*left, 2-2*mid, 1*right]
                B[i, i-1] = scheme2[0]
                B[i, i] = scheme2[1]
                B[i, i+1] = scheme2[2]
        
            self.coefmatr = A
            self.coefmatrCN = B
            return A, B
    
    def set_field1d(self, **kwargs):
        for key in kwargs.keys():
            if key == "linear":
                self.nodes = np.linspace(0, kwargs[key][0], kwargs[key][1])
                self.states = np.repeat(0.1, len(self.nodes))
                self.dx = kwargs[key][0] / (kwargs[key][1] - 1)
 
    def set_systemfluxfunction(self, function):
        # add some default functions here (darcy for instance)
        self.systemfluxfunc = function
        
    def set_initial_states(self, states):
        if isinstance(states, int) or isinstance(states, float):
            self.states = np.array([states for x in range(len(self.nodes))])
        else:
            self.states = np.array(states)
            
    def add_dirichlet_BC(self, value, where):
        if isinstance(value, int) or isinstance(value, float):
            value = [value]
            where = [where]

        for val, pos in zip(value, where):
            if pos.lower() in "western":
                self.BCs["west"] = (val, "Dirichlet", 0)
            elif pos.lower() in "eastern":
                self.BCs["east"] = (val, "Dirichlet", -1)
      
    def add_neumann_BC(self, value, where):
        if isinstance(value, int) or isinstance(value, float):
            value = [value]
            where = [where]
            
        for val, pos in zip(value, where):
            if pos.lower() in "western":
                self.BCs["west"] = (val, "Neumann", 0)
            elif pos.lower() in "eastern":
                self.BCs["east"] = (val, "Neumann", -1)
                
    def remove_BC(self, *args):
        if len(args) == 0:
            self.BCs  = {}
        else:
            for name in args:
                try:
                    self.BCs.pop(name)
                except KeyError as e:
                    raise type(e)("No boundary named " + str(name) + ".")
                    
    def add_spatialflux(self, q, name):
        def dummyfun(): pass
        if isinstance(q, int) or isinstance(q, float):
            self.spatflux[name] = np.repeat(q, len(self.nodes))
        elif isinstance(q, list) or isinstance(q, np.ndarray):
            self.spatflux[name] = np.array(q)
        elif type(q) == type(dummyfun):
            self.spatflux[name] = np.array(list(map(q, self.nodes)))
            
    def add_pointflux(self, rate, pos, name):
        if isinstance(rate, int) or isinstance(rate, float):
            rate = [rate]
            pos = [pos]
            
        f = [0.0 for x in range(len(self.nodes))]
        for r, p in zip(rate, pos):
            f[np.abs(self.nodes - p).argmin()] = r / self.dx
        self.pointflux[name] = np.array(f)
            
    def remove_spatialflux(self, *args):
        if len(args) == 0:
            self.spatflux = {}
        else:
            for name in args:
                try:
                    self.spatflux.pop(name)
                except KeyError as e:
                    raise type(e)(str(name) + " is not a spatialflux.")
                    
    def remove_pointflux(self, *args):
        if len(args) == 0:
            self.pointflux = {}
        else:
            for name in args:
                try:
                    self.pointflux.pop(name)
                except KeyError as e:
                    raise type(e)(str(name) + " is not a pointflux.")
        
    def solve(self, scheme, maxiter=5000, rmse_threshold=1e-8, mae_threshold=1e-8):
        self.stats = {"rmse" : [], "mae" : []}
        self.scheme = scheme

        # solves using the forward in time central spatial scheme (FTCS)
        # unstable for R > 0.5
        iter_step = 1
        while iter_step <= maxiter:
            if scheme == "FTCS":
                self._CMAT(self.nodes, self.states)
                self._aggregate_forcing()
                self._aggregate_boundaries()
        
                sol_p1 = np.matmul(self.coefmatr, self.states) 
                sol_p2 = np.reshape(self.forcing, (len(self.nodes), ))
                solution = sol_p1 + sol_p2
            
            elif scheme == "BTCS":
                self._CMAT(self.nodes, self.states)
                self._aggregate_forcing()
                self._aggregate_boundaries()
                
                sol_p1 = np.linalg.solve(self.coefmatr, self.states)
                sol_p2 = np.reshape(self.forcing, (len(self.nodes), ))
                solution = sol_p1 + sol_p2
            
            elif scheme == "CN":
                self._aggregate_forcing()                
                self._CMAT(self.nodes, self.states)
                self._aggregate_boundaries()
                self._aggregate_boundaries(mat="2")
                A = self.coefmatr
                B = self.coefmatrCN
                
                C = np.matmul(B, self.states)
                sol_p1 = np.linalg.solve(A, C)
                sol_p2 = np.reshape(self.forcing, (len(self.nodes), ))
                solution = sol_p1 + sol_p2
            
            rmse = RMSE(self.states, solution)
            mae = MAE(self.states, solution)
        
            if rmse < rmse_threshold:
                self.states = solution
                self.stats["rmse"].append(rmse)
                self.stats["mae"].append(mae)
                break
            
            if mae < mae_threshold:
                self.states = solution
                self.stats["rmse"].append(rmse)
                self.stats["mae"].append(mae)
                break
            
            self.stats["rmse"].append(rmse)
            self.stats["mae"].append(mae)
            
            self.states = solution
            iter_step += 1
    
    
if __name__ == "__main__":
    L = 20
    nx = 11
    domain = [L, nx]
    
    def kfun(x):
        return ksat + 0.0065*x
    
    ksat = 1.5
    def fluxfunction(x, s, gradient):
        return -ksat * gradient
    
    def fluxfunction_var_k(x, s, gradient):
        return -kfun(x) * gradient
    
    def fluxfunction_s(x, s, gradient):
        return -ksat * s * gradient
    
    FD = Flow1DFD("structured")
    FD.set_field1d(linear=domain)
    FD.set_systemfluxfunction(fluxfunction_var_k)
    #FD.k = ksat
    
    FD.add_dirichlet_BC(5, "west")
    FD.add_dirichlet_BC(5.01, "east")
    #FD.add_neumann_BC(0.007, "east")
    
    FD.add_spatialflux(0.003, "recharge")
    FD.add_pointflux(-0.05, 10, "well1")
    
    #FD.add_pointflux(-0.03, 11.001, "well1")
    
    FD.solve("FTCS")
    plt.plot(FD.nodes, FD.states, label = "FTCS")
    FD.solve("CN")
    plt.plot(FD.nodes, FD.states, label="BTCS")
    FD.solve("CN")
    plt.plot(FD.nodes, FD.states, label="CN")
    
    plt.grid()
    plt.legend()
    
    
    