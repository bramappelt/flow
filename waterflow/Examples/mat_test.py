import numpy as np
import matplotlib.pyplot as plt

from waterflow.flow1d.flowFE1d import Flow1DFE
from waterflow.utility import conductivityfunctions as condf
from waterflow.utility import fluxfunctions as fluxf
from waterflow.utility.helper import initializer
from waterflow.utility.plotting import quickplot, solverplot
from waterflow.utility.spacing import biasedspacing


# ############################# MODEL INPUT ##################################


soil, *_ = condf.soilselector([13])[0]
theta_r, theta_s, ksat, alpha, n = (soil.t_res, soil.t_sat, soil.ksat, soil.alpha, soil.n)

L = 10
nx = 11
xsp = biasedspacing(nx, power=1, rb=-L)[::-1]
initial_states = np.repeat(0, nx)

theta_h = initializer(condf.VG_pressureh, theta_r=theta_r, theta_s=theta_s,
                      a=alpha, n=n)
conductivity_func = initializer(condf.VG_conductivity, ksat=ksat, a=alpha, n=n)
storage_change = initializer(fluxf.storage_change, fun=theta_h)

FE_ut = Flow1DFE('Unsaturated transient model')
FE_ut.set_field1d(nodes=xsp, degree=1)
# FE_ut.set_gaussian_quadrature(2)
FE_ut.set_initial_states(initial_states)
FE_ut.set_systemfluxfunction(fluxf.richards_equation, kfun=conductivity_func)
FE_ut.add_dirichlet_BC(0.0, 'west')
# FE_ut.add_neumann_BC(0.0, 'east')
FE_ut.tfun = theta_h

# manual
DELTA = 1e-5                            # d
systemflux = FE_ut.systemfluxfunc       # SF(x_p, s_i, g_i)
nodes = FE_ut.nodes                     # n_i
states = FE_ut.states                   # s_i
A = np.zeros((len(nodes), len(nodes)))  # A

for i in range(0, len(nodes)-1):                                                   # i
    L = nodes[i+1] - nodes[i]                               # dx_i
    stateleft = states[i] + np.array([0, DELTA, 0])         # sl_i
    stateright = states[i+1] + np.array([0, 0, DELTA])      # sr_i
    grad = (stateright - stateleft) / L                     # grad_i

    totflux = np.array([0, 0, 0], dtype=np.float64)         
    pos, weight = FE_ut.gaussquad

    # ####
    idx = 0     # range(0, len(pos)) number of integration points
    x = FE_ut.xintegration[i][idx]
    state_x = stateleft * pos[-idx-1] + stateright * pos[idx]

    flux = [systemflux(x, s, g) for s, g in zip(state_x, grad)]
    totflux += np.array(flux) * weight[idx]
    # ####

    dfluxl = (totflux[1] - totflux[0]) / DELTA
    dfluxr = (totflux[2] - totflux[0]) / DELTA

    A[i][i] += -dfluxl
    A[i+1][i] += dfluxl
    A[i][i+1] += -dfluxr
    A[i+1][i+1] += dfluxr

west = 1            # Dirichlet (-1)
east = len(nodes)   # Neumann   (len(nodes))

# cycle 1
FE_ut._internal_forcing()
F = FE_ut.forcing
S = np.linalg.solve(A[west:east, west:east], -1*F[west:east])
states[west:east] += S
print(states)

# Cycle 2
A = np.zeros((len(nodes), len(nodes)))  # A
for i in range(0, len(nodes)-1):                                                   # i
    L = nodes[i+1] - nodes[i]                               # dx_i
    stateleft = states[i] + np.array([0, DELTA, 0])         # sl_i
    stateright = states[i+1] + np.array([0, 0, DELTA])      # sr_i
    grad = (stateright - stateleft) / L                     # grad_i

    totflux = np.array([0, 0, 0], dtype=np.float64)         
    pos, weight = FE_ut.gaussquad

    # ####
    idx = 0     # range(0, len(pos)) number of integration points
    x = FE_ut.xintegration[i][idx]
    state_x = stateleft * pos[-idx-1] + stateright * pos[idx]

    flux = [systemflux(x, s, g) for s, g in zip(state_x, grad)]
    totflux += np.array(flux) * weight[idx]
    # ####

    dfluxl = (totflux[1] - totflux[0]) / DELTA
    dfluxr = (totflux[2] - totflux[0]) / DELTA

    A[i][i] += -dfluxl
    A[i+1][i] += dfluxl
    A[i][i+1] += -dfluxr
    A[i+1][i+1] += dfluxr

FE_ut.states = states
FE_ut.forcing = np.zeros((len(nodes)))
FE_ut._internal_forcing()
F = FE_ut.forcing
S = np.linalg.solve(A[west:east, west:east], -1*F[west:east])
states[west:east] += S
print(states)