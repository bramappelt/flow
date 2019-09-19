from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from waterflow.flow1d.flowFE1d import Flow1DFE
from waterflow.utility import forcingfunctions as Ffunc
from waterflow.utility.spacing import spacing


# ###############################  MODEL INPUT ################################

L = 20
nx = 11
domain = [L, nx]
S = 1.0
dt = 1.0
runs = 1

ksat = 1.5


def kfun(x, s):
    return ksat + 0.0065*x


# ############################## FLUXFUNCTIONS ################################


def fluxfunction(x, s, gradient):
    return -1 * gradient * ksat


def fluxfunction_s(x, s, gradient):
    return -1 * gradient * ksat * s


def fluxfunction_var_k(x, s, gradient, kfun):
    return - kfun(x, s) * gradient


def fluxfunction_var_k_s(x, s, gradient, kfun):
    return - kfun(x, s) * s * gradient


# ############################## POINT FLUXES #################################

Spointflux = partial(np.interp, xp=[4.9, 4.95, 5.005],
                     fp=[-0.02, -0.029, -0.034])

# ############################## SPATIAL FLUXES ###############################


def rainfun(x):  # !! inspect new poly scaled??
    return 1.98e-5*x**3 - 7.34e-4 * x**2 + 5.36e-3 * x


def rainfunc(x):
    return x*0.001 + 0.001


a2, x2, b2, rainfun2 = Ffunc.polynomial([[0, 0.001], [5, 0.003], [10, 0.005],
                                        [15, 0.003], [20, 0.002]])


def storage_change(x, s, prevstate, dt, fun=lambda x: x, S=1):
    return - S * (fun(s) - fun(prevstate(x))) / dt


def stateposfunc(x, s):
    return (-2 + np.sin(x) + s) / 1000


# ################################ STRUCTURED #################################

FE = Flow1DFE("structured")
FE.set_field1d(linear=domain)
FE.set_systemfluxfunction(fluxfunction_var_k, kfun=kfun)
FE.set_initial_states(4.90)

FE.add_dirichlet_BC(5.0, "west")
# FE.add_dirichlet_BC(5.01, "east")
# FE.add_neumann_BC(0.01, "east")

FE.add_pointflux([-0.027, -0.015], [4.0, 8.0])
FE.add_pointflux(-0.02, 6.0)
FE.add_pointflux(Spointflux, 5.0, "Swell!")

FE.add_spatialflux(0.003, 'recharge')
FE.add_spatialflux(rainfun2, 'rainfun2')
FE.add_spatialflux(rainfunc)
FE.add_spatialflux(rainfun, "rainfun")
FE.add_spatialflux(stateposfunc, "spf")

FE.solve()

FE.transient_data()
FE.transient_dataframeify(invert=False, nodes=[0, 10, 20])
FE.save(dirname='sat_structured')

# plotting
fig, ax = plt.subplots()
for i, v in enumerate(FE.solve_data['solved_objects']):
    if i == 0:
        intrp_states = v.states_to_function()
        ax.plot(v.nodes, intrp_states(v.nodes), '.-')
    else:
        ax.plot(v.nodes, v.states, '.-')

ax.set_ylabel('Heads (m)')
ax.set_xlabel('Distance (m)')
ax.set_title(FE.id)
ax.grid()

# ############################### UNSTRUCTURED ################################

FEu = Flow1DFE("unstructured")
xsp, _ = spacing(nx, L, linear=False, loc=[4, 7], power=2, weight=10)
FEu.set_field1d(array=xsp)
FEu.set_systemfluxfunction(fluxfunction_var_k, kfun=kfun)
FEu.set_initial_states(4.90)

FEu.add_dirichlet_BC(5, "west")
# FEu.add_dirichlet_BC(5.01, "east")
# FEu.add_neumann_BC(0.01, "east")

FEu.add_pointflux([-0.027, -0.015], [4.0, 8.0], "well!")
FEu.add_pointflux(-0.02, 6.0, "well1")
FEu.add_pointflux(Spointflux, 5.0, "Swell!")

FEu.add_spatialflux(0.003, "recharge")
FEu.add_spatialflux(rainfunc, "rainfunc")
FEu.add_spatialflux(rainfun2, "rainfun2", exact=False)
FEu.add_spatialflux(rainfun, "rainfun")
FEu.add_spatialflux(stateposfunc, "stateposfunc")

FEu.solve()
FEu.transient_data()
FEu.transient_dataframeify(invert=False)
FEu.save(dirname='sat_unstructured')

# plotting
fig, ax = plt.subplots()
for i, v in enumerate(FEu.solve_data['solved_objects']):
    if i == 0:
        intrp_states = v.states_to_function()
        ax.plot(v.nodes, intrp_states(v.nodes), '.-')
    else:
        ax.plot(v.nodes, v.states, '.-')

ax.set_ylabel('Heads (m)')
ax.set_xlabel('Distance (m)')
ax.set_title(FEu.id)
ax.grid()


# ################################ TRANSIENT ##################################

FEut = Flow1DFE("unstructured & transient")
FEut.scheme = 'linear'
xsp, _ = spacing(nx, L, linear=False, loc=[4, 7], power=2, weight=10)
FEut.set_field1d(array=xsp)
FEut.set_systemfluxfunction(fluxfunction_var_k, kfun=kfun)
FEut.set_initial_states(4.9)

FEut.add_dirichlet_BC(5, "west")
# FEut.add_dirichlet_BC(5.01, "east")
# FEut.add_neumann_BC(0.1, "east")

FEut.add_pointflux([-0.027, -0.015], [4.0, 8.0], "well!")
FEut.add_pointflux(-0.02, 6.0, "well1")
FEut.add_pointflux(Spointflux, 5, 'Swell!')

FEut.add_spatialflux(0.003, "recharge")
FEut.add_spatialflux(rainfunc, "rainfunc")
FEut.add_spatialflux(rainfun2, "rainfun2")
FEut.add_spatialflux(rainfun, "rainfun")
FEut.add_spatialflux(stateposfunc, "stateposfunc")
FEut.add_spatialflux(storage_change)

FEut.solve(end_time=100, dt_max=5, threshold=1e-3)
FEut.transient_data(print_times=10)
FEut.transient_dataframeify(invert=False, nodes=[0, 10, 20])
FEut.save(dirname='sat_transient')


# plotting
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
solve_data = FEut.solve_data

for i, v in enumerate(solve_data['solved_objects']):
    if i == 0:
        intrp_states = v.states_to_function()
        ax1.plot(v.nodes, intrp_states(v.nodes), '.-')
    else:
        ax1.plot(v.nodes, v.states, '.-')

# find steady state
FEut.remove_spatialflux('storage_change')
FEut.solve()
ax1.plot(FEut.nodes, FEut.states, color='black')

ax1.set_xlabel('distance (m)')
ax1.set_ylabel('heads (m)')
ax1.set_title('Hydraulic heads')
ax1.grid(True)

ax2.plot(solve_data['time'], solve_data['dt'], '.-', color='green')
ax2.set_xlabel('time (d)')
ax2.set_ylabel('dt (d)')
ax2.grid(True)

ax3.plot(solve_data['time'], solve_data['iter'], '.-', color='blue')
ax3.set_xlabel('time (d)')
ax3.set_ylabel('iterations (-)')
ax3.grid(True)

ax4.plot(solve_data['time'][1:], np.cumsum(solve_data['iter'][1:]), '.-',
         color='red')
ax4.set_xlabel('time (d)')
ax4.set_ylabel('cumulative dt (d)')
ax4.grid(True)

fig.suptitle(FEut.id)
plt.show()
