import numpy as np
import matplotlib.pyplot as plt

from waterflow.flow1d.flowFE1d import Flow1DFE
from waterflow.utility import conductivityfunctions as CF
from waterflow.utility.plotting import quickplot, solverplot
from waterflow.utility.spacing import spacing


# ############################# MODEL INPUT ##################################


SOIL = CF.select_soil('B13')
L = 100
nx = 101
xsp, _ = spacing(nx, L)
xsp = xsp - 100
initial_states = np.repeat(0, nx)
# initial_states = np.linspace(0, -100, 101)

theta_r = SOIL['theta.res']
theta_s = SOIL['theta.sat']
a = SOIL['alpha']
n = SOIL['n']
ksat = SOIL['ksat']
name = SOIL['name']


def VG_pressureh(h, theta_r=theta_r, theta_s=theta_s, a=a, n=n):
    # to theta
    if h >= 0:
        return theta_s
    m = 1-1/n
    return theta_r + (theta_s-theta_r) / (1+(a*-h)**n)**m


def VG_conductivity(x, h, ksat=ksat, a=a, n=n):
    if h >= 0:
        return ksat
    m = 1-1/n
    h_up = (1 - (a * -h)**(n-1) * (1 + (a * -h)**n)**-m)**2
    h_down = (1 + (a * -h)**n)**(m / 2)
    return (h_up / h_down) * ksat


def richards_equation(x, s, gradient, kfun):
    return -kfun(x, s) * (gradient + 1)


def storage_change(x, s, prevstate, dt, fun=VG_pressureh, S=1):
    return - S * (fun(s) - fun(prevstate(x))) / dt


# ########################### SOLVE TRANSIENT ################################

FE_ut = Flow1DFE('Unsaturated transient model')
FE_ut.scheme = 'quintic'
FE_ut.set_field1d(array=xsp)
FE_ut.set_initial_states(initial_states)
FE_ut.set_systemfluxfunction(richards_equation, kfun=VG_conductivity)
FE_ut.add_dirichlet_BC(-0.5, 'west')
FE_ut.add_neumann_BC(-0.1, 'east')
FE_ut.tfun = VG_pressureh

FE_ut.add_spatialflux(storage_change)

FE_ut.solve(dt_min=0.01, dt_max=2, end_time=10, maxiter=500)

FE_ut.transient_dataframeify(nodes=[0, -20, -50, -80, -100], print_times=50)

FE_ut.save(dirname='unsat_transient')

quickplot(FE_ut.dft_states, x='states', y='nodes', title='Hydraulic heads over time (d)', xunit='cm', yunit='cm')
quickplot(FE_ut.dft_nodes, x='time', y='states', title='Hydraulic heads at specific nodes over time (d)', xunit='days', yunit='cm')
quickplot(FE_ut.dft_solved_times, x='time', y='dt', title='Solver', xunit='d', yunit='d', y2='iter')
quickplot(FE_ut.dft_balance_summary, x='time', y='spat-storage_change', xunit='d', yunit='', title='storage change')

solverplot(FE_ut)
plt.show()
