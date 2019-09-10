import sys
sys.path.insert(1, 'C:\\Users\\bramb\\Documents\\thesis\\flow')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d
from copy import deepcopy

from waterflow.flow1d.flowFE1d import Flow1DFE
from waterflow.utility import conductivityfunctions as CF
from waterflow.utility import fluxfunctions as FF
from waterflow.utility import forcingfunctions as Ffunc
from waterflow.utility.spacing import spacing


############################## MODEL INPUT ##################################


SOIL = CF.select_soil('B13')
L = 100
nx = 101
xsp, _ = spacing(nx, L)
xsp = xsp - 100
initial_states = np.repeat(0, nx)

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

def VG_conductivity(h, ksat=ksat, a=a, n=n):
    if h >= 0:
        return ksat
    m = 1-1/n
    h_up = (1 - (a * -h)**(n-1) * (1 + (a * -h)**n)**-m)**2
    h_down = (1 + (a * -h)**n)**(m / 2)
    return (h_up / h_down) * ksat

def richards_equation(x, psi, gradient, kfun=VG_conductivity):
    return -VG_conductivity(psi) * (gradient + 1)

def storage_change2(x, s):
    return -(VG_pressureh(s) - VG_pressureh(prevstate(x))) / dt

def storage_change(x, s, prevstate, dt):
    return -(VG_pressureh(s) - VG_pressureh(prevstate(x))) / dt


############################ SOLVE TRANSIENT ################################

FE_ut = Flow1DFE('Unsaturated transient model')
FE_ut.scheme = 'linear'
FE_ut.set_field1d(array=xsp)
FE_ut.set_initial_states(initial_states)
FE_ut.set_systemfluxfunction(richards_equation)
FE_ut.add_dirichlet_BC(0, 'west')

# think about dirichlet plot init
FE_ut.add_neumann_BC(-0.1, 'east')

FE_ut.add_spatialflux(storage_change, 'storage_change')
FE_ut.solve(transient=True, dt_max=2, end_time=4)

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
solve_data = FE_ut.solve_data

for k, v in solve_data['solved_states'].items():
    ax1.plot(v.states, v.nodes)

ax2.plot(solve_data['time_data'], solve_data['dt_data'], '.-', color='green')
ax3.plot(solve_data['time_data'], solve_data['iter_data'], '.-', color='blue')
ax4.plot(solve_data['time_data'][1:], np.cumsum(solve_data['iter_data'][1:]), '.-', color='red')

plt.grid()
plt.show()

