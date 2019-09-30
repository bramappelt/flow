import numpy as np
import matplotlib.pyplot as plt

from waterflow.flow1d.flowFE1d import Flow1DFE
from waterflow.utility import conductivityfunctions as condf
from waterflow.utility import fluxfunctions as fluxf
from waterflow.utility.helper import initializer
from waterflow.utility.plotting import quickplot, solverplot
from waterflow.utility.spacing import biasedspacing


# ############################# MODEL INPUT ##################################


soil = condf.soilselector('B13')
soiltype, theta_r, theta_s, ksat, alpha, Lambda, n, name = soil

L = 100
nx = 101
xsp = biasedspacing(nx, power=3, length=100) - L
initial_states = np.repeat(0, nx)

theta_h = initializer(condf.VG_pressureh, theta_r=theta_r, theta_s=theta_s,
                      a=alpha, n=n)
conductivity_func = initializer(condf.VG_conductivity, ksat=ksat, a=alpha, n=n)
storage_change = initializer(fluxf.storage_change, fun=theta_h)


# ########################### SOLVE TRANSIENT ################################

FE_ut = Flow1DFE('Unsaturated transient model')
FE_ut.scheme = 'linear'
FE_ut.set_field1d(array=xsp)
FE_ut.set_initial_states(initial_states)
FE_ut.set_systemfluxfunction(fluxf.richards_equation, kfun=conductivity_func)
FE_ut.add_dirichlet_BC(0.0, 'west')
FE_ut.add_neumann_BC(-0.5, 'east')
FE_ut.tfun = theta_h

FE_ut.add_spatialflux(storage_change)

FE_ut.solve(dt_min=0.01, dt_max=5, end_time=20, maxiter=500, 
            dtitlow=1.5, dtithigh=0.5, itermin=5, itermax=10,
            verbosity=False)

FE_ut.transient_dataframeify(nodes=[0, -20, -50, -80, -100], print_times=50)

FE_ut.save(dirname='unsat_transient')


fig, ax = plt.subplots()
for i in FE_ut.dft_states.keys():
    quickplot(FE_ut.dft_states[i], x='states', y=['nodes'], ax=ax, title='Hydraulic heads over time (d)',
              xlabel='states [cm]', ylabel='nodes [cm]', legend=False, grid=True)

fig, ax = plt.subplots()
for i in FE_ut.dft_nodes.keys():
    quickplot(FE_ut.dft_nodes[i], x='time', y=['states'], ax=ax, title='Hydraulic heads at specific nodes over time (d)', legend=False)

fig, ax = plt.subplots()
quickplot(FE_ut.dft_solved_times, x='time', y=['dt'], ax=ax, title='Solver')

fig, ax = plt.subplots()
quickplot(FE_ut.dft_balance_summary, x='time', y=['spat-storage_change'], ax=ax, title='storage change', legend=False)

solverplot(FE_ut)
plt.show()
