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


# ########################### SOLVE TRANSIENT ################################

FE_ut = Flow1DFE('Unsaturated transient model')
FE_ut.scheme = 'linear'
FE_ut.set_field1d(array=xsp)
FE_ut.set_initial_states(initial_states)
FE_ut.set_systemfluxfunction(fluxf.richards_equation, kfun=conductivity_func)
FE_ut.add_dirichlet_BC(0.0, 'west')
FE_ut.add_neumann_BC(-0.3, 'east')
FE_ut.tfun = theta_h

FE_ut.add_spatialflux(-0.001, 'extraction')

FE_ut.add_spatialflux(storage_change)

FE_ut.solve(dt_min=1e-5, dt_max=5, end_time=100, maxiter=500,
            dtitlow=1.5, dtithigh=0.5, itermin=3, itermax=7,
            verbosity=True)

FE_ut.transient_dataframeify(nodes=[0, -2, -5, -8, -10])

FE_ut.save(dirname='wbal_testing')


fig, ax = plt.subplots()
for i in FE_ut.dft_states.keys():
    quickplot(FE_ut.dft_states[i], x='states', y=['nodes'], ax=ax, title='Hydraulic heads over time (d)',
              xlabel='states [cm]', ylabel='nodes [cm]', legend=False,
              grid=True, save=True, filename='quickplot.png', marker='o')

fig, ax = plt.subplots()
for i in FE_ut.dft_nodes.keys():
    quickplot(FE_ut.dft_nodes[i], x='time', y=['states'], ax=ax, title='Hydraulic heads at specific nodes over time (d)', legend=False)

fig, ax = plt.subplots()
quickplot(FE_ut.dft_solved_times, x='time', y=['dt'], ax=ax, title='Solver')

fig, ax = plt.subplots()
quickplot(FE_ut.dft_balance_summary, x='time', y=['spat-storage_change'], ax=ax, title='storage change', legend=False)

fig, ax = plt.subplots()
for k, v in FE_ut.dft_balance.items():
    quickplot(v, x='fluxes', y=['nodes'], ax=ax, legend=False)

solverplot(FE_ut, save=True, filename='test.png')
plt.show()

