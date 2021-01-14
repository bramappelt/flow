# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
from waterflow.flow1d.flowFE1d import Flow1DFE
from waterflow.utility import fluxfunctions as fluxf
from waterflow.utility.spacing import spacing
from waterflow.utility.plotting import quickplot

# Define discretization
nx = 101
L = 100
# Focus nodal density around nodes 30 and 70
xsp, _ = spacing(nx, L, linear=False, loc=[30, 70], power=5, weight=4)

def kfun(x, s, ksat=7.5):
    """ Increasing conductivity to the right of the domain """
    return ksat + 0.0065*x

# Model defintion
M = Flow1DFE('Saturated transient model')
M.set_field1d(nodes=xsp)
M.set_initial_states(5.0)
M.set_systemfluxfunction(fluxf.darcy_s, kfun=kfun)
M.add_dirichlet_BC(5.0, 'west')
M.add_neumann_BC(0.0, 'east')
M.add_spatialflux(0.001, "recharge")
M.add_pointflux(-0.05, 30, 'pflux1')
M.add_pointflux(-0.07, 70, 'pflux2')
M.add_spatialflux(fluxf.storage_change)
M.solve(dt=0.01, dt_max=5, end_time=200)
M.transient_dataframeify(invert=False, print_times=15)

# Plotting
fig, ax = plt.subplots()
for key in M.dft_states.keys():
    quickplot(df=M.dft_states[key], x='nodes', y=['states'], ax=ax, xlabel='Distance (cm)',
    ylabel='Hydraulic heads (cm)', title='Hydraulic heads over time', legend=False)
plt.show()