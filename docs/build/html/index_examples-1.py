from waterflow.flow1d.flowFE1d import Flow1DFE
from waterflow.utility import conductivityfunctions as condf
from waterflow.utility import fluxfunctions as fluxf
from waterflow.utility.helper import initializer
from waterflow.utility.spacing import biasedspacing
from waterflow.utility.plotting import quickplot, solverplot
soil, *_ = condf.soilselector([13])[0]
theta_r, theta_s, ksat, alpha, n = (soil.t_res, soil.t_sat, soil.ksat, soil.alpha, soil.n)
L = 100
nx = 51
xsp = biasedspacing(nx, power=1, rb=-L)[::-1]
initial_states = np.repeat(0, nx)
theta_h = initializer(condf.VG_pressureh, theta_r=theta_r, theta_s=theta_s, a=alpha, n=n)
conductivity_func = initializer(condf.VG_conductivity, ksat=ksat, a=alpha, n=n)
storage_change = initializer(fluxf.storage_change, fun=theta_h)
M = Flow1DFE('Unsaturated transient model')
M.set_field1d(nodes=xsp)
M.set_gaussian_quadrature(2)
M.set_initial_states(initial_states)
M.set_systemfluxfunction(fluxf.richards_equation, kfun=conductivity_func)
M.add_dirichlet_BC(0.0, 'west')
M.add_neumann_BC(-0.3, 'east')
M.add_spatialflux(-0.01, 'extraction')
M.add_pointflux(-0.03, -5.5, 'pflux')
M.add_spatialflux(storage_change)
M.tfun = theta_h
M.solve(dt=0.01, end_time=15)
M.transient_dataframeify()
solverplot(M)
plt.tight_layout()
plt.show()