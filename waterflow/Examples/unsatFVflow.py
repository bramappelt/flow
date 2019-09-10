import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d
from copy import deepcopy

from waterflow.flow1d.flowFV1d import Flow1DFV
from waterflow.utility import conductivityfunctions as CF
from waterflow.utility import fluxfunctions as FF
from waterflow.utility import forcingfunctions as Ffunc
from waterflow.utility.spacing import spacing


############################## MODEL INPUT ##################################
all_soiltypes = [soiltype for soiltype in CF.STARINGREEKS["soiltype"]]
for soil in all_soiltypes[:1]:
    SOIL = CF.select_soil(soil)
    L = 200
    nx = 201
    xsp, _ = spacing(nx, L)
    xsp = xsp - 200
    initial_states = np.linspace(L*0.5, -L*0.5, nx)
    dt = 1.0
    
    theta_r = SOIL['theta.res']
    theta_s = SOIL['theta.sat']
    a = SOIL['alpha']
    n = SOIL['n']
    ksat = SOIL['ksat']
    name = SOIL['name']
    
    def VG_theta(theta, theta_r=theta_r, theta_s=theta_s, a=a, n=n):
        # to head
        m = 1-1/n
        THETA = (theta_s - theta_r) / (theta - theta_r)
        return ((THETA**(1/m) - 1) / a**n)**(1/n)
    
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
    
    def internalflux(x, s, gradient):
        #H = h + z >>> h = (H - z)*100 (from m to cm, z = x here)
        return -VG_conductivity(100*(s - x)) * gradient
    
    def richards_equation(x, psi, gradient, kfun=VG_conductivity):
        return -kfun(psi) * (gradient + 1)
    
    def storage_change(x, s):
        return -(VG_pressureh(s) - VG_pressureh(prevstate(x))) / dt
    
    def max_capillary_rise(h, z, kfun):
        return -kfun(h) / z - kfun(h)
    
    def phreatic_surface(states, nodes, x=0.0):
        func = interp1d(states, nodes)
        return float(func(x))
    
    ############################# STATIONARY MODEL ###############################

    FV = Flow1DFV("unsaturated")
    FV.set_field1d(array=xsp)
    
    FV.set_initial_states(initial_states)
    FV.set_systemfluxfunction(richards_equation)
    
    FV.add_dirichlet_BC(100, "west")
    FV.add_neumann_BC(1.0, "east")
    #FV.add_dirichlet_BC(-40, "east")
    
    FV.add_pointflux(-0.5, -50.5, "stupidwell")
    
    FV.solve(rmse_threshold=1e-12)
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True)
    ax1.plot(FV.states, FV.nodes, "-o")
    ax1.plot(initial_states, FV.nodes, ls="dashed")
    ax1.set_xlabel("pressure head h (cm)")
    ax1.set_ylabel("Depth below surface (cm)")
    ax1.grid()
    
    theta_s = np.array([VG_pressureh(h) for h in initial_states])
    ax2.plot([VG_pressureh(h) for h in FV.states], FV.nodes)
    ax2.plot(theta_s, FV.nodes, ls="dashed")
    ax2.set_xlabel("Theta volumetric water content (-)")
    ax2.set_xlim(0, max(theta_s)+0.02)
    ax2.grid()
    
    ax3.plot([VG_conductivity(h) for h in FV.states] , FV.nodes)
    ax3.set_xlabel("Hydraulic conductivity (cm/d)")
    ax3.grid()
    
    fig.suptitle(name + " FV")
    
    print(sum(FV.states))
    print("")
    FV.calcbalance(print_=True)
    print("")
    print(FV)
    
    ############################ TRANSIENT MODEL ##############################
    '''
    FVt = Flow1DFV("transient")
    FVt.set_field1d(array=xsp)
    FVt.set_systemfluxfunction(richards_equation)
    FVt.set_initial_states(initial_states)
    FVt.add_dirichlet_BC(100, "west")
    #FVt.add_neumann_BC(0.1, "east")
    FVt.add_spatialflux(storage_change, "sc")
    
    figt, (axt, axt2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    axt.plot(initial_states, FVt.nodes, ls="dashed")
    
    theta_t = np.array([VG_pressureh(h) for h in initial_states])
    axt2.plot(theta_t, FVt.nodes, ls="dashed")
    
    gwl = []
    object_list = [deepcopy(FVt)]
    rain = [(t, value) for t, value in Ffunc.rain_generator(maxtime=300)]
    for t, r in rain:
        prevstate = FVt.states_to_function()
        FVt.add_neumann_BC(r, "east")
        FVt.solve(rmse_threshold=1e-10, mae_threshold=1e-10)
        FVt.waterbalance()
        
        axt.plot(FVt.states, FVt.nodes, "-o")
        
        new_theta = np.array([VG_pressureh(h) for h in FVt.states])
        axt2.plot(new_theta, FVt.nodes, "-o")
        object_list.append(deepcopy(FVt))
        gwl.append([t, phreatic_surface(FVt.states, FVt.nodes)])
    
    object_dict = dict(enumerate(object_list))
        
    figt.suptitle(name + " -transient")
    axt.set_xlabel("pressure head h (cm)")
    axt.set_ylabel("Depth below surface (cm)")
    axt.grid()
    axt2.set_xlabel("Theta volumetric water content (-)")
    axt2.set_xlim(0, max(theta_t)+0.02)
    axt2.grid()
    
################################## PLOTTING ###################################
    
    # Plotting the complete series on one canvas
    gwl = np.array(gwl)
    rain = np.array(rain)
    
    ax11 = plt.subplot(221)
    ax11.plot(rain[:,0], rain[:,1], "-x" ,lw=0.5, color="red")
    
    ax22 = plt.subplot(223)
    ax22.plot(gwl[:,0], gwl[:,1])
    
    ax33 = plt.subplot(122)
    ax33.plot(initial_states, FV.nodes, ls="dashed")
    
    for i in range(len(object_dict)):
        ax33.plot(object_dict[i].states, object_dict[i].nodes)
    plt.show()

############################# INTERACTIVE PLOT ################################
    
    # create canvas
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.35, wspace=0.45)
    # setup all axes and limits / only run at initialization
    def init():
        ax1.set_ylim(-110, -90)
        ax1.set_xlim(0, 100)
        line1.set_data(xdata1, ydata1)
        
        ax2.set_ylim(0.0, 3.0)
        ax2.set_xlim(0, 100)
        line2.set_data(xdata2, ydata2)
        
        ax3.set_ylim(-200, 0)
        ax3.set_xlim(-130, 100)
        line3.set_data(xdata3, ydata3)
        ax3.plot(xdata3[-1], ydata3[-1], color="black", ls = "dashed")
        return line1, line2, line3,
    
    # define axes and their positions on the canvas
    ax1 = plt.subplot(223)
    line1, = ax1.plot([], [])
    xdata1, ydata1 = [], []
    ax1.grid()
    ax1.set_title("Phreatic surface")
    ax1.set_xlabel("Time (d)")
    ax1.set_ylabel("Depth (cm)")

    ax2 = plt.subplot(221)
    line2, = ax2.plot([], [], "-x" , lw=0.5, color="red")
    xdata2, ydata2 = [], []
    ax2.grid()
    ax2.set_title("Precipitation")
    ax2.set_ylabel("rate (mm/d)")
    
    ax3 = plt.subplot(122)
    line3, = ax3.plot([], [], color="magenta")
    xdata3, ydata3 = [initial_states], [FVt.nodes]
    ax3.grid()
    ax3.set_title("Pressure head distribution")
    ax3.set_xlabel("Pressure head (cm)")
    ax3.set_ylabel("Depth (cm)")
    
    # run at every iteration
    def run(data):
        t, y = data
        xdata1.append(t)
        ydata1.append(gwl[t, -1])
        
        xdata2.append(t)
        ydata2.append(y)
        
        xdata3.append(object_dict[t].states)
        ydata3.append(object_dict[t].nodes)
        
        xmin, xmax = ax1.get_xlim()
        
        if t >= xmax:
            ax1.set_xlim(xmax, xmax + 100)
            ax1.figure.canvas.draw()
            
            ax2.set_xlim(xmax, xmax + 100)
            ax2.figure.canvas.draw()
            
        line1.set_data(xdata1, ydata1)
        line2.set_data(xdata2, ydata2)
        line3.set_data(xdata3[-1], ydata3[-1])
        fig.suptitle("Time = " + str(t) + "days (FV)")
        return line1, line2, line3,

    # play animation
    ani = animation.FuncAnimation(fig, run, Ffunc.rain_generator, blit=False, 
                                  interval=1.3e2, repeat=False, init_func=init)
    plt.show()
    '''
