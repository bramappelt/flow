''' This package contains several forcing functions that can describe point
and spatial forcing '''

import numpy as np


def block(t, t0=2, t1=7, Rmin=0.000, Rmax=0.006):
    gradient = (Rmax - Rmin) / (t1 - t0)
    if t < t0:
        return Rmin
    if t > t1:
        return Rmax
    else:
        return Rmin + gradient*(t - t0)
    
def sinusoidal(t, xtrans=0.25*np.pi, ytrans=1, amp=1, period=16*np.pi):
    return np.sin((t-xtrans) * (2*np.pi / period)) * amp + ytrans

def sinusoidal2(t, xtrans=0.25*np.pi, ytrans=1, amp=1, period=3*np.pi):
    return np.sin((t-xtrans) * (2*np.pi / period)) * amp + ytrans

def switch(t, t_on=5, t_off=10, value=-0.04):
    if t_on <= t <= t_off:
        return value
    else:
        return 0
    
def polynomial(pts):
    # solves Ax = b for a polynomial of any degree
    degree = len(pts)
    x = []
    for i in range(degree):
        eq = [pts[i][0]**(degree-1-j) for j in range(degree)]
        x.append(eq)
    x = np.reshape(x, (degree, degree))
    b = np.reshape(np.array([pts[i][1] for i in range(degree)]), (degree, 1))
    a = np.linalg.solve(x, b)
    
    def fitted_poly(crs):
        # returns a function that can calculate the fitted polynomial
        function = sum([a[i][0]*crs**(degree-1-i) for i in range(degree)])
        return function
    
    return a, x, b, fitted_poly

def rain_generator(t=0, maxtime=300, mu=0, std=0.7, seed=9999):
    # samples from the right side of a normal distribution
    np.random.seed(seed)
    while t < maxtime:
        while True:
            value = np.random.normal(mu, std)
            if 0 <= value <= 0.5:
                value = 0
                t += 1
                break
            elif value > 0.5:
                t += 1
                break
        yield t, value


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # block function    
    time = np.arange(0, 10)
    retval = []
    for t in time:
        retval.append(block(t))
    
    # sinusoidal
    retval2 = []
    for t in time:
        retval2.append(sinusoidal(t))
    
    # sinusodial 2
    retval3 = []
    for t in time:
        retval3.append(sinusoidal2(t))
    
    # switch
    retval4 = []
    for t in time:
        retval4.append(switch(t))

    for t_ in range(len(time)):
        fig, ax = plt.subplots(3, sharex=True)
        ax1, ax2, ax3 = ax
        ax1.plot(time, retval, color =  "blue", label = "recharge")
        ax1.scatter(t_, retval[t_], color = "blue")
        ax1.set_ylabel("rate (m/d)")
        ax1.legend(loc=4)
        ax1.grid()
        
        ax2.plot(time, retval2, color = "red", label = "right boundary")
        ax2.scatter(t_, retval2[t_], color = "red")
        #ax2.plot(time, retval3, color = "green")
        ax2.set_ylabel("Head state (m)")
        ax2.legend(loc=4)
        ax2.grid()
        
        ax3.plot(time, retval4, color = "green", label = "well")
        ax3.scatter(t_, retval4[t_], color = "green")
        ax3.set_ylabel("rate (m/d)")
        ax3.legend(loc=4)
        ax3.grid()
        
        ax3.set_xlabel("Time (d)")
        plt.suptitle("Forcing functions")
        #fig.savefig("plots\\forcing"+str(t_)+".png")
    
    # polynomials
    ax = np.arange(-5, 10, 0.5)
    pts = [[2, 4]]
    pts2 = [[3, 4], [5, 6], [9, -3]]
    pts3 = [[-4, 5], [6, -1], [8, 3]]
    a1, x1, b1, func1 = polynomial(pts)
    a2, x2, b2, func2 = polynomial(pts2)
    a3, x3, b3, func3 = polynomial(pts3)
    
    fig, axis = plt.subplots()
    axis.plot(ax, func1(ax), color="blue")
    axis.scatter(*zip(*pts), color = "blue")
    axis.plot(ax, func2(ax), color="red")
    axis.scatter(*zip(*pts2), color = "red")
    axis.plot(ax, func3(ax), color="black")
    axis.scatter(*zip(*pts3), color = "black")
