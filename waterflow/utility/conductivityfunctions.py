''' This package contains several conductivity functions. '''
import os
import numpy as np
from pandas import read_table

import waterflow

STARINGREEKS = read_table(os.path.join(
        os.path.split(waterflow.__file__)[0],
        "DATA/StaringReeks.txt"), delimiter="\t")

def select_soil(name):
    columnnames = STARINGREEKS.columns.values
    values = STARINGREEKS[STARINGREEKS["soiltype"] == name].values.flatten()
    return dict(zip(columnnames, values))

def kfun(x, ksat=1, b=0.001):
    return ksat + x*b

# van genuchten 1980
def VG_theta(theta, theta_r=0.02, theta_s=0.43, a=0.0234, n=1.801):
    m = 1-1/n
    THETA = (theta_s - theta_r) / (theta - theta_r)
    return ((THETA**(1/m) - 1) / a**n)**(1/n)

def VG_pressureh(h, theta_r=0.02, theta_s=0.43, a=0.0234, n=1.801):
    if h >= 0:
        return theta_s
    m = 1-1/n
    return theta_r + (theta_s-theta_r) / (1+(a*-h)**n)**m
    
def VG_conductivity(h, ksat=23.41, a=0.0234, n=1.801):
    if h >= 0:
        return ksat
    m = 1-1/n
    h_up = (1 - (a * -h)**(n-1) * (1 + (a * -h)**n)**-m)**2
    h_down = (1 + (a * -h)**n)**(m / 2)
    return (h_up / h_down) * ksat
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    SOIL = select_soil("B1")
    theta_r = SOIL['theta.res']
    theta_s = SOIL['theta.sat']
    a = SOIL['alpha']
    n = SOIL['n']
    ksat = SOIL['ksat']
    name = SOIL['name']
    
    heads = np.arange(0.1, 18000, 1) * -1
    theta = [VG_pressureh(h,  theta_r=theta_r, theta_s=theta_s, a=a, n=n) for h in heads]
    cond  = [VG_conductivity(h, ksat=ksat, a=a, n=n) for h in heads]
    
    fig, ax = plt.subplots(1,2)
    ax1, ax2 = ax
    ax1.semilogy(theta, heads*-1, "-o")
    ax2.semilogx(heads*-1, cond, "-o")
    ax1.set_title(str(name) + " pF")
    ax2.set_title(str(name) + " Conductivity")
    ax1.grid()
    ax2.grid()
    plt.show()
    
    # checks
    headsT = np.arange(-10, 10, 0.5)
    thetaT = [VG_pressureh(h,  theta_r=theta_r, theta_s=theta_s, a=a, n=n) for h in headsT]
    condT = [VG_conductivity(h, ksat=ksat, a=a, n=n) for h in headsT]
    


