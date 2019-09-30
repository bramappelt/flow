""" This package contains several conductivity functions
and a soil selector function """

import os

import numpy as np
import pandas as pd

from waterflow.conf import DATA_DIR


def soilselector(soiltype):
    """ Select a soil from the Staringreeks """
    staringreeks = pd.read_table(os.path.join(DATA_DIR, "StaringReeks.txt"),
                                 delimiter="\t")
    # select row
    soildata = staringreeks.loc[staringreeks['soiltype'] == soiltype]
    # turn row to namedtuple
    return list(soildata.itertuples(name='soil', index=False))[0]


# Van Genuchten (theta)
def VG_theta(theta, theta_r, theta_s, a, n):
    m = 1-1/n
    THETA = (theta_s - theta_r) / (theta - theta_r)
    return ((THETA**(1/m) - 1) / a**n)**(1/n)


# Van Genuchten (h)
def VG_pressureh(h, theta_r, theta_s, a, n):
    # to theta
    if h >= 0:
        return theta_s
    m = 1-1/n
    return theta_r + (theta_s-theta_r) / (1+(a*-h)**n)**m


# Van Genuchten (x, h)
def VG_conductivity(x, h, ksat, a, n):
    if h >= 0:
        return ksat
    m = 1-1/n
    h_up = (1 - (a * -h)**(n-1) * (1 + (a * -h)**n)**-m)**2
    h_down = (1 + (a * -h)**n)**(m / 2)
    return (h_up / h_down) * ksat


if __name__ == "__main__":
    pass