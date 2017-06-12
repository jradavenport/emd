import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline


def resid(t,f):
    '''
    Outline:
    * do 1-point derivative to find local Max's and Min's
        * this is point of possible improvement
    * do cubic splines between Max's and Min's
    * find running mean between Max and Min
    * return residial
    '''
    maxm = argrelextrema(f, np.greater)
    minm = argrelextrema(f, np.less)

    maxCS = CubicSpline(t[maxm], f[maxm])
    maxi = maxCS(t, extrapolate=None)

    minCS = CubicSpline(t[minm], f[minm])
    mini = minCS(t, extrapolate=None)

    meani = (maxi + mini) / 2.0
    
    return f - meani
 

def IMF(t, f, Nmax=200, SDmax=0.01, Zmax=5):
    '''
    compute the intrinsic mode function
    '''
    ftmp = f
    i = 0
    Zok = 0
    
    while i<Nmax:
        r1 = resid(t, ftmp)
    
        # does the residual satisfy the 2 criteria?
        # 1) Nextrema = Nzero_crossings
        # 2) StdDev is lower than some threshold
        SD = np.sum(np.power(np.abs(ftmp - r1), 2) / np.power(ftmp, 2))

        # compute Nextrema
        maxm = argrelextrema(r1, np.greater)
        minm = argrelextrema(r1, np.less)

        # from http://stackoverflow.com/a/30281079
        Nzero = ((r1[:-1] * r1[1:]) < 0).sum()

        if np.abs(Nzero - (np.size(maxm)+np.size(minm))) < 2:
            Zok = Zok + 1
    
        if SD <= SDmax:
            i=Nmax
        
        if Zok >= Zmax:
            i=Nmax
    
        i=i+1
        ftmp = r1
    return r1



