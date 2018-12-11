from __future__ import division
import numpy as np
import math

def p_mold(T,T_0=36):
    Em = (T-T_0)
    return  1/(1+np.exp(Em))

def p_yeast(T,T_0=36):
    Em = (T-T_0)
    return  np.exp(Em)/(1+np.exp(Em))

def p_intermediate(T,T_0=36):
    Em = (T-T_0)
    return (2*np.exp(Em))/(np.exp(Em)+1)**2