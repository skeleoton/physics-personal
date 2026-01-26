#imports

import numpy as np 

def position(t, x0, v0, a):
    return x0 + v0*t + 0.5*a*t**2

def velocity(t, v0, a):
    return v0 + a*t

def acceleration(t, a):
    return np.full_like(t, a) #since a is constant