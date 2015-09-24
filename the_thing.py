__author__ = 'erik'

import numpy as np
import matplotlib.pyplot as plt


# definer variabler, funksjoner, etc.

h = 1.055 * 10**(-34)  # h-bar in J*s
pi = np.pi  # gode gamle pi = 3.1415...


'''
# psi = psi_r + i * psi_i

def psi_r(x, t):
    return ......

def psi_i(x, t):
    return ......

def V(x):
    return ......


'''

# definer området denne shiten skal virke over, eg. fra x0 -> x1
N = 10
x0 = 0
x1 = 1

X = np.linspace(x0, x1, num=N)
psi_r = psi_r(X, t)
psi_i = psi_i(X, t)
