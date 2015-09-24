# Numerisk prosjekt, Kvantefysikk 1 (FY2045), NTNU 2015


import numpy as np
import matplotlib.pyplot as plt


# definer variabler, funksjoner, etc.

h = 1.055 * 10**(-34)  # h-bar in J*s
pi = np.pi  # gode gamle pi = 3.1415...
# plt.ion() # må ha med denne for å kunne modifisere plottet etter at det er tegnet opp.

'''
# wave function == wave packet ??
# psi = psi_r + i * psi_i
# psi = Ae^(bx^2)*e^(kw-wt) ??

def Psi_r(x, t):
    return ......

def Psi_i(x, t):
    return ......

def V(x):
    return ......


def plotter(X, psi_real, psi_imag):  # definer plottefunksjonen.
    plt.clf()
    plt.plot(X, psi_real, psi_imag, 'r-', 'b-')
    plt.xlim((0,2))
    plt.ylim((-2, 2))
    plt.show()


'''


# definer området denne shiten skal virke over, eg. fra x0 -> x1
N = 10
x0 = 0
x1 = 1

X = np.linspace(x0, x1, num=N)
psi_r = Psi_r(X, t)
psi_i = Psi_i(X, t)


# må på en eller annen måte kalkulere Psi på et senere tidspunkt. hmm...

dt = 0.1

psi_r = dt/h * ( B1 )
psi_i = dt/h * ( B2 )  # B1,2 er de lange greiene han skrev i forelesning







