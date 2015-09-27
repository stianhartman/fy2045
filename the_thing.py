# Numerisk prosjekt, Kvantefysikk 1 (FY2045), NTNU 2015

# import div
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi  # gode gamle pi = 3.1415...

'''
# wave function = wave packet:
# psi = Ce^(-(x-x0)^2/(2*sigma_x**2))*e^i(kx-wt)
#     = Ce^(-(x-x0)^2/(2*sigma_x**2))*(cos(kx-wt) + i*sin(kx-wt))    # hva er C ??
#
# psi = psi_r + i * psi_i
# psi_r = Ce^(-(x-x0)^2/(2*sigma_x**2)) * ( cos(kx-wt) )
# psi_i = Ce^(-(x-x0)^2/(2*sigma_x**2)) * ( sin(kx-wt) )

'''
# def Psi(x, t):


def Psi_r(x, t):
    c = 1
    A = np.exp(-(x - np.ones(np.size(x)) * c * t) ** 2)
    return A * np.cos(k * (x - np.ones(np.size(x)) * c * t))


def Psi_i(x, t):
    c = 1
    A = np.exp(-(x - np.ones(np.size(x)) * c * t) ** 2)
    return A * np.sin(k * (x - np.ones(np.size(x)) * c * t))


'''
def V(x):
    return
'''


def plotter(X, psi_real, psi_imag):  # definer plottefunksjonen.
    plt.clf()
    plt.plot(X, psi_real, X, psi_imag)
    plt.xlim((X[0], X[-1]))
    plt.ylim((-2, 2))
    plt.show()


# definer variabler, funksjoner, etc.

h = 1  # h-bar
m = 1  # mass, m
k = 20  # wavenumber
L = 20  #
# w = ...  # omega
# E = ...  # energy


# plt.ion() # må ha med denne for å kunne modifisere plottet etter at det er tegnet opp.

# definer området denne shiten skal virke over, eg. fra x0 -> x1
N = 1000
x0 = 0
x1 = L
t0 = 0

X = np.linspace(x0, x1, num=N)
psi_r = Psi_r(X, t)
psi_i = Psi_i(X, t)

plotter(X, psi_r, psi_i)  # bare midlertidig for å se at ting funker.

'''
# må på en eller annen måte kalkulere Psi på et senere tidspunkt. hmm...

dt = 0.1

psi_r = dt/h * ( B1 )
psi_i = dt/h * ( B2 )  # B1,2 er de lange greiene han skrev i forelesning -- denne er din, Stian.

'''
