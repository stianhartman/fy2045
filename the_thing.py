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
def Psi(x, x0, w, C, sigma_x, t):
    psi_r = C * np.exp(-(x - x0)**2 / (2*sigma_x**2)) * ( np.cos(k*x - w*t))
    psi_i = C * np.exp(-(x - x0)**2 / (2*sigma_x**2)) * ( np.sin(k*x - w*t))
    psi_r[0] = 0
    psi_r[-1] = 0
    psi_i[0] = 0
    psi_i[-1] = 0
    return psi_r + 1j*psi_i



'''
def V(x):
    return
'''


def plotter(X, psi):  # definer plottefunksjonen.
    plt.clf()
    plt.plot(X, psi.real, X, psi.imag)
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
# C = ...  # normalization constant
C = 1
w = 1
sigma_x = 1


# plt.ion() # må ha med denne for å kunne modifisere plottet etter at det er tegnet opp.

# definer området denne shiten skal virke over, eg. fra x0 -> x1
N = 10
x0 = 0
x1 = L
t = 0

X = np.linspace(x0, x1, num=N)
psi = Psi(X, x0, w, C, sigma_x, t)  # går sikkert ann å gjøre denne penere eller på en bedre måte.

plotter(X, psi)  # bare midlertidig for å se at ting funker.

'''
# må på en eller annen måte kalkulere Psi på et senere tidspunkt. hmm...

dt = 0.1

psi_r = dt/h * ( B1 )
psi_i = dt/h * ( B2 )  # B1,2 er de lange greiene han skrev i forelesning -- denne er din, Stian.

'''
