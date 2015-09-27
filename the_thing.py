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

def Psi_initial(x, x0, t, dt):
    C = 1
    w = 1
    sigma_x = 2

    psi_r = C * np.exp(-(x - x0)**2 / (2*sigma_x**2)) * (np.cos(k*x - w*(t+dt/2)))
    psi_i = C * np.exp(-(x - x0)**2 / (2*sigma_x**2)) * (np.sin(k*x - w*t))
    psi_r[0] = 0
    psi_r[-1] = 0
    psi_i[0] = 0
    psi_i[-1] = 0
    return psi_r + 1j*psi_i

'''
def Psi_dt(a, dt):
    for n in range(np.size(a) - 2):
        n = n + 1
        a.imag[n] = a.imag[n-1] - dt * ( V[n] * a.real[n] - (1/(2*m*(dx**2))) *
                                         (a.real[n+1] - 2*a.real[n] + a.real[n-1]))

        a.real[n] = a.real[n-1] + dt * ( V[n] * a.imag[n] - (1/(2*m*(dx**2)))*
                                         (a.imag[n+1] - 2*a.imag[n] + a.imag[n-1]))
    return a
'''

def Psi_dt(a, dt):
    for n in range(np.size(a) - 2):
        n += 1
        a.imag[n] -= dt * (V[n] * a.real[n] - (1 / (2 * m * (dx ** 2))) *
                           (a.real[n + 1] - 2 * a.real[n] + a.real[n - 1]))

        a.real[n] += dt * (V[n] * a.imag[n] - (1 / (2 * m * (dx ** 2))) *
                           (a.imag[n + 1] - 2 * a.imag[n] + a.imag[n - 1]))
    return a


'''
def potential(a):
    b = np.zeros(np.size(a))
    for n in range(np.size(a) - 1):
        if n <= 10:
            b[n] = 0
        else:
            b[n] = 100
    return b
'''


def plotter(X, a):  # definer plottefunksjonen.
    plt.clf()
    plt.plot(X, a.real, X, a.imag)
    plt.xlim((0, 20))
    plt.ylim((-2, 2))
    plt.pause(0.01)
    plt.show()


# definer variabler, funksjoner, etc.

h = 1  # h-bar
m = 1  # mass, m
k = 20  # wavenumber
L = 20  #
# w = ...  # omega
# E = ...  # energy
# C = ...  # normalization constant


plt.ion() # må ha med denne for å kunne modifisere plottet etter at det er tegnet opp.

# definer området denne shiten skal virke over, eg. fra x0 -> x1
N = 200
x0 = 5
x1 = L
t = 0
dx = L/(N-1)
print('dx: ', dx)
dt = 0.1 * 2*m*(dx**2)  # for stabilitet
print('dt: ', dt)



# initsialisere
X = np.linspace(0, x1, num=N)
V = potential(X)  # midlertidig potential
psi = Psi_initial(X, x0, t, dt)  # går sikkert ann å gjøre denne penere eller på en bedre måte.


plotter(X, psi)  # bare midlertidig for å se at ting funker.

for n in range(1000):
    psi = Psi_dt(psi, dt)
    plotter(X, psi)
