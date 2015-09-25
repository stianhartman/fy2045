# Numerisk prosjekt, Kvantefysikk 1 (FY2045), NTNU 2015

# import div
import numpy as np
import matplotlib.pyplot as plt


# definer variabler, funksjoner, etc.

h = 1.055 * 10 ** (-34)  # h-bar in J*s
pi = np.pi  # gode gamle pi = 3.1415...
k = pi*5  # wavenumber
m = h
# w = ...  # omega
plt.ion() # må ha med denne for å kunne modifisere plottet etter at det er tegnet opp.

'''
# wave function = wave packet:
# psi = Ae^(-bx^2)*e^i(kx-wt)
#     = Ae^(-bx^2)*(cos(kx-wt) - i*sin(kx-wt))    # hva er A og b??
# fra wikipedia ser vi:
# psi(x,t) = e^(-(x-ct))^2 * ( cos(k * (x-ct)) +i*sin(k * (x-ct)))  # vet ikke helt hva c er. c muligens = omega

# psi = psi_r + i * psi_i
# psi_r = e^(-(x-ct))^2 * ( cos(k * (x-ct)) )
# psi_i = e^(-(x-ct))^2 * ( sin(k * (x-ct)) )

'''

def Psi_r(x, t):
    c = 1
    A = np.exp(-(x - np.ones(np.size(x)) * c * t) ** 2)
    return A * np.cos(k * (x - np.ones(np.size(x)) * c * t))



def Psi_i(x, t):
    c = 1
    A = np.exp(-(x - np.ones(np.size(x)) * c * t) ** 2)
    return A * np.sin(k * (x - np.ones(np.size(x)) * c * t))


def V(x):
    print(x)
    x[x < 1] = 0
    x[x >= 1] = 10
    print(x)
    return x



def plotter(X, psi_real, psi_imag):  # definer plottefunksjonen.
    plt.clf()
    plt.plot(X, psi_real, X, psi_imag)
    plt.xlim((-2*pi, 2*pi))
    plt.ylim((-2, 2))
    plt.pause(0.002)
    plt.show()


# definer området denne shiten skal virke over, eg. fra x0 -> x1
N = 1000
x0 = - 2*pi
x1 = 2*pi
t = 0
dt = 0.02
dx = (x1 - x0)/N

X = np.linspace(x0, x1, num=N)
for n in range(1):
    t = t + dt
    psi_r = Psi_r(X, t)
    psi_i = Psi_i(X, t)

    # plotter(X, psi_r, psi_i)  # bare midlertidig for å se at ting funker.


'''
# må på en eller annen måte kalkulere Psi på et senere tidspunkt. hmm...

dt = 0.1
psi_r = dt/h * ( B1 )
psi_i = dt/h * ( B2 )  # B1,2 er de lange greiene han skrev i forelesning -- denne er din, Stian.



for n in range(50):
    dt = 0.1
    dx = 0.1
    t = t + dt

    psi_r = psi_r - ( (h*dt/(2*m*dx**2)) * ( Psi_i(X + np.ones(np.size(X))*dx, t + dt/2) +
                    Psi_i(X - np.ones(np.size(X))*dx, t + dt/2) - 2*Psi_i(X, t + dt/2)) + (dt/h)*V(X)*Psi_i(X, t + dt/2))

    psi_i = psi_i


    dette fungerte ikke så bra :(
    psi_r = dt/h *  ((h/dt) * Psi_r(X, t - dt/2) - (h**2 / (2*m*(dx**2))) *
                 (Psi_i(X + np.ones(np.size(X))*dx, t) - 2*psi_i +
                  Psi_i(X - np.ones(np.size(X))*dx, t)) + V(X)*psi_i)

    psi_i = dt/h *  ((h/dt) * psi_i - (h**2 / (2*m*(dx**2))) *
                 (Psi_r(X + np.ones(np.size(X))*dx, t + dt/2) - 2*Psi_r(X, t + dt/2) +
                  Psi_r(X - np.ones(np.size(X))*dx, t + dt/2)) + V(X)*psi_r)

    plotter(X,psi_r,psi_i)

'''
