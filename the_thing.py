# Numerisk prosjekt, Kvantefysikk 1 (FY2045), NTNU 2015

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def Psi(x, x0, w, C, sigma_x, dt):
    psi_r = C * np.exp(-(x - x0)**2 / (2*sigma_x**2)) * ( np.cos(k*x))
    psi_i = C * np.exp(-(x - x0)**2 / (2*sigma_x**2)) * ( np.sin(k*x - w*dt/2))
    psi_r[0] = 0
    psi_r[-1] = 0
    psi_i[0] = 0
    psi_i[-1] = 0
    return psi_r + 1j*psi_i



def Psi_dt(a, dt, number_of_times):
    a_new = a
    for i in range(number_of_times):
        for n in range(np.size(a) - 2):
            n += 1
            a_new.imag[n] -= dt * (V[n] * a.real[n] -
                               (a.real[n + 1] - 2 * a.real[n] + a.real[n - 1]))

            a_new.real[n] += dt * (V[n] * a.imag[n] - (1 / (2 * m * (dx ** 2))) *
                               (a.imag[n + 1] - 2 * a.imag[n] + a.imag[n - 1]))
    return a_new



def potential(a):
    b = np.zeros(np.size(a))
    for n in range(np.size(a) - 1):
        if (L/2 - l/2) < a[n] < (L/2 + l/2):
            b[n] = 1/2 * E
    return b


def update():
    return 0

'''
def plotter(X, a):  # definer plottefunksjonen.
    plt.clf()
    plt.plot(X, a.real, X, a.imag)
    plt.xlim((0, 20))
    plt.ylim((-2, 2))
    plt.pause(0.001)
    plt.show()
'''

# definer variabler, funksjoner, etc.

h = 1  # h-bar
m = 1  # mass, m
k = 20  # wavenumber
L = 20  #
sigma_x = 1
C = np.sqrt(np.sqrt(1/(pi*sigma_x**2)))  # normalization constant, ser ut til å kanskje være feil??
w = 10  # omega
l = L/50  # lengde på barriere
E = h*w  # energy

# plt.ion() # må ha med denne for å kunne modifisere plottet etter at det er tegnet opp.

# definer området denne shiten skal virke over, eg. fra x0 -> x1
N = 500
x0 = 5
x1 = L
t = 0
dx = L/(N-1)
dt = 0.1 * 2*m*(dx**2)  # for stabilitet

X = np.linspace(0, L, num=N)

psi = Psi(X, x0, w, C, sigma_x, dt)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, L), ylim=(-2, 2))
line, = ax.plot([], [])

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, L, num=N)
    y = Psi_dt(psi, dt, 10)
    print(y)
    line.set_data(x, y.real)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

plt.show()