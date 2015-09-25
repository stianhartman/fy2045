__author__ = 'erik'

import numpy as np
import matplotlib.pyplot as plt

# plt.ion()

def plotter(x, y):  # definer plottefunksjonen.
    plt.clf()
    plt.plot(x, y)
    plt.xlim((-5, 5))
    plt.ylim((-2, 15))
    # plt.pause(1)
    plt.show()

def f(x):
    return -1/3 * x**3 + 3*x + 5

N = 1000
x0 = -10
x1 = 10
t = 1
dt = 0.4
dx = (x1 - x0)/N

X = np.linspace(x0, x1, num=N)


