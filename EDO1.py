#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from RK import RK1
from RK import RK2
import RK_plot

def edo1(t, Y) :
	"""
	L'équation différentielle y'' = -y (1)
	"""
	y   = Y[0]
	yp  = Y[1]
	dy  = yp
	dyp = -y
	return np.array([dy, dyp])

"""
    Solution exacte de l'equation (1) y(t) = sin(t)
"""

t = np.linspace(0.0, 10.0, 11)
y0 = np.array([0.0, 1.0])
y_RK1  = RK1      (edo1, (t[0], t[-1]), y0, t)
y_RK2  = RK2      (edo1, (t[0], t[-1]), y0, t)
y_RK45 = solve_ivp(edo1, (t[0], t[-1]), y0, t_eval = t).y

fig, ax = RK_plot.plot_solution("Approximation de $y'' = -y$ par des méthodes de RUNGE-KUTTA", t, np.sin, (y_RK1, "EULER", "+"), (y_RK2, "RK2", "x"), (y_RK45, "RK45", "rs"))
ax.set(xlim = (-0.5, 10.5), ylim = (-3.0, 3.0))
#fig.savefig("EDO1.pdf")