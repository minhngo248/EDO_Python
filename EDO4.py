# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:28:51 2021

@author: Minh NGO
"""

import numpy as np
#import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from RK import RK1
from RK import RK2
import RK_plot

def edo4(t, Y) :
	"""
	L'équation différentielle y'' + y' + sin(y) = 0
	"""
	y   = Y[0]
	yp  = Y[1]
	dy  = yp
	dyp = -yp-np.sin(y)
	return np.array([dy, dyp])

t = np.linspace(0.0, 10.0, 11)
y0 = np.array([1.0, -0.5])
y_RK1  = RK1      (edo4, (t[0], t[-1]), y0, t)
y_RK2  = RK2      (edo4, (t[0], t[-1]), y0, t)
y_RK45 = solve_ivp(edo4, (t[0], t[-1]), y0, t_eval = t).y

fig, ax = RK_plot.plot_solution("Approximation de $y''+y'+sin(y)=0$ par des méthodes de RUNGE-KUTTA", t, None, (y_RK1, "EULER", "+"), (y_RK2, "RK2", "x"), (y_RK45, "RK45", "rs"))
ax.set(xlim = (-0.5, 10.5), ylim = (-1.25, 1.25))
#fig.savefig("EDO4.pdf")