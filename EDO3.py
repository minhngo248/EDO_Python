# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:40:27 2021

@author: Minh NGO
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from RK import RK1
from RK import RK2
#import RK_plot

x = np.linspace(0,10,100)
fig = plt.figure()
ax = plt.subplot()
ax.set_title("Approximation de $y''+y'+y=0$ par des méthodes de RUNGE-KUTTA")
ax.set_xlabel("$t$")
ax.set_ylabel("$y(t)$")
f = np.exp(-x/2)*np.cos(np.sqrt(3)/2*x)
plt.plot(x,f, 'b-', label = "Solution exacte")


def edo3(t, Y) :
	"""
	L'equation différentielle y'' + y' + y = 0 (1)
	"""
	y   = Y[0]
	yp  = Y[1]
	dy  = yp
	dyp = -yp-y
	return np.array([dy, dyp])

t = np.linspace(0.0, 10.0, 11)
"""
    Solution exacte de l'eq (1): y(t) = exp(-t/2).cos(sqrt(3)/2.t)
"""
y0 = np.array([1.0, -0.5])
y_RK1  = RK1      (edo3, (t[0], t[-1]), y0, t)
y_RK2  = RK2      (edo3, (t[0], t[-1]), y0, t)
y_RK45 = solve_ivp(edo3, (t[0], t[-1]), y0, t_eval = t).y
plt.plot(t, y_RK1[0][0:], "+", label = "EULER")
plt.plot(t, y_RK2[0][0:], "x", label = "RK2")
plt.plot(t, y_RK45[0][0:], "rs", label = "RK45")
plt.legend(loc='upper right')
#plt.savefig("EDO3.pdf", dpi=300, bbox_inches='tight')
plt.show()