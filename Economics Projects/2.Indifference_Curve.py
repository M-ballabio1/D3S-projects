# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:01:07 2022

@author: matte
"""


#%% Import module

from pathlib import Path               #combine path elements with /
import os
from pylab import *
from scipy import interpolate
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt

#%% Define paths

#path = r'C:\Users\matte\PycharmProjects\pythonProject_MachineLearning\Algorithmic trading'
#immagini_path = path+r'\immagini\1.immagini'

#%% Main

mpl.rcParams["font.family"] = "cursive"


def indifference_curve(x, a, c):     #curve indifferenza consumatori
    return 5*(x-a)**(2)+c

fig = plt.figure(num=1, figsize=(15,8))
plt.style.use('bmh')
plt.title("Scelte di consumo di x1 e x2 (con x2 fisso)", {"size":24})

#utility curve U1-U2-U3
x = np.arange(0, 0.2, 0.01)
y = indifference_curve(x, 0.3, 0.6)
plt.plot(x, y, linestyle="--",linewidth=4, color="grey")

x = np.arange(0.5, 0.7, 0.01)
y = indifference_curve(x, 0.7, 0.35)
plt.plot(x, y, linestyle="--",linewidth=4, color="grey")

x = np.arange(1, 1.3, 0.01)
y = indifference_curve(x, 1.25, 0.3875)
plt.plot(x, y, linestyle="--",linewidth=4, color="grey")

#vincoli di bilancio V1-V2-V3
plt.plot([0,0.5],[1,0], label="V1", linestyle="-",linewidth=5)
plt.plot([0,1],[1,0], label="V2", linestyle="-",linewidth=5)
plt.plot([0,2],[1,0], label="V3", linestyle="-",linewidth=5)

#Price Offer Curve
x = np.array([0.1, 0.6, 1.2, 1.5])
y = np.array([0.8, 0.4, 0.4, 0.5])
xnew = np.arange(0.1, 1.5, 0.01)
func = interpolate.interp1d(x, y, kind='quadratic')
ynew = func(xnew)
plt.plot(xnew, ynew, label='Price Offer Curve', linestyle="--",linewidth=7, color='y')

plt.plot(0.1, 0.8, 'rs', 0.6, 0.4, 'bs', 1.2, 0.4, 'ys')

plt.legend(["U1","U2","U3","V1","V2","V3","Price Offer Curve","A","B","C"], loc=0, ncol=2, prop={"size":20})
plt.xlim(0,2.2)
plt.ylim(0,1.2)
plt.xlabel("X1",{"size":25})
plt.ylabel("X2",{"size":25})

plt.xticks([])
plt.yticks([])
plt.show()







