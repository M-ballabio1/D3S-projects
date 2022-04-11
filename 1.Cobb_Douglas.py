# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:39:47 2022

@author: matte
"""

#%% Import module
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#%% Define paths

#path = sys.argv[1]                                     #usa come file path quella corrente
#path = r'C:\Users\matte\PycharmProjects\pythonProject_MachineLearning\Algorithmic trading'
#immagini_path = path+(r"\immagini\2.immagini")

#%% Main

k = np.arange(0, 200, 2.5)
k1 = np.arange(0, 200, 20)
l = np.arange(0, 200, 2.5)
l1 = np.arange(0, 200, 10)
z = 0.5

#%% Functions

def f_CobbDouglas(k, l):
    # Create alpha and z
    z = 1
    #alpha = 0.33
    return k**(1/2) * l**(1/2)

def returns_to_scale(K, L, gamma):
    y1 = f_CobbDouglas(k, l)
    y2 = f_CobbDouglas(k*gamma, l*gamma)
    y_ratio = y2 / y1
    return y_ratio / gamma

def marginal_products(K, L, epsilon):
    mpl = (f_CobbDouglas(k, l + epsilon) - f_CobbDouglas(k, l)) / epsilon
    mpk = (f_CobbDouglas(k + epsilon, l) - f_CobbDouglas(k, l)) / epsilon
    return mpl, mpk
    
def f_Fixed_Proportions(k, l):
     return k * l
 
#%% 

# marginal product of labor (MPL) and marginal product of capital (MPK)

mpl, mpk = marginal_products(1.0, 0.5,  1e-4)
print(f"marginal product of labor (MPL) = {mpl}, marginal product of capital (MPK) = {mpk}")

# Visualization of Technology with Cobb-Douglas Curve:
## Example equations: output (y) = [(x1^(1/2) , x2^(1/2)]

K, L = np.meshgrid(k, l)
Y = f_CobbDouglas(K, L)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("K")
ax.set_ylabel("L")
ax.set_zlabel("Y")
ax.contour(K, L, Y, 100) 
plt.show()

# Visualization of Technology with Fixed proportion Curve:
## Example equations: output (y) = [(x1 , 0.5*(x2)]
    
K1, L1 = np.meshgrid(k1, l1)
Y1 = f_Fixed_Proportions(K1, L1)

fig1 = plt.figure()
ax1 = Axes3D(fig1)
ax1.set_xlabel("K")
ax1.set_ylabel("L")
ax1.set_zlabel("Y")
ax1.contour(K1, L1, Y1, 100) 
plt.show()