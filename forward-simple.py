"""
Assuming constant D
Original PDE, after multiplied by T (Normalize by final time)
    u_t = D laplace u + ρ u(1-u)

where t = [0 1], D = D T, ρ = ρ T 

In polar coordinate , laplace u(r,θ) = u_rr + 1/r u_r  + 1/r^2 u_θθ

For radial symmetry solution , u_θθ = 0

At r=R, Neuman boundary condition u_r = 0
At r=0, symmetric at origin,  u_r = 0

Therefore, the whole pde is 1d pde on domain [0,R]: 
    
    u_t = D (u_xx + 1/x u_x)  + ρ u (1-u), neuman bc



In PINN, we use a diffused domain, x=[0,1]
    (uϕ)_t = D div  ( ϕ grad u) + ρ ϕ u(1-u)
           = D grad ϕ . grad u + D  ϕ laplace u  + ρ ϕ u(1-u)     
In polar coordinate, grad u = u_r + 1/r u_θ. 
If radial symmetric, u_θ = 0
We have
    u_t =   D ( ϕ_x u_x + ϕ u_xx + ϕ 1/x u_x ) + ρ ϕ u (1-u), 
    neuman bc at x=0 (for radial symmetry)
    choose neuman bc at x=1 (can be arbitrary)

based on the manual
https://py-pde.readthedocs.io/en/latest/getting_started.html
this package use finite difference. The grid points are placed at cell center
By default of pde.solve, ScipySolver with an automatic, adaptive time step provided by scipy is used.
"""

import numpy as np
import matplotlib.pyplot as plt
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph, UnitGrid


R = 0.5 # circle domain
N = 50 # grid size
T = 450 # final time

# time normalized coefficients
D = T*0.13e-4
rho = T*0.025


#%% this does not use diffused domain
# based on the example 
#https://py-pde.readthedocs.io/en/latest/examples_gallery/pde_heterogeneous_diffusion.html#sphx-glr-examples-gallery-pde-heterogeneous-diffusion-py
# eq = PDE({"u": f"{D}*laplace(u) + {D}*d_dx(u)/x + {rho}*u * (1-u)"},bc = {'derivative': 0})
# grid = CartesianGrid([[0, R]], N)
# u0 = ScalarField.from_expression(grid, "0.1*exp(-1000*x**2)")

# # solve the equation and store the trajectory
# storage = MemoryStorage()
# eq.solve(u0, t_range=1, tracker=storage.tracker(0.1)) #interface to sample the solution


# #%% plot_kymograph(storage)
# x = np.array(storage.grid.coordinate_arrays)
# y = np.array(storage.data)
# plt.plot(x.T,y.T)
#%% this part use diffused domain


phi = "(0.5+0.5* tanh((0.5-x)/0.01))"
pde = f"{D}*dot(gradient({phi}),gradient(u)) + {D}*{phi}*laplace(u) + {D}*{phi}*d_dx(u)/x + {phi}*{rho}*u*(1-u)"
eq = PDE({"u": pde},bc = {'derivative': 0})

# the witdith of the diffused domain is approximated 0.02
# need to choose h such that arond 5 grid point resolve the interface

import time
t = time.time()
grid = CartesianGrid([[0, 0.5]], 1000)
u0 = ScalarField.from_expression(grid, "0.1*exp(-1000*x**2)")
# solve the equation and store the trajectory
storage = MemoryStorage()
eq.solve(u0, t_range=1, tracker=storage.tracker(1/15))



# plot_kymograph(storage)
x = np.array(storage.grid.coordinate_arrays)
y = np.array(storage.data)
plt.plot(x.T,y.T)
plt.xlim([0,R])

print(time.time() - t)
