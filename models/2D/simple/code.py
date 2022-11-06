# Scientific Libraries
import numpy as np
np.set_printoptions(edgeitems=10,linewidth=300)
import deepxde as dde
import tensorflow as tf
import nibabel as nib
from scipy.interpolate import interpn
from tensorflow_graphics.math.interpolation import trilinear as interp3d

# System/Timing Libraries
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.insert(1, '..')

# Plotting Libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.express as px
import plotly.graph_objects as go
import PlotlyReference as pr
from evtk.hl import pointsToVTK

filepath="./DDA/2D-MRI/inverse-ray/1/"
print(filepath)

D = tf.Variable(1e-3) # 1.3e-5
rho = tf.Variable(1e-3) # 0.025

# creates a 0.05 radius interval/circle/sphere/hypersphere IC
def ic(x):
    r = tf.square(x[:, 0:1]) + tf.square(x[:, 1:2])
    return 0.1*tf.exp(-1000*r)

# define reaction-diffusion Fisher-Kolmogorov equations
def pde(v, u):
    phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2)) / 0.01)
    u_t = dde.grad.jacobian(u*phi, v, j=2)
    u_x = dde.grad.jacobian(u, v, j=0)
    u_y = dde.grad.jacobian(u, v, j=1)
    
    u_xx = dde.grad.jacobian(u_x*phi, v, j=0)
    u_yy = dde.grad.jacobian(u_y*phi, v, j=1)
    
    return [u_t - 300*(1e-2*D*(u_xx + u_yy) + 10*rho*phi*u*(1-u))]


# remember to check the cutoffs of the cropped tumor bounds
geom = dde.geometry.geometry_nd.Hypercube([0.00,0.00,0.00], 
                                          [1.38,1.67,1.23])

observe_x = np.load("./numerical-data/ray-2D/observe_x_20k.npy")
observe_y = dde.PointSetBC(observe_x, np.load("./numerical-data/ray-2D/observe_y_20k.npy"), component=0)

data = dde.data.PDE(
    geom,
    pde,
    [observe_y],
    num_domain=0,
    anchors=observe_x,
    train_distribution="pseudo"
    )

net = dde.maps.FNN([3] + [100] * 3 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(
    lambda x, u: u
    * x[:, 2:3]
    + ic(x)
)

def sample_spherical(npoints, ndim=2):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T

def sample_train_points(n):
    # model parameters, uniform sampling
    t = np.random.uniform(0,1,(n,1))

    space = sample_spherical(n) * np.random.uniform(0.0, 0.5, n).reshape((n,1))
    return np.hstack((space, t))

variable = dde.callbacks.VariableValue([D, rho], precision=6, period=10, filename=filepath + "variables.dat")


model = dde.Model(data, net)
model.compile("adam", 1e-3)

losshistory, _ = model.train(400000, callbacks=[variable])

model.compile("adam", 5e-4)

losshistory, _ = model.train(400000, callbacks=[variable])

model.save(filepath + "model.ckpt")
pr.lossplot(losshistory, save_png=True, save_html=True, folder=filepath)

inp = sample_train_points(200000)

save_png=True
save_html=True
show=False


def U_T(v, u):
    phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2)) / 0.01)
    u_t = dde.grad.jacobian(u*phi, v, j=2)
    return [u_t]
def RHS(v, u):
    phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2)) / 0.01)
    u_x = dde.grad.jacobian(u, v, j=0)
    u_y = dde.grad.jacobian(u, v, j=1)
    
    u_xx = dde.grad.jacobian(u_x*phi, v, j=0)
    u_yy = dde.grad.jacobian(u_y*phi, v, j=1)
    
    return [300*(0.13e-4*(u_xx + u_yy) + 0.025*phi*u*(1-u))]

def DIFFUSION(v, u):
    phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2)) / 0.01)
    u_x = dde.grad.jacobian(u, v, j=0)
    u_y = dde.grad.jacobian(u, v, j=1)
    
    u_xx = dde.grad.jacobian(u_x*phi, v, j=0)
    u_yy = dde.grad.jacobian(u_y*phi, v, j=1)
    
    return [300*0.13e-4*(u_xx + u_yy)]

def LOGISTIC(v, u):
    phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2)) / 0.01)
    return [300*phi*0.025*u*(1-u)]

out = model.predict(inp)
# pff_out = model.sess.run(pff(model.data.train_x.astype("float32")))
u_t = model.predict(inp, operator=U_T)[0]
rhs = model.predict(inp, operator=RHS)[0]
res = model.predict(inp, operator=pde)[0]
dif = model.predict(inp, operator=DIFFUSION)[0]
log = model.predict(inp, operator=LOGISTIC)[0]

pr.radialplot(inp, u_t, title="∂u/∂t v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="∂u/∂t", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="LHS", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, rhs, title="DΔu + ρu(1-u) v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="DΔu + ρu(1-u)", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="RHS", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, res, title="Residual v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Residual: ∂u/∂t - (DΔu + ρu(1-u))", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Residual", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, out, title="Tumor Concentration v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="u", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="u", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, dif, title="Diffusion v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Dif", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Diffusion", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, log, title="Logistic v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Log", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Log", folder=filepath, ics=[0., 0., 0.])
