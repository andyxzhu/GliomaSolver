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

filepath="./DDA/3D-MRI/inverse-spherical/inverse-ray/18/"
print(filepath)

D = tf.Variable(1e-3)
rho = tf.Variable(1e-2)

# creates a 0.05 radius interval/circle/sphere/hypersphere IC
def ic(x):
    r = tf.square(x[:, 0:1]) + tf.square(x[:, 1:2]) + tf.square(x[:, 2:3])
    return 0.1*tf.exp(-1000*r)

# define reaction-diffusion Fisher-Kolmogorov equations
def pde(v, u):
    phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2 + v[:, 2:3]**2)) / 0.01)
    u_t = dde.grad.jacobian(u*phi, v, j=3)
    u_x = dde.grad.jacobian(u, v, j=0)
    u_y = dde.grad.jacobian(u, v, j=1)
    u_z = dde.grad.jacobian(u, v, j=2)
    
    u_xx = dde.grad.jacobian(u_x*phi, v, j=0)
    u_yy = dde.grad.jacobian(u_y*phi, v, j=1)
    u_zz = dde.grad.jacobian(u_z*phi, v, j=2)
    
    return [u_t - 300*(1e-2*D*(u_xx + u_yy + u_zz) + rho*phi*u*(1-u))]


# remember to check the cutoffs of the cropped tumor bounds
geom = dde.geometry.geometry_nd.Hypercube([0.00,0.00,0.00,0.00], 
                                          [1.38,1.67,1.23,1.00])

observe_x = np.load("./numerical-data/ray-3D/observe_x_50k_t60-300-fixed.npy")
observe_y = dde.PointSetBC(observe_x, np.load("./numerical-data/ray-3D/observe_y_50k_t60-300-fixed.npy"), component=0)

data = dde.data.PDE(
    geom,
    pde,
    [observe_y],
    num_domain=0,
    anchors=observe_x,
    train_distribution="pseudo"
    )

net = dde.maps.FNN([4] + [100] * 3 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(
    lambda x, u: u
    * x[:, 3:4]
    + ic(x)
)

variable = dde.callbacks.VariableValue([D, rho], precision=6, period=10, filename=filepath + "variables.dat")
model = dde.Model(data, net)

model.compile("adam", 1e-3, loss_weights=[1., 10.])
losshistory, _ = model.train(400000, callbacks=[variable])

model.save(filepath + "model.ckpt")

model.compile("adam", 5e-4, loss_weights=[1., 10.])
losshistory, _ = model.train(400000, callbacks=[variable])

model.save(filepath + "model.ckpt")

pr.lossplot(losshistory, save_png=True, save_html=True, folder=filepath)

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T

def sample_train_points(n):
    # model parameters, uniform sampling
    t = np.random.uniform(0,1,(n,1))

    space = sample_spherical(n) * np.random.uniform(0.0, 0.5, n).reshape((n,1))
    return np.hstack((space, t))

inp = sample_train_points(200000)

save_png=True
save_html=True
show=False

def figures(v, u):
    phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2 + v[:, 2:3]**2)) / 0.01)
    u_t = dde.grad.jacobian(u*phi, v, j=3)
    u_x = dde.grad.jacobian(u, v, j=0)
    u_y = dde.grad.jacobian(u, v, j=1)
    u_z = dde.grad.jacobian(u, v, j=2)
    
    u_xx = dde.grad.jacobian(u_x*phi, v, j=0)
    u_yy = dde.grad.jacobian(u_y*phi, v, j=1)
    u_zz = dde.grad.jacobian(u_z*phi, v, j=2)

    DIFFUSION = 300*(1e-2*D*(u_xx + u_yy + u_zz))
    LOGISTIC = 300*(rho*phi*u*(1-u))
    RHS = DIFFUSION + LOGISTIC
    RES = u_t - RHS

    return [u, u_t, RHS, RES, DIFFUSION, LOGISTIC]

out, u_t, rhs, res, dif, log = model.predict(inp, figures)

pr.radialplot(inp, u_t, title="∂u/∂t v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="∂u/∂t", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="LHS", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, rhs, title="DΔu + ρu(1-u) v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="DΔu + ρu(1-u)", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="RHS", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, res, title="Residual v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Residual: ∂u/∂t - (DΔu + ρu(1-u))", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Residual", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, out, title="Tumor Concentration v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="u", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="u", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, dif, title="Diffusion v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Dif", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Diffusion", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(inp, log, title="Logistic v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Log", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Log", folder=filepath, ics=[0., 0., 0.])

pr.radialplot(observe_x, model.predict(observe_x) - np.load("./numerical-data/ray-3D/observe_y_50k_t0-200-fixed.npy"), title="Observation Residual v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Observation Residual", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="OBS_RES", ics=[0., 0., 0.], folder=filepath)
