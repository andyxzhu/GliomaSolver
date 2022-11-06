# Scientific Libraries
import numpy as np
np.set_printoptions(edgeitems=10,linewidth=300)
import deepxde as dde
import tensorflow as tf
import nibabel as nib
from scipy.interpolate import interpn
from tensorflow_graphics.math.interpolation import trilinear as interp3d
import scipy.io
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

filepath="./DDA/2D-MRI/inverse-ray/complex/2/"
print(filepath)

D = tf.Variable(1e-2)
rho = tf.Variable(1e-2)

# creates a 0.05 radius interval/circle/sphere/hypersphere IC
def ic(x):
    r = tf.square(x[:, 0:1]) + tf.square(x[:, 1:2])
    return 0.1*tf.exp(-1000*r)

# define reaction-diffusion Fisher-Kolmogorov equations
def pde(v, u):
    # phi = 1
    phi = 0.5 + 0.5*tf.tanh((0.25 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2)) / 0.01)
    # phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(v[:, 0:1]**2 + v[:, 1:2]**2)) / 0.01)
    u_t = dde.grad.jacobian(u*phi, v, j=2)
    u_x = dde.grad.jacobian(u, v, j=0)
    u_y = dde.grad.jacobian(u, v, j=1)
    
    u_xx = dde.grad.jacobian(u_x*phi, v, j=0)
    u_yy = dde.grad.jacobian(u_y*phi, v, j=1)
    
    dist = tf.sqrt(tf.reduce_sum(tf.square(v[:, 0:2] - tf.constant([[1.2, 1.59]], dtype=tf.float32)), 1))
    weights = 1/(dist+1e-4)
    # return [(u_t - ((dDdx*phi + d*dphidx)*u_x + d*phi*u_xx + (dDdy*phi+d*dphidy)*u_y + d*phi*u_yy + rho*phi*u*(1-u)))]   
    # return [(u_t - 150*(1e-3*D*(u_xx + u_yy) + rho*phi*u*(1-u)))]
    return [(u_t - 150*(1e-3*D*(u_xx + u_yy) + rho*phi*u*(1-u))) * tf.reshape(tf.sqrt(weights), (-1, 1)) * tf.sqrt(tf.cast(tf.shape(v)[0], tf.float32)/tf.reduce_sum(weights))]


# remember to check the cutoffs of the cropped tumor bounds
geom = dde.geometry.geometry_nd.Hypercube([0.00,0.00,0.00], 
                                          [2,2,2])

observe_x = np.load("numerical-data/ray-2D-complex/observe_x_uniform_t27-150.npy")
idx = np.argwhere(((observe_x[:, 0]-1.2)**2 + (observe_x[:, 1]-1.59)**2) <= 0.25**2)[:, 0]
observe_x = observe_x[idx]
observe_x[:, 0] -= 1.2
observe_x[:, 1] -= 1.59

observe_y = dde.PointSetBC(observe_x, np.load("numerical-data/ray-2D-complex/observe_y_uniform_t27-150.npy")[idx], component=0, weights=1/(np.sqrt((observe_x[:, 0])**2 + (observe_x[:, 1])**2)+1e-4))

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
    t = np.random.uniform(0.18,1,(n,1))

    space = sample_spherical(n) * np.random.uniform(0.0, 0.25, (n,1))
    return np.hstack((space, t))

variable = dde.callbacks.VariableValue([D, rho], precision=6, period=10, filename=filepath + "variables.dat")


model = dde.Model(data, net)
model.compile("adam", 1e-3, loss_weights=[1., 10.])

losshistory, _ = model.train(400000, callbacks=[variable])

model.compile("adam", 5e-4, loss_weights=[1., 10.])

losshistory, _ = model.train(400000, callbacks=[variable])

model.save(filepath + "model.ckpt")
pr.lossplot(losshistory, save_png=True, save_html=True, folder=filepath)