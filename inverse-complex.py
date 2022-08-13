# Scientific Libraries
import numpy as np
np.set_printoptions(edgeitems=10,linewidth=300)
import deepxde as dde
import tensorflow as tf

# System/Timing Libraries
from datetime import datetime
import sys
sys.path.insert(1, '..')

# Plotting Libraries
import plotly.express as px
import plotly.graph_objects as go
import PlotlyReference as pr

filepath=""
print(filepath)

D = tf.Variable(1e-3)
rho = tf.Variable(1e-3)
### LOAD DIFFERENT GEOMETRIES ###
# load pff
x = np.linspace(0.005, 1.515, 152)
y = np.linspace(0.005, 1.845, 185)
z = np.linspace(0.005, 1.565, 157)
xx, yy, zz = np.meshgrid(x,y,z, indexing="ij")

def pff_data():
    img = nib.load("./scans/Atlas_pff.nii")
    a = np.array(img.dataobj)[:, :, :, 0]
    a = a[0:185, 0:152, 0:157]
    return a

def gm_data():
    img = nib.load("./scans/Atlas_gm.nii")
    a = np.flip(np.transpose(np.array(img.dataobj)[:, :, :, 0], axes=[1,0,2]), 0)
    a = a[22:207, 22:174, 6:163]
    return a

def wm_data():
    img = nib.load("./scans/Atlas_wm.nii")
    a = np.flip(np.transpose(np.array(img.dataobj)[:, :, :, 0], axes=[1,0,2]), 0)
    a = a[22:207, 22:174, 6:163]
    return a

def csf_data():
    img = nib.load("./scans/Atlas_csf.nii")
    a = np.flip(np.transpose(np.array(img.dataobj)[:, :, :, 0], axes=[1,0,2]), 0)
    a = a[22:207, 22:174, 6:163]
    return a

## INTERPOLATIONS
pff = pff_data()
gm = gm_data()
wm = wm_data()
csf = csf_data()

tissue = wm + gm
pWM = (tissue > csf)*wm
pGM = (tissue > csf)*gm

tissue = pWM + pGM
pWM = np.where(tissue > 0, pWM / tissue, 0.)
pGM = np.where(tissue > 0, pGM / tissue, 0.)

def pff_np(v):
    return pff[np.int32(100*v[:, 1:2]), np.int32(100*v[:, 0:1]), np.int32(100*v[:, 2:3])]
pff_mapped = pff_np(np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())))

def diff_np(v):
    return (pWM + 0.1*pGM)[np.int32(100*v[:, 1:2]), np.int32(100*v[:, 0:1]), np.int32(100*v[:, 2:3])]

diff_mapped = diff_np(np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())))

def pff(v):
    last = v[:, 2:3]*0 + 99.
    
    return interp3d.interpolate(pff_mapped.reshape((152, 185, 157, 1)).astype("float32"), tf.concat([100*v[:, 0:2], last], axis=1))*(0.5 + 0.5*tf.tanh((0.47 - tf.sqrt((v[:, 0:1]-1.05)**2 + (v[:, 1:2]-1.20)**2)) / 0.01))
def diff(v):
    last = v[:, 2:3]*0 + 99.
    
    return interp3d.interpolate(diff_mapped.reshape((152, 185, 157, 1)).astype("float32"), tf.concat([100*v[:, 0:2], last], axis=1))

# creates a 0.05 radius interval/circle/sphere/hypersphere IC
def ic(x):
    r = tf.square(x[:, 0:1]-1.05) + tf.square(x[:, 1:2]-1.20)
    return 0.5*tf.exp(-1000*r)

# define reaction-diffusion Fisher-Kolmogorov equations
def pde(v, u):
    d = diff(v)
    phi = pff(v)
    u_t = dde.grad.jacobian(u*phi, v, j=2)
    u_x = dde.grad.jacobian(u, v, j=0)
    u_y = dde.grad.jacobian(u, v, j=1)
    
    u_xx = dde.grad.jacobian(u_x*phi*d, v, j=0)
    u_yy = dde.grad.jacobian(u_y*phi*d, v, j=1)

    return [u_t - 150*(1e-2*D*(u_xx + u_yy) + 10*rho*phi*u*(1-u))]


# remember to check the cutoffs of the cropped tumor bounds
geom = dde.geometry.geometry_nd.Hypercube([0.00,0.00,0.00], 
                                          [2,2,2])

observe_x = np.load("./DDA/2D-MRI/diffusion-and-boundary/atlas-pff/random-sampling/1/observe_x_26830.npy")
observe_y = dde.PointSetBC(observe_x, np.load("./DDA/2D-MRI/diffusion-and-boundary/atlas-pff/random-sampling/1/observe_y_26830.npy"), component=0)

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

    space = sample_spherical(n) * 0.5 * (np.random.uniform(0.0, 1, n)**0.75).reshape((n,1)) + np.array([[1.05, 1.20]])
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


def figures(v, u):
    d = diff(v)
    phi = pff(v)
    u_t = dde.grad.jacobian(u*phi, v, j=2)
    u_x = dde.grad.jacobian(u, v, j=0)
    u_y = dde.grad.jacobian(u, v, j=1)
    
    u_xx = dde.grad.jacobian(u_x*phi*d, v, j=0)
    u_yy = dde.grad.jacobian(u_y*phi*d, v, j=1)

    RHS = 150*(0.13e-4*(u_xx + u_yy) + 0.025*phi*u*(1-u))
    DIFFUSION = 150*(0.13e-4*(u_xx + u_yy))
    LOGISTIC = 150*(0.025*phi*u*(1-u))
    RES = u_t - 150*(0.13e-4*(u_xx + u_yy) + 0.025*phi*u*(1-u))

    return [u*phi, u_t, RHS, RES, DIFFUSION, LOGISTIC]

out, u_t, rhs, res, dif, log = model.predict(inp, figures)

pr.radialplot(inp, u_t, title="∂u/∂t v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="∂u/∂t", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="LHS", folder=filepath, ics=[1.05, 1.20, 0.])

pr.radialplot(inp, rhs, title="DΔu + ρu(1-u) v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="DΔu + ρu(1-u)", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="RHS", folder=filepath, ics=[1.05, 1.20, 0.])

pr.radialplot(inp, res, title="Residual v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Residual: ∂u/∂t - (DΔu + ρu(1-u))", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Residual", folder=filepath, ics=[1.05, 1.20, 0.])

pr.radialplot(inp, out, title="Tumor Concentration v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="u", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="u", folder=filepath, ics=[1.05, 1.20, 0.])

pr.radialplot(inp, dif, title="Diffusion v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Dif", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Diffusion", folder=filepath, ics=[1.05, 1.20, 0.])

pr.radialplot(inp, log, title="Logistic v. Radial Distance", xaxis="Distance from Tumor Origin", yaxis="Log", height=1080, save_png=save_png, save_html=save_html, show=show, size=22, save_title="Log", folder=filepath, ics=[1.05, 1.20, 0.])
