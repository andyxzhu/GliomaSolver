# GliomaSolver
Glioblastoma is an aggressive brain tumor with cells that infiltrate and proliferate rapidly into surrounding brain tissue. Recently, physics-informed neural networks (PINNs) have emerged as a novel method in scientific machine learning for solving nonlinear PDEs. Compared to traditional solvers, PINNs leverage unsupervised deep learning methods to minimize residuals across mesh-free domains, enabling greater flexibility while avoiding the need for complex grid constructions. 

Here, we implement a general method for solving time-dependent diffusion-reaction PDE models of glioblastoma and inferring biophysical parameters from numerical data via PINNs. We evaluate the PINNs over patient-specific geometries, accounting for individual variations with diffusion mobilities derived from pre-operative MRI scans. Using synthetic data, we demonstrate the performance of our algorithm in patient-specific geometries. 

We show that PINNs are capable of solving parameter inference inverse problems in approximately one hour, expediting previous approaches by 20--40 times owing to the robust interpolation capabilities of machine learning algorithms.
