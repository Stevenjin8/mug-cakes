import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from mug_cakes import bo, gp, kernel

np.random.seed(47)
N = 10
X = np.random.rand(N).reshape(-1, 1)
X[:5] *= 0.7
X[5:] = X[5:] * 0.7 + 0.4
Z = np.zeros(N, dtype=np.uint64)
Z[:5] = 1
B = np.array([-1, 1], dtype=np.float64)
N_b = 2

scale2 = 0.1**2
s2f = 0.3**2
var_b = 1**2
s2e = 0.5**2
K = kernel.rbf(X, X, scale2, s2f)

cov2 = bo.full_cov(K, N_b, Z, s2f, 0)[:N, :N]
y = multivariate_normal(cov=cov2).rvs().flatten()
y += B[Z]

cov1 = K.copy()
cov2 = bo.full_cov(K, N_b, Z, s2f, 0)[:N, :N]
cov3 = bo.full_cov(K, N_b, Z, s2f, var_b)[:N, :N]

N_star = 4000
X_star = np.linspace(0, 1, N_star).reshape(-1, 1)
k_s = kernel.rbf(X, X_star, scale2, s2f)
K_2s = kernel.rbf(X_star, X_star, scale2, s2f)

m1 = gp.conditional_mean(y, np.linalg.inv(cov1), k_s.T)
v1 = gp.conditional_var(K_2s, np.linalg.inv(cov1), k_s.T)
fig, ax = plt.subplots()
ax.fill_between(X_star.flatten(), m1 - 2 * v1**0.5, m1 + 2 * v1**0.5, color="lightblue")
ax.plot(X_star, m1)
ax.scatter(X[Z == 0], y[Z == 0])
ax.scatter(X[Z == 1], y[Z == 1])
ax.set_ylim(-1.5, 1.5)
fig.savefig("fig/noiseless-posterior.png", dpi=500, bbox_inches="tight")

m2 = gp.conditional_mean(y, np.linalg.inv(cov2), k_s.T)
v2 = gp.conditional_var(K_2s, np.linalg.inv(cov2), k_s.T)
fig, ax = plt.subplots()
ax.fill_between(X_star.flatten(), m2 - 2 * v2**0.5, m2 + 2 * v2**0.5, color="lightblue")
ax.plot(X_star, m2)
ax.scatter(X[Z == 0], y[Z == 0])
ax.scatter(X[Z == 1], y[Z == 1])
ax.set_ylim(-1.5, 1.5)
fig.savefig("fig/noisy-posterior.png", dpi=500, bbox_inches="tight")


m3 = gp.conditional_mean(y, np.linalg.inv(cov3), k_s.T)
v3 = gp.conditional_var(K_2s, np.linalg.inv(cov3), k_s.T)
fig, ax = plt.subplots()
ax.fill_between(X_star.flatten(), m3 - 2 * v3**0.5, m3 + 2 * v3**0.5, color="lightblue")
ax.plot(X_star, m3)
ax.scatter(X[Z == 0], y[Z == 0])
ax.scatter(X[Z == 1], y[Z == 1])
ax.set_ylim(-1.5, 1.5)

fig.savefig("fig/biased-posterior.png", dpi=500, bbox_inches="tight")
