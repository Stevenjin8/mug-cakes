import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm
from scipy.stats import multivariate_normal

from mug_cakes.kernel import rbf

plt.rcParams.update({"axes.labelsize": "x-large"})
X = np.linspace(0, 1, 100).reshape(-1, 1)


scale2 = 0.2**2
s2f = 1**2

K = rbf(X, X, scale2, s2f)

np.random.seed(42)
fig, ax = plt.subplots(figsize=(5, 4))
for _ in range(3):
    y = multivariate_normal.rvs(cov=K)
    ax.plot(X, y, "k")

ax.set(xlabel="$x$", ylabel="$f$")
fig.savefig("fig/gp-sample2d.png", bbox_inches="tight", dpi=500)


N = 50
X = np.linspace(0, 1, N)
Y = np.linspace(0, 1, N)
X, Y = np.meshgrid(X, Y)
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
XX = np.hstack((X, Y))

K = rbf(XX, XX, scale2, s2f)
Z = multivariate_normal.rvs(cov=K)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")

ax.plot_surface(
    X.reshape(N, N), Y.reshape(N, N), Z.reshape(N, N), linewidth=1, cmap=cm["bone"]
)
ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$f$")

fig.savefig("fig/gp-sample3d.png", dpi=500)
