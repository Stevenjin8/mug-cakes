import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

from mug_cakes import bo, gp, kernel

X = np.loadtxt("data/experiments.txt")
y = X[:, -2]
B = X[:, -1]
X = X[:, :-2]
B = np.array(B, dtype=np.uint64)
N = X.shape[0]

# fig, ax = plt.subplots()
# ax.scatter([1,2,3,4], y[:4])
# ax.scatter(range(5, len(y) + 1), y[4:])

dist = {
    0: X[:, 0],
    1: X[:, 1],
    2: X[:, 2],
    3: X[:, 3],
}


fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(range(1, X.shape[0] + 1), X[:, 3] + X[:, 2] + X[:, 1] + X[:, 0], label="Milk", color="lightgrey")
ax.bar(range(1, X.shape[0] + 1), X[:, 2] + X[:, 1] + X[:, 0], label="Cocoa", color="Chocolate")
ax.bar(range(1, X.shape[0] + 1), X[:, 1] + X[:, 0], label="Sugar", color="LightBlue")
ax.bar(range(1, X.shape[0] + 1), X[:, 0], label="Flour", color="BurlyWood")

ax.set_xlabel("Iteration")
ax.set_ylabel("Proportion")
ax.legend(bbox_to_anchor=(0.3, 1.27))

fig.savefig("fig/progression.png", dpi=500, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5, 5))
xs = np.arange(B.shape[0])
# for i in range(int(B.max() + 1)):
#     xs = np.arange(B.shape[0])
#     ax.scatter(xs[B == i] + 1, y[B == i], label=f"Observer {i + 1}")
ax.scatter(xs + 1, y)
ax.legend(loc="upper left")

ax.set_xlabel("Iteration")
ax.set_ylabel("Score")


fig.savefig("fig/ys.png", dpi=500, bbox_inches="tight")

s2f = 1**2
scale2 = 0.08**2
s2e = 0.3**2
var_b = 0.2**2
J = 2

fig, ax = plt.subplots(figsize=(5, 5))

K = kernel.rbf(X, X, scale2=scale2, s2f=s2f)
full_var = bo.full_cov(K, J, B, s2e, var_b)
mean = gp.conditional_mean(
    y,
    np.linalg.inv(full_var[:N, :N]),
    full_var[N:, :N],
)
var = gp.conditional_covar(
    full_var[N:, N:],
    np.linalg.inv(full_var[:N, :N]),
    full_var[N:, :N],
)

mean_diff = mean[0] - mean[1]
var_diff = var[0,0] + var[1, 1] - 2 * var[1, 0]

dom = np.linspace(-1.25, 0.5, 1000)
img = stats.norm.pdf(dom, loc=mean_diff, scale=var_diff ** 0.5)
ax.plot(dom, img, "gray", label="Density")
dom = np.linspace(mean_diff - 2 * var_diff ** 0.5, mean_diff + 2 * var_diff ** 0.5, 1000)
img = stats.norm.pdf(dom, loc=mean_diff, scale=var_diff ** 0.5)
ax.fill_between(dom, img, color="lightgrey", label="95% Credible Interval")
ax.vlines(0, 0, img.max(), "r", "dashed", label="0")

ax.set_xlabel("$b_0 - b_1$")
ax.set_ylabel("Probability Density")
ax.legend(loc="upper left")

fig.savefig("fig/bais-diff.png", dpi=500, bbox_inches="tight")

#plt.show()
