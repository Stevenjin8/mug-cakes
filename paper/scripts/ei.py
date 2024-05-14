import matplotlib.pyplot as plt
import numpy as np

from mug_cakes import bo, gp, kernel
plt.rcParams.update({'axes.labelsize': 'x-large'})

SIZE = 100
X = np.linspace(0, 1, 5).reshape(-1, 1)
y = np.array([0, 1, 0, 1.3, 0])
s2f = 0.5**2
ell2 = 0.2**2
s2e = 0.2**2
K = kernel.rbf(X, X, scale2=ell2, s2f=s2f)
S = K + s2e * np.eye(X.shape[0])

X_star = np.linspace(0, 1, 1000).reshape(-1, 1)
K_star = kernel.rbf(X_star, X, ell2, s2f)
K_2star = kernel.rbf(X_star, X_star, ell2, s2f)
post_mean = gp.conditional_mean(y, np.linalg.inv(S), K_star)
post_var = gp.conditional_var(K_2star, np.linalg.inv(S), K_star)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
ax1.fill_between(
    X_star.flatten(),
    post_mean - 2 * post_var**0.5,
    post_mean + 2 * post_var**0.5,
    color="lightgrey",
)
ax1.set_ylim(-0.4, 1.6)
ax1.plot(X_star, post_mean, "k--")
ax1.set_ylabel("$f$")

x_M = np.array([0.75])
ax1.vlines(x_M, -2, 2, "orangered", "dashed")
ax2.vlines(x_M, -2, 2, "orangered", "dashed")
# ax3.vlines(x_M, -2, 2, "orangered", "dashed")


ax1.scatter(X[:3], y[:3], s=SIZE)
ax1.scatter([X[3]], [y[3]], c="red", s=SIZE)
ax1.scatter(X[4:], y[4:], c="#1f77b4", s=SIZE)


eis = []
for x in X_star:
    eis.append(-bo.minus_expected_diff(x, x_M, X, y, np.linalg.inv(S), ell2, s2f))
ax2.plot(X_star, eis, "#006400")
ax2.set_ylim(-0.001, 0.03)
ax2.set_ylabel("$a_{vEI}$")
ax2.set_xlabel("$x$")


eis = []
post_mean_M = gp.conditional_mean(y, np.linalg.inv(S), K)[3]
post_var_M = gp.conditional_mean(y, np.linalg.inv(S), K)[3]
for x in X_star:
    gp.expected_improvement
    eis.append(-bo.minus_expected_diff(x, x_M, X, y, np.linalg.inv(S), ell2, s2f))
# ax3.plot(X_star, eis)
# ax3.set_ylim(-0.001, 0.03)
# ax3.set_ylabel("vEI")

fig.savefig("fig/ei.png", dpi=400, bbox_inches="tight")
