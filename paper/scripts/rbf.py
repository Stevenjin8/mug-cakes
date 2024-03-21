import numpy as np
import matplotlib.pyplot as plt

from mug_cakes.kernel import rbf

X_star = np.linspace(-3, 3, 1000).reshape(-1, 1)
zero = np.array([[0]])
y1 = rbf(X_star, zero, 1**2, 1**2)
y2 = rbf(X_star, zero, 0.7**2, 1**2)
y3 = rbf(X_star, zero, 1**2, 0.7**2)
y4 = rbf(X_star, zero, 0.7**2, 0.7**2)

fig, ax = plt.subplots()
ax.plot(X_star, y1, label=r"$\ell^2 = 1^2, \sigma^2_f=1^2$")
ax.plot(X_star, y2, label=r"$\ell^2 = 0.7^2, \sigma^2_f=1^2$")
ax.plot(X_star, y3, label=r"$\ell^2 = 1^2, \sigma^2_f=0.7^2$")
ax.plot(X_star, y4, label=r"$\ell^2 = 0.7^2, \sigma^2_f=0.7^2$")

ax.legend()
fig.savefig("fig/rbf.png", dpi=400)
