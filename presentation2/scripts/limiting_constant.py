import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(43)
theta = 0
s2 = 1
x_list = [-0.5]

fig, ax = plt.subplots()


def update(_):
    x_new = np.random.normal(scale=s2**0.5)
    x_list.append(x_new)
    X = np.array(x_list)
    N = len(x_list)

    s2_hat = X.var(ddof=1) / (N)
    mu_hat = X.mean()

    dom = np.linspace(-2, 2, 1000)
    ax.clear()
    ax.fill_between(
        dom,
        stats.norm.pdf(dom, loc=mu_hat, scale=s2_hat**0.5),
        color="lightgray",
        label="Posterior Density",
    )
    ax.axvline(0, 0, 5, color="red", label="True Value")
    ax.axvline(mu_hat, 0, 5, color="k", linestyle="--", label="Posterior Mean")
    ax.scatter(
        x_list,
        np.zeros_like(X) + 0.3,
        color="orange",
        alpha=0.4,
        label="Noisy Observations",
    )
    ax.legend()
    ax.set(xlim=[-2.5, 2.5], ylim=[0, 5])


ani = animation.FuncAnimation(fig=fig, func=update, frames=200, interval=50)
# plt.show()
ani.save(filename="fig/linear_constant.mp4", writer="ffmpeg", dpi=1000)
