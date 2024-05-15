import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mug_cakes import bo, gp, kernel


def t(x):
    return np.sin(x * 10) + x - 0.5


np.random.seed(45)
N = 5
s2 = 0.3**2
x_list = [0.8]
w = np.array([2, 0.1, -0.7])
y_list = [t(0.8) + np.random.normal(scale=s2**0.5)]

w_0 = np.array([0, 0, 0])
V_0 = np.eye(3)
V_0inv = np.linalg.inv(V_0)


l2 = 0.12**2
s2f = 1

fig, ax = plt.subplots()


def update(_, ax):
    X = np.array(x_list).reshape(-1, 1)
    y = np.array(y_list)
    B = np.zeros_like(y, dtype=np.uint64)
    dom = np.linspace(-1, 1, 1000).reshape(-1, 1)
    K = kernel.rbf(X, X, l2, s2f)
    K_s = kernel.rbf(dom, X, l2, s2f)
    K_ss = kernel.rbf(dom, dom, l2, s2f)
    S = bo.full_cov(K, 1, B, s2, 0)[:-1, :-1]

    expected = gp.conditional_mean(y, np.linalg.inv(S), K_s)
    var = gp.conditional_var(K_ss, np.linalg.inv(S), K_s)

    ax.clear()
    ax.plot(dom, t(dom), color="red", label="True value")
    ax.plot(dom, expected, "k--", alpha=0.7, label="Expected")
    ax.fill_between(
        dom.reshape(-1),
        expected - 2 * var**0.5,
        expected + 2 * var**0.5,
        color="lightgray",
        label="95% Credible Interval",
    )
    # ax.axvline(0.795, -100, 100, color="red", label="True Max")
    # ax.axvline(dom[expected.argmax()], -100, 100, color="k", label="Est. Max")
    ax.scatter(X[:, 0], y, color="orange", alpha=0.9, label="Noisy Samples")
    ax.legend(loc="upper left")
    ax.set(xlim=[0, 1], ylim=[-1.5, 3], xlabel="$y$", ylabel="$F$")

    ei = stats.norm.cdf((expected - 2) / var**0.5)
    x_new = dom[ei.argmax(), 0]
    x_list.append(x_new)
    y_list.append(t(x_new) + np.random.normal(scale=s2**0.5))


np.random.seed(46)
ani = animation.FuncAnimation(
    fig=fig, func=lambda x: update(x, ax), frames=120, interval=100
)
ani.save(filename="fig/bo-limiting.gif", writer="ffmpeg", dpi=400)
