import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from mug_cakes import bo, gp, kernel

np.random.seed(42)


def t(x):
    return np.sin(x * 10) + x


np.random.seed(42)
N = 5
s2 = 0.2**2
x_list = [0]
w = np.array([2, 0.1, -0.7])
y_list = [t(0) + np.random.normal(scale=s2**0.5)]

w_0 = np.array([0, 0, 0])
V_0 = np.eye(3)
V_0inv = np.linalg.inv(V_0)


l2 = 0.1**2
s2f = 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))


def update(_, ax1, ax2):
    x_new = np.random.rand()
    x_list.append(x_new)
    y_list.append(t(x_new) + np.random.normal(scale=s2**0.5))
    X = np.array(x_list).reshape(-1, 1)
    y = np.array(y_list)
    B = np.zeros_like(y[:-1], dtype=np.uint64)
    dom = np.linspace(0, 1, 1000).reshape(-1, 1)
    K = kernel.rbf(X[:-1], X[:-1], l2, s2f)
    K_s = kernel.rbf(dom, X[:-1], l2, s2f)
    K_ss = kernel.rbf(dom, dom, l2, s2f)
    S = bo.full_cov(K, 1, B, s2, 0)[:-1, :-1]

    expected = gp.conditional_mean(y[:-1], np.linalg.inv(S), K_s)
    var = gp.conditional_var(K_ss, np.linalg.inv(S), K_s)

    ax1.clear()
    ax1.plot(dom, t(dom), color="red", label="True value")
    ax1.plot(dom, expected, "k--", alpha=0.7, label="Expected value")
    ax1.fill_between(
        dom.reshape(-1),
        expected - 2 * var**0.5,
        expected + 2 * var**0.5,
        color="lightgray",
        label="95% Credible Interval",
    )
    ax1.scatter(X[:-1, 0], y[:-1], color="orange", alpha=0.8, label="Noisy Samples")
    ax1.axvline([X[-1, 0]], -1.5, 4, color="b", linestyle="--", label="Next sample")
    ax1.legend(loc="upper left")
    ax1.set(xlim=[0, 1], ylim=[-1.5, 4])

    ax2.clear()
    ax2.fill_between(
        dom.reshape(-1),
        var**0.5,
        color="lightgray",
    )
    ax2.axvline([X[-1, 0]], 0, 1, color="b", linestyle="--", label="Next sample")
    ax2.set(xlim=[0, 1], ylim=[0, 1], ylabel=r"$\sigma$")


for i in range(20):
    update(None, ax1, ax2)
    fig.savefig(f"fig/sample_{i}.png", dpi=700)
