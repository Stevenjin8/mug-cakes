import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mug_cakes import bo, gp, kernel, utils

np.random.seed(42)


def t(x):
    return np.sin(x * 10) + x - 0.5


xs = iter([0.78, 0.5, 0.48, 0.47, 1])

np.random.seed(42)
N = 5
s2 = 0.3**2
x_list = []
w = np.array([2, 0.1, -0.7])
y_list = []

w_0 = np.array([0, 0, 0])
V_0 = np.eye(3)
V_0inv = np.linalg.inv(V_0)


l2 = 0.12**2
s2f = 1**2

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))


def update(_, ax1):
    x_new = next(xs)
    x_list.append(x_new)
    y_list.append(t(x_new) + np.random.normal(scale=s2**0.5))
    X = np.array(x_list).reshape(-1, 1)
    y = np.array(y_list)
    B = np.zeros_like(y, dtype=np.uint64)
    dom = np.linspace(0, 1, 1000).reshape(-1, 1)
    K = kernel.rbf(X, X, l2, s2f)
    K_s = kernel.rbf(dom, X, l2, s2f)
    K_ss = kernel.rbf(dom, dom, l2, s2f)
    S = bo.full_cov(K, 1, B, s2, 0)[:-1, :-1]

    expected = gp.conditional_mean(y, np.linalg.inv(S), K_s) + 3.5
    var = gp.conditional_var(K_ss, np.linalg.inv(S), K_s)

    ax1.clear()
    ax1.plot(dom, expected, "k--", alpha=0.7, label="Expected")
    # ax1.plot(dom, t(dom), color="red", label="True value", alpha=0.7)
    ax1.fill_between(
        dom.reshape(-1),
        expected - 2 * var**0.5,
        expected + 2 * var**0.5,
        color="lightgray",
        label="95% Credible Interval",
    )
    ax1.scatter(X, y + 3.5, color="orange", alpha=1, label="Noisy Samples", s=120)
    ax1.legend(loc="upper left")
    ax1.set(xlim=[0, 1], ylim=[1.5, 7], xlabel="Proportion of Flour", ylabel="Quality")


fig.savefig(f"fig/sample_prior.png", dpi=500, bbox_inches="tight")
for i in range(5):
    update(None, ax1)
    fig.savefig(f"fig/sample_{i}.png", dpi=500, bbox_inches="tight")

X = np.array(x_list).reshape(-1, 1)
y = np.array(y_list)
B = np.zeros_like(y, dtype=np.uint64)
dom = np.linspace(0, 1, 1000).reshape(-1, 1)
K = kernel.rbf(X, X, l2, s2f)
K_s = kernel.rbf(dom, X, l2, s2f)
K_ss = kernel.rbf(dom, dom, l2, s2f)
S = bo.full_cov(K, 1, B, s2, 0)[:-1, :-1]

expected = gp.conditional_mean(y, np.linalg.inv(S), K_s) + 3.5
var = gp.conditional_var(K_ss, np.linalg.inv(S), K_s)

ax1.clear()
ax1.plot(dom, expected, "k--", alpha=0.7, label="Expected")
ax1.fill_between(
    dom.reshape(-1),
    expected - 2 * var**0.5,
    expected + 2 * var**0.5,
    color="lightgray",
    label="95% Credible Interval",
)
ax1.scatter(X[:, 0], y[:] + 3.5, color="orange", alpha=1, label="Noisy Samples", s=120)
ax1.legend(loc="upper left")
ax1.set(xlim=[0, 1], ylim=[1.5, 7], xlabel="Proportion of Flour", ylabel="Quality")


ax1.axvline(0.85, -1.5, 4)

ax1.axvline(0.5, -1.5, 4)

ax1.axvline(0.1, -1.5, 4)


fig.savefig(f"fig/sample_final.png", dpi=500, bbox_inches="tight")

#####################################################################################################
#####################################################################################################
#####################################################################################################

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
ax1.clear()
ax1.plot(dom, expected, "k--", alpha=0.7, label="Expected")
ax1.fill_between(
    dom.reshape(-1),
    expected - 2 * var**0.5,
    expected + 2 * var**0.5,
    color="lightgray",
    label="95% Credible Interval",
)
ax1.scatter(X[:, 0], y[:] + 3.5, color="orange", alpha=1, label="Noisy Samples", s=120)
ax1.legend(loc="upper left")
ax1.set(xlim=[0, 1], ylim=[1.5, 7], xlabel="Proportion of Flour", ylabel="Quality")

ax2.clear()
ax2.plot(
    dom.reshape(-1),
    stats.norm.cdf((expected - 2 - 3.5) / var**0.5),
    color="g",
)
ax2.set(
    xlim=[0, 1],
    ylim=[0, 0.03],
    ylabel=r"Probability of Improvement",
    xlabel="Proportion of Flour",
)

best = dom[stats.norm.cdf((expected - 2 - 3.5) / var**0.5).argmax()]
ax1.axvline(best, -1.5, 4)
ax2.axvline(best, 0, 1)

ax1.axvline(0.5, -1.5, 4)
ax2.axvline(0.5, 0, 1)

ax1.axvline(0.1, -1.5, 4)
ax2.axvline(0.1, 0, 1)

ax1.set(xlim=[0, 1], ylim=[1.5, 7], xlabel="", ylabel="Quality")
fig.savefig(f"fig/sample_ei.png", dpi=500, bbox_inches="tight")
