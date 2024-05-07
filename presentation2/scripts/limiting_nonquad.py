import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def t(x):
    return -np.sin(x * 10) + x - 0.5


np.random.seed(42)
N = 5
s2 = 0.2**2
x_list = [0]
w = np.array([2, 0.1, -0.7])
y_list = [t(0) + np.random.normal(scale=s2**0.5)]

w_0 = np.array([0, 0, 0])
V_0 = np.eye(3)
V_0inv = np.linalg.inv(V_0)

fig, ax = plt.subplots()


def update(_):
    x_new = np.random.rand()
    x_list.append(x_new)
    y_list.append(t(x_new) + np.random.normal(scale=s2**0.5))
    N = len(x_list)
    X = np.ones((N, 3))
    X[:, 1] = x_list
    X[:, 2] = x_list
    X[:, 2] = X[:, 2] * X[:, 2]
    y = np.array(y_list)
    V_N = s2 * np.linalg.inv(s2 * V_0inv + X.T @ X)
    V_Ninv = V_0inv + 1 / s2 * X.T @ X
    w_N = V_N @ V_0inv @ w_0 + 1 / s2 * V_N @ X.T @ y
    print(w_N)

    dom = np.linspace(-1, 1, 1000)
    expected = dom * dom * w_N[2] + dom * w_N[1] + w_N[0]
    var = (
        V_N[0, 0]
        + dom**1 * 2 * V_N[0, 1]
        + dom**2 * (V_N[1, 1] + 2 * V_N[0, 2])
        + dom**3 * (2 * V_N[2, 1])
        + dom**4 * V_N[2, 2]
    )

    ax.clear()
    ax.plot(dom, t(dom), color="red", label="True value")
    ax.plot(dom, expected, "k--", alpha=0.7, label="Expected")
    ax.fill_between(
        dom,
        expected - 2 * var**0.5,
        expected + 2 * var**0.5,
        color="lightgray",
        label="95% Credible Interval",
    )
    ax.scatter(X[:, 1], y, color="orange", alpha=0.3, label="Noisy Samples")
    ax.legend(loc="upper left")
    ax.set(xlim=[0, 1], ylim=[-1.5, 3])


ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=100)
ani.save(filename="fig/nonquad-limiting.gif", writer="ffmpeg", dpi=400)
