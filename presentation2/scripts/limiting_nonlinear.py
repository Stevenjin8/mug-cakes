import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def t(x):
    return np.abs(np.sin(x * 10) + 3 * (x - 0.5) ** 2) * x ** 2 * 2.1 + 1


np.random.seed(42)
N = 5
s2 = 0.2**2
x_list = [0.5]
y_list = [t(0.5)]

w_0 = np.array([0, 0])
V_0 = np.eye(2)
V_0inv = np.linalg.inv(V_0)

fig, ax = plt.subplots()


def update(_):
    x_new = np.random.rand()
    x_list.append(x_new)
    y_list.append(t(x_new) + np.random.normal(scale=s2**0.5))
    N = len(x_list)
    X = np.ones((N, 2))
    X[:, 0] = x_list
    y = np.array(y_list)
    V_N = s2 * np.linalg.inv(s2 * V_0inv + X.T @ X)
    w_N = V_N @ V_0inv @ w_0 + 1 / s2 * V_N @ X.T @ y

    dom = np.linspace(0, 1, 500)
    expected = dom * w_N[0] + w_N[1]
    var = V_N[1, 1] + dom * dom * V_N[0, 0] + 2 * dom * V_N[0, 1]
    ax.clear()
    ax.plot(dom, t(dom) , color="red", label="True value")
    ax.plot(dom, expected, "k--", alpha=0.7, label="Expected value")
    ax.fill_between(
        dom,
        expected - 2 * var**0.5,
        expected + 2 * var**0.5,
        color="lightgray",
        label="95% Credible Interval",
    )
    ax.scatter(X[:, 0], y, color="orange", alpha=0.3, label="Noisy Samples")
    ax.legend()
    ax.set(xlim=[0, 1], ylim=[0.7, 3.3])


ani = animation.FuncAnimation(fig=fig, func=update, frames=200, interval=50)
ani.save(filename="fig/nonlinear-limiting.mp4", writer="ffmpeg", dpi=1000)
