import matplotlib.pyplot as plt
import numpy as np

from mug_cakes import utils


def t(x):
    return -np.sin(x * 10) + x + 2.5


dom = np.linspace(0, 1, 1000)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set(ylim=(3 - 1.4, 4.4), xlabel="Proportion of Flour", ylabel="Quality")

ax.plot(dom, dom - 0.55 + 3, "k--", label="Expected")
ax.legend(loc="upper left")
fig.savefig("fig/linear.png", dpi=700, bbox_inches="tight")
ax.plot(dom, t(dom), "r", label="True")
ax.legend(loc="upper left")
fig.savefig("fig/linear-over.png", dpi=700, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5, 5))
ax.set(ylim=(3 - 1.4, 4.4), xlabel="Propotion of Flour", ylabel="Quality")

betas = np.array([-1.19318736 + 3, 3.83517873, -2.7453495])
ax.plot(dom, dom * dom * betas[2] + dom * betas[1] + betas[0], "k--", label="Expected")
ax.legend(loc="upper left")
fig.savefig("fig/quadratic.png", dpi=700, bbox_inches="tight")
ax.plot(dom, t(dom), "r", label="True")
ax.legend(loc="upper left")
fig.savefig("fig/quadratic-over.png", dpi=700, bbox_inches="tight")
