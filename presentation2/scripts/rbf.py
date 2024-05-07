import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 4))
dom = np.linspace(-4, 4, 1000)
img = np.exp(- dom * dom / 2)
ax.plot(dom, img)
ax.set(ylabel="$\kappa$", xlabel=r"$||\mathbf{x} - \mathbf{x}'||$")
fig.savefig(f"fig/rbf.png", dpi=500, bbox_inches="tight")
