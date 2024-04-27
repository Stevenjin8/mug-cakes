import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot([0, 1], [0, 1], "r")
ax.scatter([1], [1], color="red", marker="x")
fig.savefig("fig/linear.png", dpi=1000)
