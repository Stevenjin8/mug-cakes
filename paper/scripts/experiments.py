import matplotlib.pyplot as plt
import numpy as np

X = np.loadtxt("data/experiments.txt")
y = X[:, -2]
B = X[:, -1]
X = X[:, :-2]
B = np.array(B, dtype=np.uint64)

# fig, ax = plt.subplots()
# ax.scatter([1,2,3,4], y[:4])
# ax.scatter(range(5, len(y) + 1), y[4:])

dist = {
    0: X[:, 0],
    1: X[:, 1],
    2: X[:, 2],
    3: X[:, 3],
}


fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(range(X.shape[0]), X[:, 3] + X[:, 2] + X[:, 1] + X[:, 0])
ax.bar(range(X.shape[0]), X[:, 2] + X[:, 1] + X[:, 0])
ax.bar(range(X.shape[0]), X[:, 1] + X[:, 0])
ax.bar(range(X.shape[0]), X[:, 0])

fig.savefig("fig/progression.png", dpi=500, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5, 5))
for i in range(int(B.max() + 1)):
    xs = np.arange(B.shape[0])
    ax.scatter(xs[B == i], y[B == i], label=f"Observer {i}")
ax.legend(loc="upper left")
fig.savefig("fig/ys.png", dpi=500, bbox_inches="tight")
