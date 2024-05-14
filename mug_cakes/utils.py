from io import StringIO
from typing import List

import numpy as np
from numpy.typing import NDArray


def to_measurements(r: NDArray[np.float64], total: int) -> NDArray[np.uint64]:
    # get r in sixteenths
    r = r * 16
    t = (r * total).round()
    t = np.array(t, dtype=np.uint64)
    ones, t = np.divmod(t, 16)
    halves, t = np.divmod(t, 16 // 2)
    quarters, t = np.divmod(t, 16 // 4)
    eighths, t = np.divmod(t, 16 // 8)
    sixteenths, t = np.divmod(t, 16 // 16)
    return np.vstack(
        (
            ones.reshape(1, -1),
            halves.reshape(1, -1),
            quarters.reshape(1, -1),
            eighths.reshape(1, -1),
            sixteenths.reshape(1, -1),
        )
    )


def format_ingredients(m: NDArray[np.uint64], ingredients: List[str]) -> str:
    cell_size = 20
    tio = StringIO()
    tio.write(
        " " * cell_size,
    )
    for x in ingredients:
        tio.write(f"{x:<{cell_size}}")
    tio.write("\n")

    tio.write("ones:".ljust(cell_size))
    for x in m[0]:
        tio.write(f"{x:<{cell_size}}")
    tio.write("\n")

    tio.write("halves:".ljust(cell_size))
    for x in m[1]:
        tio.write(f"{x:<{cell_size}}")
    tio.write("\n")

    tio.write("quarters:".ljust(cell_size))
    for x in m[2]:
        tio.write(f"{x:<{cell_size}}")
    tio.write("\n")

    tio.write("eighths:".ljust(cell_size))
    for x in m[3]:
        tio.write(f"{x:<{cell_size}}")
    tio.write("\n")

    tio.write("16ths:".ljust(cell_size))
    for x in m[4]:
        tio.write(f"{x:<{cell_size}}")
    tio.write("\n")
    tio.seek(0)
    return tio.read()


# Hide X and Y axes label marks
def clear_tick(ax):
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])
