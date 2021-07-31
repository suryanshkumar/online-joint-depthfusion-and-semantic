import numpy as np
import dt


def compute_tsdf(grid):

    new_grid = np.copy(grid).astype(np.float64)

    new_grid[np.where(new_grid == 0.)] = 2.
    new_grid[np.where(new_grid == 1.)] = 0.
    new_grid[np.where(new_grid == 2.)] = 1.

    new_grid = 10.e6*new_grid

    tsdf, i = dt.compute(new_grid)

    tsdf = np.sqrt(tsdf)

    return tsdf







