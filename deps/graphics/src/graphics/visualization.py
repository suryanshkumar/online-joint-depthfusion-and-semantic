from mayavi import mlab
import numpy as np
import os

def plot_voxelgrid(grid):

    xx, yy, zz = np.where(grid == 1)
    mlab.points3d(xx, yy, zz,
                  mode='cube',
                  color=(0, 1, 0),
                  scale_factor=1)

    mlab.show()
    mlab.close(all=True)


def plot_tsdf(tsdf):

    xx, yy, zz = np.meshgrid(np.arange(tsdf.shape[0]),
                             np.arange(tsdf.shape[1]),
                             np.arange(tsdf.shape[2]))


    xx = xx.ravel()
    yy = yy.ravel()
    zz = zz.ravel()

    s = np.ravel(tsdf, order='C')

    mlab.points3d(xx, yy, zz, s,
                  mode='cube',
                  scale_factor=1)

    mlab.show()

def plot_mesh(mesh, save=False, path=None):

    vertices = mesh['vertex']

    # extract vertex coordinates
    (x, y, z) = (vertices[t] for t in ('x', 'y', 'z'))

    mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')

    if 'face' in mesh:
        tri_idx = mesh['face']['vertex_indices']
        triangles = np.asarray(tri_idx)

        mlab.triangular_mesh(x, y, z, triangles)

    if save:
        assert path is not None
        mlab.savefig(os.path.join(path, 'VoxelGT.obj'))

    mlab.show()

