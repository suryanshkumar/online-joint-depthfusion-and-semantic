import plyfile
import numpy as np
from skimage.measure import marching_cubes_lewiner

from mayavi import mlab

def extract_mesh_marching_cubes(volume, color=None, level=0.5,
                                step_size=1.0, gradient_direction="ascent"):

    if level > volume.max() or level < volume.min():
        return

    verts, faces, normals, values = marching_cubes_lewiner(volume[:, :, :, 0],
                                                           level=level,
                                                           step_size=step_size,
                                                           gradient_direction=gradient_direction)

    ply_verts = np.empty(len(verts),
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    ply_verts["x"] = verts[:, 0]
    ply_verts["y"] = verts[:, 1]
    ply_verts["z"] = verts[:, 2]

    ply_verts = plyfile.PlyElement.describe(ply_verts, "vertex")

    if color is None:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,))])
    else:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,)),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        ply_faces["red"] = color[0]
        ply_faces["green"] = color[1]
        ply_faces["blue"] = color[2]

    ply_faces["vertex_indices"] = faces
    ply_faces = plyfile.PlyElement.describe(ply_faces, "face")

    return plyfile.PlyData([ply_verts, ply_faces])


def plot_mesh(mesh):

    vertices = mesh['vertex']

    # extract vertex coordinates
    (x, y, z) = (vertices[t] for t in ('x', 'y', 'z'))

    mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')

    if 'face' in mesh:
        tri_idx = mesh['face']['vertex_indices']
        triangles = np.asarray(tri_idx)
        mlab.triangular_mesh(x, y, z, triangles)

    mlab.show()


def plot_grid(grid, eye=None):

    xx, yy , zz = np.where(grid == 1)
    mlab.points3d(xx, yy, zz,
                  mode='cube',
                  color=(0, 1, 0),
                  scale_factor=1)

    if eye is not None:
        xx, yy, zz = eye[:, 0].ravel(), eye[:, 1].ravel(), eye[:, 2].ravel()
        mlab.points3d(xx, yy, zz,
                      mode='cube',
                      color=(1, 0, 0),
                      scale_factor=1)

    mlab.show()




