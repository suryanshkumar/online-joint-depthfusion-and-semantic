import plyfile
import numpy as np
from skimage.measure import marching_cubes_lewiner
import mcubes


def extract_mesh_marching_cubes(volume, color=None, level=-1e-07,
                                step_size=1., gradient_direction="ascent"):

    print(np.unique(volume))

    if level > volume.max() or level < volume.min():
        return

    verts, faces = mcubes.marching_cubes(volume, level)


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
