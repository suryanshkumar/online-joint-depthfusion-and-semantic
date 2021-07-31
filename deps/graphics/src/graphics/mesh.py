import numpy as np


class Mesh(object):

    def __init__(self):

        self.mesh = None
        self.vertices = None
        self.faces = None

    def from_obj(self, obj):

        self.mesh = obj
        self.vertices = np.asarray(obj.vertices)
        self.faces = np.asarray(obj.meshes[None].faces)

    def plot(self):

        from mayavi import mlab

        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        z = self.vertices[:, 2]

        triangles = self.faces

        mlab.triangular_mesh(x, y, z, triangles)
        mlab.show()
        mlab.close(all=True)


