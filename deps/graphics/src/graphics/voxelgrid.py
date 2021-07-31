import numpy as np

from .sampler import mesh_to_pointcloud
from .transform import compute_tsdf
from .utils import extract_mesh_marching_cubes

from tsdf import depth_rendering

from gridData import OpenDX
from tqdm import tqdm


class FeatureGrid(object):

    def __init__(self, resolution, bbox, n_features=10, origin=None):

        self._resolution = resolution
        self._bbox = bbox
        self._n_features = n_features
        self._origin = origin

        xshape = np.diff(bbox[0, :])
        yshape = np.diff(bbox[1, :])
        zshape = np.diff(bbox[2, :])

        self._shape = (xshape, yshape, zshape, n_features)

        self._data = np.zeros(self._shape)

    @property
    def resolution(self):
        return self._resolution

    @property
    def bbox(self):
        return self._bbox

    @property
    def origin(self):
        return self._origin

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape



class Voxelgrid(object):

    def __init__(self, resolution, bbox=None, origin=None, initial_value=0.):

        self.resolution = resolution

        self._volume = None
        self._bbox = None
        self._origin = None

        if (bbox is not None) and (origin is not None):

            self._bbox = bbox

            volume_shape = np.diff(self._bbox, axis=1).ravel() / self.resolution
            volume_shape = np.ceil(volume_shape).astype(np.int32).tolist()  # round up

            self._volume = initial_value*np.ones(volume_shape).astype('float16')
            self._origin = origin

    def from_pointcloud(self, pointcloud):

        mesh = dict()
        mesh['vertex'] = pointcloud.points
        mesh['face'] = dict()
        mesh['face']['vertex_indices'] = pointcloud.mesh

        minx = pointcloud.points.x.min()
        miny = pointcloud.points.y.min()
        minz = pointcloud.points.z.min()
        maxx = pointcloud.points.x.max()
        maxy = pointcloud.points.y.max()
        maxz = pointcloud.points.z.max()

        diffx = maxx - minx
        diffy = maxy - miny
        diffz = maxz - minz

        minx -= self.resolution * diffx
        maxx += self.resolution * diffx
        miny -= self.resolution * diffy
        maxy += self.resolution * diffy
        minz -= self.resolution * diffz
        maxz += self.resolution * diffz

        self._bbox = np.array(((minx, maxx), (miny, maxy), (minz, maxz)),
                              dtype=np.float32)

        volume_shape = np.diff(self._bbox, axis=1).ravel()/self.resolution
        volume_shape = np.ceil(volume_shape).astype(np.int32).tolist() # round up

        self._volume = np.zeros(volume_shape)
        self._origin = np.asarray([minx, miny, minz])

        for row, point in tqdm(pointcloud.points.iterrows(), total=len(pointcloud.points)):
            x = int((point['x'] - minx) / self.resolution)
            y = int((point['y'] - miny) / self.resolution)
            z = int((point['z'] - minz) / self.resolution)

            self._volume[x, y, z] = 1.

    def from_obj(self, obj):

        # TODO: not the right approach, need to implement correct way of voxelization

        vertices = np.asarray(obj.vertices)
        faces = np.asarray(obj.meshes[None].faces)
        n_points = 100000

        pcl = mesh_to_pointcloud(vertices, faces, n_points)

        minx = pcl[:, 0].min(axis=0)
        maxx = pcl[:, 0].max(axis=0)
        miny = pcl[:, 1].min(axis=0)
        maxy = pcl[:, 1].max(axis=0)
        minz = pcl[:, 2].min(axis=0)
        maxz = pcl[:, 2].max(axis=0)

        diffx = maxx - minx
        diffy = maxy - miny
        diffz = maxz - minz

        minx -= self.resolution * diffx
        maxx += self.resolution * diffx
        miny -= self.resolution * diffy
        maxy += self.resolution * diffy
        minz -= self.resolution * diffz
        maxz += self.resolution * diffz

        nx = int(np.ceil((maxx - minx) / self.resolution)[0])
        ny = int(np.ceil((maxy - miny) / self.resolution)[0])
        nz = int(np.ceil((maxz - minz) / self.resolution)[0])

        self._bbox = np.array(((minx, maxx), (miny, maxy), (minz, maxz)),
                              dtype=np.float32)
        self._volume = np.zeros((nx, ny, nz))

        for point in pcl:

            x = int((point[0] - minx) / self.resolution)
            y = int((point[1] - miny) / self.resolution)
            z = int((point[2] - minz) / self.resolution)

            self._volume[x, y, z] = 1.

    def from_array(self, array, bbox):

        self._volume = array
        self._bbox = bbox
        self._origin = bbox[:, 0]

    def from_dx(self, path):

        # parse dx file
        dx = OpenDX.field()
        dx.read(path)

        origin = dx.components['positions'].origin
        delta = dx.components['positions'].delta[0, 0]
        shape = dx.components['positions'].shape
        data = dx.components['data'].array

        data = data.reshape(shape)

        self._volume = data

        minx = origin[0]
        miny = origin[1]
        minz = origin[2]

        maxx = minx + delta*shape[0]
        maxy = miny + delta*shape[1]
        maxz = minz + delta*shape[2]

        diffx = maxx - minx
        diffy = maxy - miny
        diffz = maxz - minz

        self.resolution = delta
        self._bbox = np.array(((minx, maxx), (miny, maxy), (minz, maxz)),
                              dtype=np.float32)

        self._origin = origin

    @property
    def bbox(self):
        assert self._bbox is not None
        return self._bbox

    @property
    def volume(self):
        assert self._volume is not None
        return self._volume

    @volume.setter
    def volume(self, volume):
        self._volume = volume

    @property
    def origin(self):
        assert self._origin is not None
        return self._origin

    @property
    def shape(self):
        assert self._volume is not None
        return self._volume.shape

    def transform(self, mode='normal'):
        if mode == 'normal':
            dist1 = compute_tsdf(self._volume.astype(np.float64))
            dist1[dist1 > 0] -= 0.5
            dist2 = compute_tsdf(np.ones(self._volume.shape) - self._volume)
            dist2[dist2 > 0] -= 0.5
            # print(np.where(dist == 79.64923100695951))
            self._volume = np.copy(dist1-dist2)
        if mode == 'flipped':
            dist1 = compute_tsdf(self._volume.astype(np.float64))
            dist1[dist1 > 0] -= 0.5
            dist2 = compute_tsdf(np.ones(self._volume.shape) - self._volume)
            dist1[dist2 > 0] -= 0.5
            # print(np.where(dist == 79.64923100695951))
            tsdf = dist1 - dist2
            tsdf = np.sign(tsdf)*(np.max(tsdf) - tsdf)
            self._volume = np.copy(tsdf)

    def get_tsdf(self):

        assert self._volume is not None

        tsdf = compute_tsdf(self._volume)

        return tsdf

    def get_frame(self, intrinsics, extrinsics, shape, frame):

        # TODO: not clean and efficient
        extrinsics = extrinsics.astype(np.float32)
        intrinsics = intrinsics.astype(np.float32)

        offset = np.asarray([self.bbox[0, 0], self.bbox[1, 0], self.bbox[2, 0]], dtype=np.float32)
        volume = np.expand_dims(self._volume, axis=-1).astype(np.float32)


        resolution = self.resolution

        depth = depth_rendering(extrinsics, intrinsics, shape,
                                volume, resolution, offset, frame)

        return depth

    def compare(self, reference):

        from mayavi import mlab

        xx1, yy1, zz1 = np.where((reference != 0.) & (self._volume != 0))
        mlab.points3d(xx1, yy1, zz1,
                      mode='cube',
                      color=(1, 0, 0),
                      scale_factor=1)


        xx2, yy2, zz2 = np.where(((self._volume != 0.) & (reference == 0.)))
        mlab.points3d(xx2, yy2, zz2,
                      mode='cube',
                      color=(0, 1, 0),
                      scale_factor=1)


        xx3, yy3, zz3 = np.where((self._volume == 0.) & (reference != 0.))
        mlab.points3d(xx3, yy3, zz3,
                      mode='cube',
                      color=(0, 0, 1),
                      scale_factor=1)

        mlab.show()

    def plot(self, mode='grid', reference=None, point=None):
        from .visualization import plot_mesh
        from mayavi import mlab

        if mode == 'grid':

            xx, yy, zz = np.where(self._volume != 0)
            mlab.points3d(xx, yy, zz,
                          mode='cube',
                          color=(0, 1, 0),
                          scale_factor=1)

            if point is not None:

                if len(point.shape) == 1:

                    xx, yy, zz = point[0], point[1], point[2]
                    mlab.points3d(xx, yy, zz,
                                  mode='cube',
                                  color=(0, 0, 1))
                else:

                    for p in point:
                        xx, yy, zz = p[0], p[1], p[2]
                        mlab.points3d(xx, yy, zz,
                                      mode='cube',
                                      color=(0, 0, 1))


        elif mode == 'mesh':

            mesh = extract_mesh_marching_cubes(self._volume)
            plot_mesh(mesh)

        mlab.show()

    def save(self, filename):

        np.savez

    def __getattr__(self, x, y, z):
        return self._volume[x, y, z]


