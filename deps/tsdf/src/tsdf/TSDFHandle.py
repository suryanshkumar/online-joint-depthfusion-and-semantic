import pandas as pd
import numpy as np

from pyntcloud import PyntCloud

from .TSDFVolume import TSDFVolume, MulticlassTSDFVolume


class TSDF:

    def __init__(self, bbox, resolution, resolution_factor,
                 free_space_vote=0.5, occupied_space_vote=1, volume_shape=None):
        self.bbox = bbox
        self.resolution = resolution

        self.tsdf_volume = TSDFVolume(2,
                                      bbox,
                                      resolution,
                                      resolution_factor,
                                      free_space_vote=free_space_vote,
                                      occupied_space_vote=occupied_space_vote,
                                      volume_shape=volume_shape)

        self.resolution = resolution

    def fuse(self, depth_proj_matrix, depth_map, weight_map):
        self.tsdf_volume.fuse(depth_proj_matrix, depth_map, weight_map)

    def sanity_fuse(self, depth_proj_matrix, depth_map, weight_map):
        self.tsdf_volume.sanity_fuse(depth_proj_matrix, depth_map, weight_map)

    def get_volume(self):
        return self.tsdf_volume.get_volume()

    def get_depth(self, extrinsics, intrinsics, shape):
        return self.tsdf_volume.get_depth(extrinsics, intrinsics, shape)

    def get_mask(self):
        return self.tsdf_volume.get_mask()

    def extract_mesh(self):
        from .utils import extract_mesh_marching_cubes
        return extract_mesh_marching_cubes(self.tsdf_volume.get_volume()[:, :, :, 0])

    def plot(self, mode='grid', points=None):

        if mode == 'grid':
            # TODO: make correct for TSDF
            grid = self.get_volume()[:, :, :, 0]
            from .utils import plot_grid
            if points is not None:
                offset = np.asarray([self.bbox[0, 0],
                                     self.bbox[1, 0],
                                     self.bbox[2, 0]])
                eye = ((points - offset)/self.resolution).astype(int)
                plot_grid(grid, eye=eye)
            else:
                plot_grid(grid)

        if mode == 'mesh':
            from .utils import plot_mesh
            mesh = self.extract_mesh()
            plot_mesh(mesh)


class MulticlassTSDF:

    def __init__(self, num_labels, bbox, resolution, resolution_factor,
                 free_space_vote=0.5, occupied_space_vote=1):

        self.tsdf_volume = MulticlassTSDFVolume(num_labels, bbox,
                                                resolution, resolution_factor,
                                                free_space_vote=free_space_vote,
                                                occupied_space_vote=occupied_space_vote)

        self.resolution = resolution

    def fuse(self, depth_proj_matrix, label_proj_matrix, depth_map, label_map):
        self.tsdf_volume.fuse(depth_proj_matrix,
                              label_proj_matrix,
                              depth_map,
                              label_map)

    def get_volume(self):
        return self.tsdf_volume.get_volume()

    def plot_labelled_volume(self, label_map):

        volume = self.tsdf_volume.get_volume()

        grid_labelled = np.argmin(volume, axis=-1)
        grid_occupied = (np.min(volume, axis=-1) < 0)

        occupied_idxs = np.column_stack(np.where(grid_occupied == True)) * self.resolution
        occupied_labels = grid_labelled[np.where(grid_occupied == True)]

        # voxel colors
        colors = []
        for label in occupied_labels:
            colors.append(label_map[label])
        colors = np.asarray(colors)

        # voxel alphas
        alphas = len(colors) * [255]

        dataframe = dict()

        dataframe['x'] = occupied_idxs[:, 0]
        dataframe['y'] = occupied_idxs[:, 1]
        dataframe['z'] = occupied_idxs[:, 2]
        dataframe['red'] = colors[:, 0]
        dataframe['green'] = colors[:, 1]
        dataframe['blue'] = colors[:, 2]
        dataframe['alpha'] = alphas

        pcl = pd.DataFrame(data=dataframe, columns=dataframe.keys())
        pcl = PyntCloud(pcl)
        pcl.plot()

    def to_meshlab(self, path, filename):
        raise NotImplementedError





