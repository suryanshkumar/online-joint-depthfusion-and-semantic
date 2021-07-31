import os
import h5py

import numpy as np
import torch

from torch.utils.data import Dataset
from graphics import Voxelgrid
import trimesh
import skimage

from utils.metrics import evaluation, semantic_evaluation
from utils.mapping import get_mapping

from scipy.ndimage import median_filter


class Database(Dataset):

    def __init__(self, dataset, config):

        super(Database, self).__init__()

        self.device = config.device
        self.implementation = config.implementation

        self.transform = config.transform
        self.initial_value = config.init_value
        self.semantics = config.semantics
        self.semantic_grid = config.semantic_grid
        self.pad = config.pad

        if self.semantics:
            self.n_classes = config.n_classes

        self.scenes = []
        self.state = {} # signalizes, for a given scene, if the voxel grid contains integrated frames or not
        self.origin = {}
        self.resolution = {}

        self.scenes_gt = {}
        self.scenes_est = {}
        self.fusion_weights = {}
        self.ids_gt = {}
        self.ids_est = {}
        self.scores = {}

        for s in dataset.scenes:
            self.scenes.append(s)
            try:
                grid = dataset.get_grid(s, self.initial_value, self.semantic_grid) # tuple (sdf, semantics)
            except: # in case no gt is available (e.g. scannet)
                grid = dataset.create_grid(s, self.initial_value)
            self.state[s] = False # voxelgrid is empty
            self.scenes_gt[s] = grid[0]

            self.origin[s] = grid[0].origin
            self.resolution[s] = grid[0].resolution

            init_volume = self.initial_value * np.ones_like(grid[0].volume, dtype=np.float16)
            self.scenes_est[s] = Voxelgrid(self.scenes_gt[s].resolution)
            self.scenes_est[s].from_array(init_volume, self.scenes_gt[s].bbox)

            self.fusion_weights[s] = np.zeros_like(grid[0].volume, dtype=np.float16)

            if self.semantics:
                if self.semantic_grid:
                    self.ids_gt[s] = grid[1]

                init_volume = np.zeros_like(grid[0].volume, dtype=np.uint8)
                self.ids_est[s] = Voxelgrid(grid[0].resolution)
                self.ids_est[s].from_array(init_volume, grid[0].bbox)

                init_volume = np.zeros_like(grid[0].volume, dtype=np.float16)
                self.scores[s] = Voxelgrid(grid[0].resolution)
                self.scores[s].from_array(init_volume, grid[0].bbox)

        self.to_torch()
        # self.reset()

    def __getitem__(self, item):

        sample = dict()

        sample['origin'] = self.origin[item]
        sample['resolution'] = self.resolution[item]
        sample['gt'] = self.scenes_gt[item].volume
        sample['current'] = self.scenes_est[item].volume
        sample['weights'] = self.fusion_weights[item]
        if self.semantics:
            sample['ids_est'] = self.ids_est[item].volume
            sample['scores'] = self.scores[item].volume
            if self.semantic_grid:
                sample['ids_gt'] = self.ids_gt[item].volume
        else:
            sample['histograms'] = None
            sample['ids_est'] = None
            sample['ids_gt'] = None
            sample['scores'] = None

        # sample = self.transform(sample)           

        return sample

    def __len__(self):
        return len(self.scenes_gt)

    def filter(self, value=2.):
        for s in self.scenes:
            weights = self.fusion_weights[s]
            self.scenes_est[s].volume[weights < value] = self.initial_value
            self.fusion_weights[s][weights < value] = 0

    def filter_semantics(self, value=5):
        for s in self.scenes:
            self.ids_est[s].volume = median_filter(self.ids_est[s].volume, size=value)

    def get_mesh(self, scene_id, semantics=False):
        voxel_size = self.resolution[scene_id]
        vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(self.scenes_est[scene_id].volume, 
                                                                            level=0, 
                                                                            spacing=(voxel_size, voxel_size, voxel_size))
        if semantics:
            points = translate_points(vertices, self.origin[scene_id])

            x = (np.round((points[:, 0] - self.origin[scene_id][0]) / voxel_size)).astype(np.uint16)
            y = (np.round((points[:, 1] - self.origin[scene_id][1]) / voxel_size)).astype(np.uint16)
            z = (np.round((points[:, 2] - self.origin[scene_id][2]) / voxel_size)).astype(np.uint16)
            ids = [self.ids_est[scene_id].volume[x[i], y[i], z[i]] for i in range(len(x))]
            del x, y, z

            map_rgb = get_mapping()
            map_rgb[0] = [128, 128, 128] # gray instead of black
            rgb = map_rgb[ids]
            rgb = rgb / 255.0
        else:
            rgb = None

        return vertices, faces, normals, rgb

    def save_to_workspace(self, workspace, mode, save_mode='ply'):
        for s in self.scenes:
            if self.state[s]:

                tsdf_volume = self.scenes_est[s].volume
                weight_volume = self.fusion_weights[s]

                if save_mode == 'tsdf':
                    tsdf_file = s.replace('/', '.') + '.tsdf_' + mode + '.hf5'
                    weight_file = s.replace('/', '.') + '.weights_' + mode + '.hf5'

                    workspace.save_tsdf_data(tsdf_file, tsdf_volume)
                    workspace.save_weights_data(weight_file, weight_volume)

                    if self.semantics:
                        semantic_file = s.replace('/', '.') + '.semantic_' + mode + '.hf5'
                        semantic_volume = self.ids_est[s].volume
                        workspace.save_semantic_data(semantic_file, semantic_volume)
                        
                elif save_mode == 'ply':
                    ply_file = s.replace('/', '.') + '_' + mode + '.ply'
                    workspace.save_ply_data(ply_file, tsdf_volume)

                elif save_mode == 'test':
                    tsdf_file = s.replace('/', '.') + '.tsdf_' + mode + '.hf5'
                    weight_file = s.replace('/', '.') + '.weights_' + mode + '.hf5'
                    ply_file = s.replace('/', '.') + '_' + mode + '.ply'

                    workspace.save_tsdf_data(tsdf_file, tsdf_volume)
                    workspace.save_weights_data(weight_file, weight_volume)

                    if self.semantics:
                        semantic_file = s.replace('/', '.') + '.semantic_' + mode + '.hf5'
                        semantic_volume = self.ids_est[s].volume
                        workspace.save_semantic_data(semantic_file, semantic_volume)

                    workspace.save_ply_data(ply_file, tsdf_volume)


    def save(self, path, save_mode='ply', scene_id=None):
        if scene_id is None:
            raise NotImplementedError
        else:
            if save_mode =='tsdf':
                filename = '{}.tsdf.hf5'.format(scene_id.replace('/', '.'))
                weightname = '{}.weights.hf5'.format(scene_id.replace('/', '.'))
                semname = '{}.semantics.hf5'.format(scene_id.replace('/', '.'))

                with h5py.File(os.path.join(path, filename), 'w') as hf:
                    hf.create_dataset("TSDF",
                                      shape=self.scenes_est[scene_id].volume.shape,
                                      data=self.scenes_est[scene_id].volume)
                with h5py.File(os.path.join(path, weightname), 'w') as hf:
                    hf.create_dataset("weights",
                                      shape=self.fusion_weights[scene_id].shape,
                                      data=self.fusion_weights[scene_id])
                if self.semantics:
                    with h5py.File(os.path.join(path, semname), 'w') as hf:
                        hf.create_dataset("semantics",
                                          shape=self.ids_est[scene_id].volume.shape,
                                          data=self.ids_est[scene_id].volume)

            elif save_mode == 'ply':
                ply_file = scene_id.replace('/', '.') + '.ply'
                filename = os.path.join(path, ply_file)
                voxel_size = self.resolution[scene_id]
                vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(self.scenes_est[scene_id].volume, 
                                                                                    level=0, 
                                                                                    spacing=(voxel_size, voxel_size, voxel_size))
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
                mesh.export(filename)

            elif save_mode == 'test':
                filename = '{}.tsdf.hf5'.format(scene_id.replace('/', '.'))
                weightname = '{}.weights.hf5'.format(scene_id.replace('/', '.'))
                semname = '{}.semantics.hf5'.format(scene_id.replace('/', '.'))

                with h5py.File(os.path.join(path, filename), 'w') as hf:
                    hf.create_dataset("TSDF",
                                      shape=self.scenes_est[scene_id].volume.shape,
                                      data=self.scenes_est[scene_id].volume)
                with h5py.File(os.path.join(path, weightname), 'w') as hf:
                    hf.create_dataset("weights",
                                      shape=self.fusion_weights[scene_id].shape,
                                      data=self.fusion_weights[scene_id])

                ply_file = scene_id.replace('/', '.') + '.ply'
                filename = os.path.join(path, ply_file)
                voxel_size = self.resolution[scene_id]
                
                vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(self.scenes_est[scene_id].volume, 
                                                                                     level=0, 
                                                                                     spacing=(voxel_size, voxel_size, voxel_size))

                # IMPORTANT: Process=False avoid Trimesh to reorder the vertices, so that the semantics can be correctly mapped
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
                mesh.export(filename)

                if self.semantics:
                    with h5py.File(os.path.join(path, semname), 'w') as hf:
                        hf.create_dataset("semantics",
                                          shape=self.ids_est[scene_id].volume.shape,
                                          data=self.ids_est[scene_id].volume)

                    ply_file = scene_id.replace('/', '.') + '_semantic.ply'
                    filename = os.path.join(path, ply_file)

                    points = translate_points(vertices, self.origin[scene_id])

                    x = (np.round((points[:, 0] - self.origin[scene_id][0]) / voxel_size)).astype(np.uint16)
                    y = (np.round((points[:, 1] - self.origin[scene_id][1]) / voxel_size)).astype(np.uint16)
                    z = (np.round((points[:, 2] - self.origin[scene_id][2]) / voxel_size)).astype(np.uint16)
                    ids = [self.ids_est[scene_id].volume[x[i], y[i], z[i]] for i in range(len(x))]
                    del x, y, z

                    map_rgb = get_mapping()
                    rgb = map_rgb[ids]
                    # encode ids in the alpha value
                    rgba = np.concatenate((rgb, np.array(ids)[:, np.newaxis]), axis=1)
                    mesh.visual.vertex_colors = rgba
                    mesh.export(filename)


    # @profile
    def evaluate(self, mode='train', workspace=None):

        eval_results = {}
        eval_results_scene_save = {}
        if workspace is not None:
            workspace.log('-------------------------------------------------------', 
                mode)
        for scene_id in self.scenes:
            if self.state[scene_id]:
                if workspace is None:
                    print('Evaluating ', scene_id, '...')
                else:
                    workspace.log('Evaluating {} ...'.format(scene_id),
                                  mode)

                est = self.scenes_est[scene_id].volume
                gt = self.scenes_gt[scene_id].volume

                mask = (self.fusion_weights[scene_id] > 0)

                eval_results_scene = evaluation(est, gt, mask)

                del est, gt, mask

                eval_results_scene_save[scene_id] = eval_results_scene

                for key in eval_results_scene.keys():
                    if workspace is None:
                        print(key, eval_results_scene[key])
                    else:
                        workspace.log('{} {}'.format(key, eval_results_scene[key]), mode)

                    if not eval_results.get(key):
                        eval_results[key] = eval_results_scene[key]
                    else:
                        eval_results[key] += eval_results_scene[key]

        # normalizing metrics
        for key in eval_results.keys():
            eval_results[key] /= len(self.scenes_est.keys())

        if mode == 'test':
            return eval_results, eval_results_scene_save
        else:
            return eval_results

    def evaluate_semantics(self, mode='train', workspace=None):

        eval_results = {}
        class_iou_scene_save = {}
        if workspace is not None:
            workspace.log('-------------------------------------------------------', mode)
        for scene_id in self.scenes:
            if self.state[scene_id]:
                if workspace is None:
                    print('Evaluating ', scene_id, '...')
                else:
                    workspace.log('Evaluating {} ...'.format(scene_id), mode)

                est = self.ids_est[scene_id].volume
                gt = self.ids_gt[scene_id].volume

                mask = (self.fusion_weights[scene_id] > 0) 

                eval_results_scene, class_iou_scene = semantic_evaluation(est, gt, mask, self.n_classes)
                del est, gt, mask

                class_iou_scene_save[scene_id] = class_iou_scene

                for key in eval_results_scene.keys():
                    if workspace is None:
                        print(key, eval_results_scene[key])
                    else:
                        workspace.log('{} {}'.format(key, eval_results_scene[key]), mode)

                    if not eval_results.get(key):
                        eval_results[key] = eval_results_scene[key]
                    else:
                        eval_results[key] += eval_results_scene[key]

        # normalizing metrics
        for key in eval_results.keys():
            eval_results[key] /= len(self.scenes_est.keys())

        return eval_results, class_iou_scene_save

    def reset(self, scene_id=None):
        if scene_id:
            self.state[scene_id] = False
            self.scenes_est[scene_id].volume = self.initial_value * np.ones(self.scenes_est[scene_id].volume.shape, dtype=np.float16)
            self.fusion_weights[scene_id] = np.zeros(self.scenes_est[scene_id].volume.shape, dtype=np.float16)
            if self.semantics:
                self.ids_est[scene_id].volume = np.zeros(self.ids_est[scene_id].volume.shape, dtype=np.uint8)
                self.scores[scene_id].volume = np.zeros(self.scores[scene_id].volume.shape, dtype=np.float16)
            self.to_torch(gt=False, scenes=scene_id)

        else:
            for scene_id in self.scenes:
                self.state[scene_id] = False
                self.scenes_est[scene_id].volume = self.initial_value * np.ones(self.scenes_est[scene_id].volume.shape, dtype=np.float16)
                self.fusion_weights[scene_id] = np.zeros(self.scenes_est[scene_id].volume.shape, dtype=np.float16)
                if self.semantics:
                    self.ids_est[scene_id].volume = np.zeros(self.ids_est[scene_id].volume.shape, dtype=np.uint8)
                    self.scores[scene_id].volume = np.zeros(self.scores[scene_id].volume.shape, dtype=np.float16)

            self.to_torch(gt=False)

    def remove(self, scene_id):
        self.state[scene_id] = False
        self.scenes_est[scene_id] = None
        self.scenes_gt[scene_id] = None
        self.fusion_weights[scene_id] = None
        if self.semantics:
            self.ids_est[scene_id] = None
            self.scores[scene_id] = None
            if self.semantic_grid:
                self.ids_gt[scene_id] = None

    def to_numpy(self):
        for scene_id in self.scenes:
            self.origin[scene_id] = self.origin[scene_id].detach().cpu().numpy()
            self.scenes_est[scene_id].volume = self.scenes_est[scene_id].volume.detach().cpu().numpy()
            self.scenes_gt[scene_id].volume = self.scenes_gt[scene_id].volume.detach().cpu().numpy()
            self.fusion_weights[scene_id] = self.fusion_weights[scene_id].detach().cpu().numpy()
            if self.semantics:
                self.ids_est[scene_id].volume = self.ids_est[scene_id].volume.detach().cpu().numpy()
                self.scores[scene_id].volume = self.scores[scene_id].volume.detach().cpu().numpy()
                if self.semantic_grid:
                    self.ids_gt[scene_id].volume = self.ids_gt[scene_id].volume.detach().cpu().numpy()

    def to_torch(self, gt=True, scenes=None):
        if scenes is None:
            scenes = self.scenes
        else:
            scenes = [scenes]

        for scene_id in scenes:
            self.scenes_est[scene_id].volume = torch.from_numpy(self.scenes_est[scene_id].volume)
            if gt:
                self.origin[scene_id] = torch.from_numpy(self.origin[scene_id])
                self.scenes_gt[scene_id].volume = torch.from_numpy(self.scenes_gt[scene_id].volume)
            self.fusion_weights[scene_id] = torch.from_numpy(self.fusion_weights[scene_id])

            if self.implementation == 'efficient':
                self.origin[scene_id] = self.origin[scene_id].to(self.device)
                self.scenes_est[scene_id].volume = self.scenes_est[scene_id].volume.to(self.device)
                self.fusion_weights[scene_id] = self.fusion_weights[scene_id].to(self.device)

            if self.semantics:
                self.ids_est[scene_id].volume = torch.from_numpy(self.ids_est[scene_id].volume)
                self.scores[scene_id].volume = torch.from_numpy(self.scores[scene_id].volume)
                if gt:
                    if self.semantic_grid:
                        self.ids_gt[scene_id].volume = torch.from_numpy(self.ids_gt[scene_id].volume)
                if self.implementation == 'efficient':
                    self.ids_est[scene_id].volume = self.ids_est[scene_id].volume.to(self.device)
                    self.scores[scene_id].volume = self.scores[scene_id].volume.to(self.device)


def translate_points(points, origin):
    # points: points (=vertices) of the estimated grid
    # origin: coordinates of the ground truth box
    R = np.diag(np.ones(4))
    R[0:3, -1] = origin - [np.amin(points[:, 0]), np.amin(points[:, 1]), np.amin(points[:, 2])]
    points = np.concatenate([points, np.ones((len(points), 1))], axis=1)    # homogeneous coordinates
    points = np.array(np.dot(points, np.transpose(R)))[:, 0:3]              # rotation (only translation in this case)
    return points

