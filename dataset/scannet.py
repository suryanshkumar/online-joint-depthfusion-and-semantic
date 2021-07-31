import os
import sys
import glob
import h5py
import random
import numpy as np
import cv2
import trimesh
import math
from torch.utils.data import Dataset
from graphics import Voxelgrid
from utils.mapping import *


class ScanNet(Dataset):

    def __init__(self, config_data):

        self.root_dir = config_data.root_dir

        self.resolution = (config_data.resy, config_data.resx) # numpy format [rows, columns]
        self.pad = config_data.pad

        self.augmentations = config_data.augmentations
        self.normalize = config_data.normalize
        self.transform = config_data.transform
        self.frame_ratio = config_data.frame_ratio

        self.scene_list = config_data.scene_list
        self.input = config_data.input
        self.target = config_data.target
        self.semantics = config_data.semantics
        self.mode = config_data.mode
        self.intensity_gradient = config_data.intensity_grad
        self.truncation_strategy = config_data.truncation_strategy
        self.fusion_strategy = config_data.fusion_strategy

        self._scenes = []

        if config_data.data_load_strategy == 'hybrid':
            # The loading strategy 'hybrid' will always only load from 
            # at most 1 trajectory of a scene at a time
            self._get_scene_load_order(config_data.load_scenes_at_once)
        else:
            # The loading strategy 'max_depth_diversity' loads all 
            # trajectories from all scenes at the same time.
            self.scenedir = None

        self._load_color()
        self._load_cameras()
        self._load_intriniscs()
        if self.input != 'image':
            self._load_depth()
        if self.target == 'depth_gt':
            self._load_depth_gt()

        if self.semantics == 'nyu40':
            self.rgb_map = scannet_color_palette()
            self.label_map = ids_to_nyu40()
            self.names_map = scannet_nyu40_names()
            self._load_semantic_gt()
        elif self.semantics == 'nyu20':
            self.rgb_map = [scannet_color_palette()[i] for i in scannet_main_ids()] 
            self.label_map = ids_to_nyu20()
            self.names_map = scannet_nyu20_names()
            self.main_ids = np.array(scannet_main_ids())
            self._load_semantic_gt()

    def _get_scene_load_order(self, nbr_load_scenes):
        # create list of training scenes
        # format: scans/scene0000_00 = [scans_dir]/scene[id]_[traj]

        scenes_list = list()
        with open(self.scene_list, 'r') as file:
            for line in file:
                if line.split(' ')[0].split('/')[1] not in scenes_list:
                    scenes_list.append(line.split(' ')[0].split('/')[1])

        self._scenes = scenes_list

        scenes_dict = {k: [] for k in {scene.split('_')[0] for scene in scenes_list}}
        for scene in self._scenes:
            scenes_dict[scene.split('_')[0]].append(scene.split('_')[1])

        # make sure nbr_load_scenes <= len(trajectory_list)
        if nbr_load_scenes > len(scenes_dict):
            raise ValueError('nbr_load_scenes variable must be lower than the number of scenes')

        listdir = {k: [] for k in range(nbr_load_scenes)}
        while scenes_dict:
            scene_indices = random.sample(range(0, len(scenes_dict)), min(len(scenes_dict), nbr_load_scenes))

            for key, scene_idx in enumerate(scene_indices):
                scene = list(scenes_dict.keys())[scene_idx]
                for traj in scenes_dict[scene]:
                    listdir[key].append(scene + '_' + traj)

            scenes_dict = {k: v for idx, (k, v) in enumerate(scenes_dict.items()) if idx not in scene_indices}

        self.scenedir = {k: random.sample(v, len(v))  for k, v in listdir.items()}

    def _hybrid_load(self, modality):
        # create the full lists for each key in scenedir
        img_paths = []
        tmp_dict = dict()

        for key in self.scenedir:
            tmp_list = list()
            for scene in self.scenedir[key]:
                files = glob.glob(os.path.join(self.root_dir, scene, modality, '*'))
                files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                for file in files:
                    tmp_list.append(file)
            # replace the short list with the long list including all file paths
            tmp_dict[key] = tmp_list
        
        # fuse the lists into one
        # find key with longest list
        max_key = max(tmp_dict, key=lambda x: len(tmp_dict[x]))
        idx = 0
        while idx < len(tmp_dict[max_key]):
            for key in tmp_dict:
                if idx < len(tmp_dict[key]):
                    img_paths.append(tmp_dict[key][idx])
            idx += 1

        img_paths = img_paths[::self.frame_ratio]
        
        return img_paths

    def _load_from_list(self, position, debug=False):
        # reading files from list
        img_paths = []

        with open(self.scene_list, 'r') as fp:
            for line in fp:
                line = line.rstrip('\n').split(' ')
                if line[0].split('/')[1] not in self._scenes:
                    self._scenes.append(line[0].split('/')[1])
                files = glob.glob(os.path.join(self.root_dir, line[position], '*'))
                files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                for file in files:
                    img_paths.append(file)
                    if debug: print(file)
        
        img_paths = img_paths[::self.frame_ratio]

        # perhaps it will be important to order the frames for testing and training the fusion network.
        img_paths.sort(key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))

        return img_paths

    def _load_depth(self): 
        # loads the paths of the noisy depth images to a list
        self.depth_images = self._hybrid_load('depth') if self.scenedir else self._load_from_list(0)

    def _load_depth_gt(self): 
        # loads the paths of the ground truth depth images to a list
        self.depth_images_gt = self._hybrid_load('depth') if self.scenedir else self._load_from_list(0)

    def _load_color(self):
        # loads the paths of the RGB images to a list
        self.color_images = self._hybrid_load('color') if self.scenedir else self._load_from_list(1)

    def _load_semantic_gt(self):
        # loads the paths of the ground truth semantic images to a list
        self.semantic_images_gt = self._hybrid_load('label-filt') if self.scenedir else self._load_from_list(2)

    def _load_cameras(self):
        # loads the paths of the camera extrinsic matrices to a list
        self.cameras = self._hybrid_load('pose') if self.scenedir else self._load_from_list(3)

    def _load_intriniscs(self):
        # loads the paths of the camera intrinsic matrices to a dict
        self.intrinsics = {}
        with open(self.scene_list, 'r') as file:
            for line in file:
                line = line.rstrip('\n').split(' ')
                scene = line[0].split('/')[1]
                intrinsics = np.loadtxt(os.path.join(self.root_dir, line[-1], 'intrinsic_depth.txt'))
                kx = self.resolution[1] / 640
                ky = self.resolution[0] / 480
                k = np.array([[kx, 0, kx], [0, ky, ky], [0, 0, 1]]).astype(np.float32)
                intrinsics = np.matmul(k, intrinsics[0:3, 0:3])
                self.intrinsics.update({scene: intrinsics})
                    
    @property
    def scenes(self):
        return self._scenes

    def __len__(self):
        return len(self.color_images)

    def __getitem__(self, item):
        
        sample = dict()
        sample['item_id'] = item 

        # load rgb image
        file = self.color_images[item]

        pathsplit = file.split('/')
        scene = pathsplit[-3]
        frame = os.path.splitext(pathsplit[-1])[0]
        frame_id = '{}/{}'.format(scene, frame)
        sample['frame_id'] = frame_id

        image = cv2.imread(file)
        image = cv2.resize(image, self.resolution[::-1], interpolation=cv2.INTER_NEAREST) # expects shape as columnsXrows

        if self.semantics:
            # load ground truth semantics
            file = self.semantic_images_gt[item]
            semantic = cv2.imread(file, -1)
            semantic = cv2.resize(semantic, self.resolution[::-1], interpolation=cv2.INTER_NEAREST)

            if self.augmentations is not None:
                image, semantic = self.augmentations(image, semantic)

            semantic = np.array([self.label_map[s] for s in semantic.flatten()])
            semantic = semantic.reshape(self.resolution)
            sample['semantic_gt'] = semantic.astype(np.uint8)

        if self.normalize:
            mean = [99.09, 113.94, 126.81]
            std = [69.64, 71.31, 73.16]
            image = (image - mean) / std

        sample['image'] = image.astype(np.float32)

        if self.input == 'depth_gt':
            # load input depth map
            file = self.depth_images[item]
            depth = cv2.imread(file, -1)
            depth = cv2.resize(depth, self.resolution[::-1], interpolation=cv2.INTER_NEAREST) / 1000.
            if np.any(np.isnan(depth)):
                print("NaN in depth input")
            sample[self.input] = depth.astype(np.float32)

            # define mask
            mask = (depth > 0.01)
            sample['mask'] = mask

        if self.target == 'depth_gt':
            # load ground truth depth map
            file = self.depth_images_gt[item]
            depth = cv2.imread(file, -1)
            depth = cv2.resize(depth, self.resolution[::-1], interpolation=cv2.INTER_NEAREST) / 1000.
            sample[self.target] = depth.astype(np.float32)

        # load extrinsics
        # the fusion code expects that the camera coordinate system is such that z is in the

        # camera viewing direction, y is down and x is to the right.
        file = self.cameras[item]

        sample['extrinsics'] = np.loadtxt(file).astype(np.float32)
        sample['intrinsics'] = self.intrinsics[scene]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_grid(self, scene, truncation, semantic_grid):
        file = os.path.join(self.root_dir, 'scans', scene, scene + '_sdf.hdf')

        # read from hdf file
        try: 
            f = h5py.File(file, 'r')
        except: 
            f = h5py.File(file.replace('scans', 'scans_test'), 'r')
        voxels = np.array(f['sdf'][0]).astype(np.float16)
        if self.truncation_strategy == 'artificial':
            voxels[np.abs(voxels) >= truncation] = truncation
        elif self.truncation_strategy == 'standard':
            voxels[voxels > truncation] = truncation
            voxels[voxels < -truncation] = -truncation

        if semantic_grid:
            labels = np.array(f['sdf'][1]).astype(np.uint8) # Max 255 labels
            labels[voxels > truncation] = 0
            labels[voxels < -truncation] = 0

        # Add padding to grid to give more room to fusion net
        voxels = np.pad(voxels, self.pad, 'constant', constant_values=-truncation)
        print(scene, voxels.shape)
        bbox = np.zeros((3, 2))
        bbox[:, 0] = f.attrs['bbox'][:, 0] - self.pad * f.attrs['voxel_size'] * np.ones((1,1,1))
        bbox[:, 1] = bbox[:, 0] + f.attrs['voxel_size'] * np.array(voxels.shape)

        voxelgrid = Voxelgrid(f.attrs['voxel_size'])
        voxelgrid.from_array(voxels, bbox)
        if semantic_grid:
            labels = np.pad(labels, self.pad, 'constant', constant_values=0)
            semantic_grid = Voxelgrid(f.attrs['voxel_size'])
            semantic_grid.from_array(labels, bbox)
            return (voxelgrid, semantic_grid)
        return (voxelgrid,)

    def create_grid(self, scene, truncation):
        file = os.path.join(self.root_dir, 'scans', scene, scene + '_vh_clean_2.ply')
        try:
            points = trimesh.load(file).vertices
        except:
            points = trimesh.load(file.replace('scans', 'scans_test')).vertices

        voxel_size = 0.01

        bbox = np.zeros((3,2))
        bbox[:, 0] = [np.amin(points[:,0]), np.amin(points[:,1]), np.amin(points[:,2])]
        bbox[:, 1] = [np.amax(points[:,0]), np.amax(points[:,1]), np.amax(points[:,2])]
        del points

        vx = math.ceil((bbox[0, 1] - bbox[0, 0])/voxel_size) + 1
        vy = math.ceil((bbox[1, 1] - bbox[1, 0])/voxel_size) + 1
        vz = math.ceil((bbox[2, 1] - bbox[2, 0])/voxel_size) + 1

        voxels = truncation * np.ones((vx, vy, vz), dtype=np.float16)
        voxels = np.pad(voxels, self.pad, 'constant', constant_values=truncation)

        bbox[:, 0] = bbox[:, 0] - self.pad * voxel_size * np.ones((1,1,1))
        bbox[:, 1] = bbox[:, 0] + voxel_size * np.array(voxels.shape)

        voxelgrid = Voxelgrid(voxel_size)
        voxelgrid.from_array(voxels, bbox)
        return (voxelgrid,)

    def get_input_frame(self, frame_id, frame=None):
        if frame is None:
            frame_id = frame_id.split("/")
            path = os.path.join(self.root_dir, 'scans', frame_id[0], 'color', frame_id[1] + '.jpg')
            frame = cv2.imread(path).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, self.resolution[::-1], interpolation=cv2.INTER_NEAREST)
        return frame.astype(np.uint8)

    def get_depth_frame(self, frame_id, frame=None):
        if frame is None:
            frame_id = frame_id.split("/")
            path = os.path.join(self.root_dir, 'scans', frame_id[0], 'depth', frame_id[1] + '.png')
            frame = cv2.imread(path, -1).astype(np.float32)
        
        frame = cv2.resize(frame, self.resolution[::-1], interpolation=cv2.INTER_NEAREST)
        frame = frame / frame.max() * 255.0
        frame = frame[:, :, np.newaxis].repeat(3, axis=-1)
        return frame.astype(np.uint8)

    def get_semantic_frame(self, frame_id, frame=None):
        if frame is None:
            frame_id = frame_id.split("/")
            path = os.path.join(self.root_dir, 'scans', frame_id[0], 'label-filt', frame_id[1] + '.png')
            frame = cv2.imread(path, -1).astype(np.uint8)

        frame = cv2.resize(frame, self.resolution[::-1], interpolation=cv2.INTER_NEAREST)
        frame = np.array([self.rgb_map[int(s)] for s in frame.flatten().squeeze()])
        frame = frame.reshape(*self.resolution, 3)
        return frame.astype(np.uint8)

    def output_test(self, frame_id, frame):
        frame = self.main_ids[frame.flatten()].reshape(self.resolution[::-1])
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
        
        frame_id = frame_id.split("/")
        frame_name = '{}_{:6}.png'.format(frame_id[0], frame_id[1])
        path = os.path.join(self.root_dir, 'test_2d', frame_name)
        cv2.imwrite(path, frame.astype(np.uint64))