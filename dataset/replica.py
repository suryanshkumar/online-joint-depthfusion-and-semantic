import os
import sys
import glob
import h5py
import random
import cv2
import numpy as np

from torch.utils.data import Dataset
from graphics import Voxelgrid
from utils.mapping import *

class Replica(Dataset):

    def __init__(self, config_data):

        self.root_dir = config_data.root_dir

        self.resolution = (config_data.resy, config_data.resx)
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
        if self.input != 'image':
            self._load_depth()
        if self.target == 'depth_gt':
            self._load_depth_gt()

        if self.semantics == 'class30':
            self.rgb_map = replica_color_palette()
            #self.label_map = ids_to_nyu40()
            self.names_map = replica_names()
            self._load_semantic_images_gt()

    def _get_scene_load_order(self, nbr_load_scenes):
        # create list of training scenes
        scenes_list = list()
        with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
            for line in file:
                if line.split(' ')[0].split('/')[0] not in scenes_list:
                    scenes_list.append(line.split(' ')[0].split('/')[0])

        self._scenes = scenes_list

        # make sure nbr_load_scenes <= len(trajectory_list)
        if nbr_load_scenes > len(scenes_list):
            raise ValueError('nbr_load_scenes variable is lower than the number of scenes')
        # create nbr_load_scenes empty lists
        listdir = dict()
        for i in range(nbr_load_scenes):
            listdir[i] = list()

        # sample from trajectory_list and fill listdir
        while scenes_list:
            if nbr_load_scenes > len(scenes_list):
                scene_indices = random.sample(range(0, len(scenes_list)), len(scenes_list))
            else:
                scene_indices = random.sample(range(0, len(scenes_list)), nbr_load_scenes)

            for key, scene_idx in enumerate(scene_indices):
                listdir[key].append(scenes_list[scene_idx])

            scenes_list = [val for idx, val in enumerate(scenes_list) if idx not in scene_indices]

        # add the trajectories to the listdir
        for key in listdir:
            # create new list to replace the old one
            new_list_element = list()
            for scene in listdir[key]:
                for i in range(3):
                    new_list_element.append(scene + '/' + str(i + 1))

            # shuffle the list before replacing the old one
            random.shuffle(new_list_element)
            listdir[key] = new_list_element

        self.scenedir = listdir

    def _hybrid_load(self, modality):
        # create the full lists for each key in scenedir
        img_paths = []
        tmp_dict = dict()

        for key in self.scenedir:
            tmp_list = list()
            for trajectory in self.scenedir[key]:
                files = glob.glob(os.path.join(self.root_dir, trajectory, modality, '*'))
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

        if self.mode == 'val' or self.mode == 'test':
            img_paths = img_paths[::self.frame_ratio]
        
        return img_paths


    def _load_from_list(self, position, debug=False):
        # reading files from list
        img_paths = []

        with open(self.scene_list, 'r') as file:
            for line in file:
                line = line.rstrip('\n').split(' ')
                if line[0].split('/')[0] not in self._scenes:
                    self._scenes.append(line[0].split('/')[0])
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
        if self.scenedir is not None:
            if self.input == 'tof_depth':
                self.depth_images = self._hybrid_load('left_depth_noise_5.0')
            elif self.input == 'depth_gt':
                self.depth_images = self._hybrid_load('left_depth_gt')
            else:
                raise NotImplementedError
        else:
            if self.input == 'tof_depth':
                self.depth_images = self._load_from_list(1)
            elif self.input == 'depth_gt':
                self.depth_images = self._load_from_list(0)
            else:
                raise NotImplementedError

    def _load_depth_gt(self): 
        # loads the paths of the ground truth depth images to a list
        if self.scenedir is not None:
            self.depth_images_gt = self._hybrid_load('left_depth_gt')
        else:
            self.depth_images_gt = self._load_from_list(0)

    def _load_color(self):
        # loads the paths of the RGB images to a list
        if self.scenedir is not None:
            self.color_images = self._hybrid_load('left_rgb')
        else:
            self.color_images = self._load_from_list(-3)

    def _load_cameras(self):
        # loads the paths of the camera matrices to a list
        if self.scenedir is not None:
            self.cameras = self._hybrid_load('left_camera_matrix')
        else:
            self.cameras = self._load_from_list(-2)

    def _load_semantic_images_gt(self):
        # loads the paths of the ground truth semantic images to a list
        if self.scenedir is not None:
            self.semantic_images_gt = self._hybrid_load('left_' + self.semantics)
        else:
            self.semantic_images_gt = self._load_from_list(-1)


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
        scene = pathsplit[-4]
        trajectory = pathsplit[-3]
        frame = os.path.splitext(pathsplit[-1])[0]
        frame_id = '{}/{}/{}'.format(scene, trajectory, frame)
        sample['frame_id'] = frame_id

        image = cv2.imread(file) #BGR
        image = cv2.resize(image, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)

        if self.semantics:
            # load ground truth semantics
            file = self.semantic_images_gt[item]
            semantic = cv2.imread(file, -1)[:, :, 0]
            semantic = cv2.resize(semantic, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)

            if self.augmentations is not None:
                image, semantic = self.augmentations(image, semantic)

            sample['semantic_gt'] = semantic.astype(np.uint8)

        if self.normalize:
            mean = [179.66761167, 179.55742948, 188.2114891]
            std = [12.46442902, 12.55030275, 13.12021586]
            image = (image - mean) / std

        sample['image'] = image.astype(np.float32)

        if self.input in {'tof_depth', 'depth_gt'}:
            # load input depth map
            file = self.depth_images[item]
            depth = cv2.imread(file, -1)
            depth = cv2.resize(depth, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST) / 1000.
            sample[self.input] = depth.astype(np.float32)

            # define mask
            mask = (depth > 0.05) & (depth < 5.)
            sample['mask'] = mask

        if self.target == 'depth_gt':
            # load ground truth depth map
            file = self.depth_images_gt[item]
            depth = cv2.imread(file, -1)
            depth = cv2.resize(depth, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST) / 1000.
            sample[self.target] = depth.astype(np.float32)

        # load extrinsics
        file = self.cameras[item]

        extrinsics = np.loadtxt(file)
        extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
        # the fusion code expects that the camera coordinate system is such that z is in the
        # camera viewing direction, y is down and x is to the right. This is achieved by a serie of rotations
        rot_180_around_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).astype(np.float32)
        rot_180_around_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).astype(np.float32)
        rot_90_around_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(np.float32)
        rotation = np.matmul(rot_180_around_z, rot_180_around_y)
        extrinsics =  np.matmul(rotation, extrinsics[0:3, 0:4])
        extrinsics = np.linalg.inv(np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0))

        sample['extrinsics'] = np.matmul(rot_90_around_x, extrinsics[0:3, 0:4])

        hfov = 90.
        f = self.resolution[0]/2.*(1./np.tan(np.deg2rad(hfov)/2))
        shift = self.resolution[0]/2

        # load intrinsics
        intrinsics = np.asarray([[f, 0., shift],
                                 [0., f, shift],
                                 [0., 0., 1.]])

        sample['intrinsics'] = intrinsics

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_grid(self, scene, truncation, semantic_grid):
        if self.semantics:
            file = os.path.join(self.root_dir, scene, 'gt_semantic_sdf', 'semantic_sdf.hdf')
        else:
            file = os.path.join(self.root_dir, scene, 'gt_semantic_sdf', 'sdf.hdf')

        # read from hdf file!
        f = h5py.File(file, 'r')
        voxels = np.array(f['sdf'][0]).astype(np.float16)
        if self.truncation_strategy == 'artificial':
            voxels[np.abs(voxels) >= truncation] = truncation
        elif self.truncation_strategy == 'standard':
            voxels[voxels > truncation] = truncation
            voxels[voxels < -truncation] = -truncation
            #voxels[voxels < -truncation] = truncation

        if self.semantics:
            labels = np.array(f['sdf'][1]).astype(np.uint8) # Max 255 labels
            labels[voxels > truncation] = 0                 # Free space is undefined (ID 0)
            labels[voxels < -truncation] = 0                 # Free space is undefined (ID 0)

        # Add padding to grid to give more room to fusion net
        voxels = np.pad(voxels, self.pad, 'constant', constant_values=-truncation)
        print(scene, voxels.shape)
        bbox = np.zeros((3, 2))
        bbox[:, 0] = f.attrs['bbox'][:, 0] - self.pad * f.attrs['voxel_size'] * np.ones((1,1,1))
        bbox[:, 1] = bbox[:, 0] + f.attrs['voxel_size'] * np.array(voxels.shape)

        voxelgrid = Voxelgrid(f.attrs['voxel_size'])
        voxelgrid.from_array(voxels, bbox)
        if self.semantics:
            labels = np.pad(labels, self.pad, 'constant', constant_values=0)
            semantic_grid = Voxelgrid(f.attrs['voxel_size'])
            semantic_grid.from_array(labels, bbox)
            return (voxelgrid, semantic_grid)
        return (voxelgrid,)

    def get_input_frame(self, frame_id, frame=None):
        if frame is None:
            frame_id = frame_id.split("/")
            path = os.path.join(self.root_dir, frame_id[0], frame_id[1], 'left_rgb', frame_id[2] + '.png')
            frame = cv2.imread(path).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)
        return frame.astype(np.uint8)

    def get_depth_frame(self, frame_id, frame=None):
        if frame is None:
            frame_id = frame_id.split("/")
            path = os.path.join(self.root_dir, frame_id[0], frame_id[1], 'left_depth_gt', frame_id[2] + '.png')
            frame = cv2.imread(path, -1).astype(np.float32)
        
        frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)
        frame = frame / frame.max() * 255.0
        frame = frame[:, :, np.newaxis].repeat(3, axis=-1)
        return frame.astype(np.uint8)

    def get_semantic_frame(self, frame_id, frame=None):
        if frame is None:
            frame_id = frame_id.split("/")
            path = os.path.join(self.root_dir, frame_id[0], frame_id[1], 'semantic_depth_gt', frame_id[2] + '.png')
            frame = cv2.imread(path, -1).astype(np.uint8)[:, :, 0]

        frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)
        frame = np.array([self.rgb_map[int(s)] for s in frame.flatten().squeeze()])
        frame = frame.reshape(self.resolution[0], self.resolution[1], 3)
        return frame.astype(np.uint8)

