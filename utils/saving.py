import os
import json
import torch
import numpy as np
import h5py
import sys
import skimage
import trimesh

from skimage import io
from skimage.exposure import rescale_intensity, is_low_contrast

from utils.mapping import get_mapping


def save_tsdf(filename, data):
	with h5py.File(filename, 'w') as file:
		file.create_dataset('TSDF',
			  shape=data.shape,
			  data=data,
			  compression='gzip',
			  compression_opts=9)


def save_weights(filename, data):
	with h5py.File(filename, 'w') as file:
		file.create_dataset('weights',
			  shape=data.shape,
			  data=data,
			  compression='gzip', 
			  compression_opts=9)

def save_semantics(filename, data):
	with h5py.File(filename, 'w') as file:
		file.create_dataset('semantics',
			  shape=data.shape,
			  data=data,
			  compression='gzip',
			  compression_opts=9)


def save_ply(filename, data):
	voxel_size = 0.01
	vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(data, 
																		 level=0, 
																		 spacing=(voxel_size, voxel_size, voxel_size))
	mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
	mesh.export(filename)


def save_image(filename, image):
	if is_low_contrast(image):
		rescale_intensity(image, in_range='image', out_range='dtype')

	io.imsave(filename, image, check_contrast=False)

	return image


def save_config_to_json(path, config):
	"""Saves config to json file
	"""
	with open(os.path.join(path, 'config.json'), 'w') as file:
		json.dump(config, file)


def save_checkpoint(state, checkpoint, is_best=False, name=None):
	"""Saves model and training parameters
	at checkpoint + 'last.pth.tar'.
	If is_best==True, also saves
	checkpoint + 'best.pth.tar'
	Args:
	   state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
	   is_best: (bool) True if it is the best model seen till now
	   is_best: (bool) True if it is the final model
	   checkpoint: (string) folder where parameters are to be saved
	"""
	if not os.path.exists(checkpoint):
	   print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
	   os.mkdir(checkpoint)

	if is_best:
		if name is not None:
			filepath = os.path.join(checkpoint, name)
			torch.save(state, filepath)
		else:
			filepath = os.path.join(checkpoint, 'best.pth.tar')
			torch.save(state, filepath)
	else:
		filepath = os.path.join(checkpoint, 'last.pth.tar')
		torch.save(state, filepath)
