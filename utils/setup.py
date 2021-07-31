import torch
import os
import logging
import functools

from dataset import Replica
from dataset import ScanNet

from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, StepLR

import trimesh
import skimage.measure

from modules.database import Database

from utils import transform
from utils.saving import *
from utils.augmentations import *
from utils.schedulers import *
from utils.loss import *

from easydict import EasyDict
from copy import copy



def get_data_config(config, mode):
	data_config = copy(config.DATA)

	data_config.device = config.SETTINGS.device
	
	try: data_config.implementation = config.SETTINGS.implementation
	except: data_config.implementation = None

	if mode == 'train':
		data_config.mode = 'train'
		data_config.scene_list = data_config.train_scene_list
		data_config.frame_ratio = config.TRAINING.train_ratio
	elif mode == 'val':
		data_config.mode = 'val'
		data_config.scene_list = data_config.val_scene_list
		data_config.frame_ratio = config.TRAINING.val_ratio
	elif mode == 'test':
		data_config.mode = 'test'
		data_config.scene_list = data_config.test_scene_list
		data_config.frame_ratio = config.TESTING.test_ratio

	try: x = data_config.fusion_strategy
	except AttributeError: data_config.fusion_strategy = None

	try: x = data_config.data_load_strategy
	except AttributeError: data_config.data_load_strategy = None

	try: x = data_config.intensity_grad
	except AttributeError: data_config.intensity_grad = False

	try: data_config.n_classes = config.SEMANTIC_2D_MODEL.n_classes
	except: data_config.n_classes = None

	if mode == 'train':
		try: data_config.augmentations = get_composed_augmentations(config.DATA.augmentations)
		except AttributeError: data_config.augmentations = None
	else:
		data_config.augmentations = None

	data_config.transform = transform.ToTensor()

	return data_config


def get_data(dataset, config):
	try:
		return eval(dataset)(config.DATA)
	except AttributeError:
		return eval(dataset)(config)


def get_database(dataset, config):
	#TODO: make this better
	database_config = copy(config)
	#database_config.transform = transform.ToTensor()
	#database_config.scene_list = eval('config.DATA.{}_scene_list'.format(mode))
	return Database(dataset, database_config)


def get_workspace(config):
	workspace_path = os.path.join(config.SETTINGS.experiment_path,
								  config.TIMESTAMP)
	workspace = Workspace(workspace_path)
	workspace.save_config(config)
	return workspace


def get_logger(path, name='training'):
	filehandler = logging.FileHandler(os.path.join(path, '{}.logs'.format(name)), 'a')
	consolehandler = logging.StreamHandler()

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	filehandler.setFormatter(formatter)
	consolehandler.setFormatter(formatter)

	logger = logging.getLogger(name)

	for hdlr in logger.handlers[:]:  # remove all old handlers
		logger.removeHandler(hdlr)

	logger.addHandler(filehandler)  # set the new handler
	logger.addHandler(consolehandler)

	logger.setLevel(logging.DEBUG)

	return logger


def get_composed_augmentations(aug_dict):
	key2aug = {
		"gamma": AdjustGamma,
		"hue": AdjustHue,
		"brightness": AdjustBrightness,
		"saturation": AdjustSaturation,
		"contrast": AdjustContrast,
		"rcrop": RandomCrop,
		"hflip": RandomHorizontallyFlip,
		"vflip": RandomVerticallyFlip,
		"scale": Scale,
		"rscale_crop": RandomScaleCrop,
		"rsize": RandomSized,
		"rsizecrop": RandomSizedCrop,
		"rotate": RandomRotate,
		"translate": RandomTranslate,
		"ccrop": CenterCrop,
	}
	if aug_dict is None:
		return None

	augmentations = []
	for aug_key, aug_param in aug_dict.items():
		augmentations.append(key2aug[aug_key](aug_param))
	return Compose(augmentations)


def get_optimizer(opt_dict):
	key2opt = {
		"sgd": SGD,
		"adam": Adam,
		"asgd": ASGD,
		"adamax": Adamax,
		"adadelta": Adadelta,
		"adagrad": Adagrad,
		"rmsprop": RMSprop,
	}
	if opt_dict is None:
		return SGD

	else:
		opt_name = opt_dict.name
		if opt_name not in key2opt:
			raise NotImplementedError("Optimizer {} not implemented".format(opt_name))
		return key2opt[opt_name]


def get_scheduler(optimizer, scheduler_dict):
	key2scheduler = {
		"constant_lr": ConstantLR,
		"poly_lr": PolynomialLR,
		"multi_step": MultiStepLR,
		"step": StepLR,
		"cosine_annealing": CosineAnnealingLR,
		"exp_lr": ExponentialLR,
	}
	if scheduler_dict is None:
		return ConstantLR(optimizer)

	s_type = scheduler_dict["name"]
	scheduler_dict.pop("name")

	warmup_dict = {}
	if "warmup_iters" in scheduler_dict:
		warmup_dict["warmup_iters"] = scheduler_dict.get("warmup_iters", 100)
		warmup_dict["mode"] = scheduler_dict.get("warmup_mode", "linear")
		warmup_dict["gamma"] = scheduler_dict.get("warmup_factor", 0.2)

		scheduler_dict.pop("warmup_iters", None)
		scheduler_dict.pop("warmup_mode", None)
		scheduler_dict.pop("warmup_factor", None)

		base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
		return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

	return key2scheduler[s_type](optimizer, **scheduler_dict)


def get_loss_function(loss_dict, device):
	key2loss = {
		"cross_entropy": CrossEntropy,
		"bootstrapped_cross_entropy": BootstrappedCrossEntropy,
		"multi_scale_cross_entropy": MultiScaleCrossEntropy,
		"fusion": FusionLoss
	}

	if loss_dict is None:
		return CrossEntropy

	else:
		loss_name = loss_dict.name
		loss_params = {k: v for k, v in loss_dict.items() if k not in {"name", "weight"}}
		try:	# weight may not be defined
			if loss_dict.weight:
				weight = np.loadtxt(loss_dict.weight)
				print("Loaded weights from", loss_dict.weight)
				loss_params['weight'] = torch.Tensor(weight).to(device)
		except:
			pass

		if loss_name not in key2loss:
			raise NotImplementedError("Loss {} not implemented".format(loss_name))

		return key2loss[loss_name](**loss_params)


class Workspace(object):

	def __init__(self, path):

		self.workspace_path = path
		self.model_path = os.path.join(path, 'model')
		self.log_path = os.path.join(path, 'logs')
		self.output_path = os.path.join(path, 'output')

		os.makedirs(self.workspace_path)
		os.makedirs(self.model_path)
		os.makedirs(self.log_path)
		os.makedirs(self.output_path)

		self.writer = SummaryWriter(self.log_path)

		self._init_logger()

	def _init_logger(self):
		self.train_logger = get_logger(self.log_path, 'training')
		self.val_logger = get_logger(self.log_path, 'validation')

	def save_config(self, config):
		print('Saving config to ', self.workspace_path)
		save_config_to_json(self.workspace_path, config)

	def save_model_state(self, state, is_best=False, name=None):
		save_checkpoint(state, self.model_path, is_best, name)

	def save_tsdf_data(self, file, data):
		tsdf_file = os.path.join(self.output_path, file)
		save_tsdf(tsdf_file, data)

	def save_weights_data(self, file, data):
		weight_files = os.path.join(self.output_path, file)
		save_weights(weight_files, data)

	def save_semantic_data(self, file, data):
		semantic_file = os.path.join(self.output_path, file)
		save_semantics(semantic_file, data)

	def save_ply_data(self, file, data):
		ply_files = os.path.join(self.output_path, file)
		save_ply(ply_files, data)

	def log(self, message, mode='train'):
		if mode == 'train':
			self.train_logger.info(message)
		elif mode == 'val':
			self.val_logger.info(message)

