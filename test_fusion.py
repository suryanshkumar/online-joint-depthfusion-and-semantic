import torch
import argparse
import os
import random
import numpy as np
import time

from utils import loading
from utils import setup
from utils import transform

from modules.pipeline import Pipeline

from tqdm import tqdm

def arg_parse():
	parser = argparse.ArgumentParser(description='Script for testing RoutedFusion')
	parser.add_argument('--config', required=True)

	args = parser.parse_args()
	return vars(args)


def test_fusion(config):
	
	# define output dir
	model_name = config.TESTING.fusion_model_path.split('/')[-3]
	if config.DATA.semantics:
		test_dir = os.path.join(config.SETTINGS.experiment_path, model_name, 'test', config.DATA.semantics)
	else:
		test_dir = os.path.join(config.SETTINGS.experiment_path, model_name, 'test', 'no_semantics')

	if not os.path.exists(test_dir):
		os.makedirs(test_dir)

	if config.SETTINGS.gpu:
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	config.SETTINGS.device = device

	# get test dataset
	test_data_config = setup.get_data_config(config, mode='test')
	dataset = setup.get_data(config.DATA.dataset, test_data_config)
	loader = torch.utils.data.DataLoader(dataset, 
										 batch_size=config.TESTING.test_batch_size, 
										 shuffle=config.TESTING.test_shuffle, 
										 pin_memory=True,
										 num_workers=config.SETTINGS.num_workers)

	# get test database
	database = setup.get_database(dataset, test_data_config)
	
	# setup pipeline
	pipeline = Pipeline(config)
	pipeline = pipeline.to(device)

	total_params = sum(p.numel() for p in pipeline._fusion_network.parameters())
	print('Fusion Parameters:', total_params)  

	# load pretrained fusion model into parameters
	checkpoint = torch.load(config.TESTING.fusion_model_path, map_location=device)
	weights = loading.remove_parent(checkpoint["model_state"], '_fusion_network')
	pipeline._fusion_network.load_state_dict(weights)

	# load pretrained semantic models into parameters
	if config.DATA.semantics and config.DATA.semantic_strategy == 'predict':
		checkpoint = torch.load(config.TESTING.semantic_2d_model_path, map_location=device)
		weights = loading.remove_parent(checkpoint["model_state"], 'module')
		pipeline._semantic_2d_network.load_state_dict(weights)

	pipeline.eval()
	with torch.no_grad():
		for i, batch in tqdm(enumerate(loader), total=len(dataset), mininterval=30):
			if not torch.all(torch.isfinite(batch['extrinsics'])): continue
			# put all data on GPU
			batch = transform.to_device(batch, device)
			# fusion pipeline
			pipeline.fuse(batch, database, device)

	database.to_numpy()

	# filter outliers
	print('Filter outliers > {}'.format(config.TESTING.outlier_filter_val))
	database.filter(value=config.TESTING.outlier_filter_val)

	if config.DATA.semantics:
		database.filter_semantics(value=5)

	logger = setup.get_logger(test_dir, 'test')
	
	# evaluate test scenes
	test_eval, test_eval_per_scene = database.evaluate(mode='test')
	# save test_eval to log file
	logger.info('Average test results over test scenes:')
	for metric in test_eval:
		logger.info(metric + ': ' + str(test_eval[metric]))

	logger.info('Per scene results')
	for scene in test_eval_per_scene:
		logger.info('Scene: ' + scene)
		for metric in test_eval_per_scene[scene]:
			logger.info(metric + ': ' + str(test_eval_per_scene[scene][metric]))
	logger.info('\n')

	if config.DATA.semantics and config.DATA.semantic_grid:
		test_eval, class_iou_per_scene = database.evaluate_semantics(mode='test')
		logger.info('Average semantic results over test scenes')
		for k, v in test_eval.items():
			logger.info("{:12}:\t{}".format(k, v))

		logger.info('Per scene semantic results:')
		for scene in class_iou_per_scene:
			logger.info('Scene: ' + scene)
			for k, v in class_iou_per_scene[scene].items():
				logger.info("{}: {}".format(k, v))
		logger.info('\n')

	# save ply-files of test scenes
	for scene_id in database.scenes_est.keys():
		database.save(path=test_dir, save_mode=config.SETTINGS.save_mode, scene_id=scene_id)


if __name__ == '__main__':

	# parse commandline arguments
	args = arg_parse()

	# load config
	test_config = loading.load_config_from_yaml(args['config'])

	test_fusion(test_config)
