import torch
import argparse
import datetime
import os
import numpy as np
import shutil

from tqdm import tqdm
from numpy import random

from utils import setup
from utils import loading
from utils.saving import save_image
from utils.metrics import runningScore

from modules.adapnet import AdapNet


def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', required=True)
	parser.add_argument('--comment', type=str, default='')

	args = parser.parse_args()

	return vars(args)


def prepare_input_data(batch, config, device):
	inputs = {}
	inputs['image'] = batch['image'] / 255.0
	inputs['image'] = inputs['image'].to(device).float() # (batch size, channels, height, width)

	if config.DATA.input != 'image':
		inputs[config.DATA.input] = batch[config.DATA.input].unsqueeze(1)
		inputs[config.DATA.input] = inputs[config.DATA.input].to(device).float()

	target = batch[config.DATA.target] # (batch size, height, width)
	target = target.to(device).long()

	return inputs, target


def weights_init(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.xavier_normal_(m.weight)


def test(args, config):

	# set seed for reproducibility
	if config.SETTINGS.seed:
		np.random.seed(config.SETTINGS.seed)
		torch.manual_seed(config.SETTINGS.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		
	# define output dir
	path = os.path.dirname(os.path.dirname(config.TESTING.semantic_model_path)) 
	test_dir = path + '/test'
	out_dir = path + '/output'

	if not os.path.exists(test_dir):
		os.makedirs(test_dir)

	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)  # remove test images
	os.makedirs(out_dir)

	if config.SETTINGS.gpu:
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')
	config.SETTINGS.device = device
		
	# get test dataset
	test_data_config = setup.get_data_config(config, mode='test')
	test_dataset = setup.get_data(config.DATA.dataset, test_data_config)
	test_loader = torch.utils.data.DataLoader(test_dataset, 
												batch_size=config.TESTING.test_batch_size, 
												shuffle=config.TESTING.test_shuffle, 
												pin_memory=True,
												num_workers=config.SETTINGS.num_workers)

	# define metrics
	ignore_index = 0 if config.DATA.dataset == 'ScanNet' else -100
	running_metrics_test = runningScore(config.SEMANTIC_MODEL.n_classes, ignore_index=ignore_index)

	# define logger
	logger = setup.get_logger(test_dir, 'test')
	
	# define model
	model = AdapNet(config.SEMANTIC_MODEL)

	if os.path.isfile(config.TESTING.semantic_model_path):
		logger.info("Model {}".format(config.TESTING.semantic_model_path))
		checkpoint = torch.load(config.TESTING.semantic_model_path, map_location=device)
		weights = checkpoint["model_state"]
		weights = loading.remove_parent(weights, 'module')
		model.load_state_dict(weights)
	else:
		print("No model found at '{}'".format(config.TESTING.semantic_model_path))

    # if available, divide batch size on multiple GPUs
	if torch.cuda.device_count() > 1 and config.SETTINGS.multigpu:
		print("Let's use {} GPUs".format(torch.cuda.device_count()))
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
		model = nn.DataParallel(model)

	model = model.to(device)

	total_params = sum(p.numel() for p in model.parameters())
	print('Parameters:', total_params)    

	n_test_batches = int(len(test_dataset) / config.TESTING.test_batch_size)

	# sample validation visualization frames
	test_vis_ids = np.random.choice(np.arange(0, n_test_batches), 20, replace=False)

	model.eval()
	with torch.no_grad():
		for i, batch in enumerate(tqdm(test_loader, total=n_test_batches, mininterval=30)):

			inputs, target = prepare_input_data(batch, config, device)

			if inputs[config.DATA.input].shape[1] == 1:
				inputs[config.DATA.input] = inputs[config.DATA.input].repeat(1, 3, 1, 1)

			if config.SEMANTIC_MODEL.stage == 1:
				output = model.forward(inputs[config.DATA.input])
			else:
				output = model.forward(inputs['image'], inputs[config.DATA.input]) # res, aux1, aux2

			res = torch.softmax(output[0], dim=1)
			pred = res.data.max(1)[1].cpu().numpy()
			gt = target.data.cpu().numpy()

			running_metrics_test.update(gt, pred)

			# visualize frames
			if i in test_vis_ids:
				frame_id = batch['frame_id'][0]
				frame_input = test_dataset.get_input_frame(frame_id)
				frame_depth = test_dataset.get_depth_frame(frame_id)

				frame_est = torch.max(output[0], 1)[1]
				frame_est = frame_est[0, :, :].cpu().detach().numpy()
				frame_est = test_dataset.get_semantic_frame(frame_id, frame_est)
				frame_gt = target[0, :, :].cpu().detach().numpy()
				frame_gt = test_dataset.get_semantic_frame(frame_id, frame_gt)

				w_line = np.ones((frame_input.shape[0], 1, 3)).astype(np.uint8)
				frame_cat = np.concatenate([frame_input, w_line, frame_depth, w_line, frame_gt, w_line, frame_est], axis=1)
				save_image('{}/{}.png'.format(out_dir, frame_id.replace('/','_')), frame_cat)

	score, class_iou = running_metrics_test.get_scores()

	logger.info('Average test results over test scenes')
	for k, v in score.items():
		logger.info("{:12}:\t{}".format(k, v))
	for k, v in class_iou.items():
		name = test_dataset.names_map[int(k)]
		logger.info("{:2}: {:16}:\t{}".format(k, name, v))
	logger.info('\n')


if __name__ == '__main__':

	# get arguments
	args = arg_parser()

	# get configs
	config = loading.load_config_from_yaml(args['config'])
	print(args['comment'])

	# test
	test(args, config)

