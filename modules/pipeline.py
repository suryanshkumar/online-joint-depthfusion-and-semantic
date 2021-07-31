import torch
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

from modules.extractor import Extractor
from modules.model import FusionNet_v1, FusionNet_v2, FusionNet_v3
from modules.integrator import Integrator
from modules.adapnet import AdapNet

class Pipeline(torch.nn.Module):

	def __init__(self, config):

		super(Pipeline, self).__init__()

		self.config = config
		self.n_points = config.FUSION_MODEL.n_points

		if config.DATA.semantics:
			self.n_classes = config.SEMANTIC_2D_MODEL.n_classes

		config.FUSION_MODEL.resx = config.DATA.resx
		config.FUSION_MODEL.resy = config.DATA.resy

		if config.FUSION_MODEL.name == 'v1':
			self._fusion_network = FusionNet_v1(config.FUSION_MODEL)
		elif config.FUSION_MODEL.name == 'v2':
			self._fusion_network = FusionNet_v2(config.FUSION_MODEL)
		elif config.FUSION_MODEL.name == 'v3':
			self._fusion_network = FusionNet_v3(config.FUSION_MODEL)

		if config.DATA.semantics and config.DATA.semantic_strategy == 'predict':
			self._semantic_2d_network = AdapNet(config.SEMANTIC_2D_MODEL)
		else:
			self._semantic_2d_network = None

		self._extractor = Extractor(config)
		self._integrator = Integrator(config)

	def _segmentation(self, data):
		inputs = {}
		inputs['image'] = data['image'] / 255.0
		inputs['image'] = inputs['image'].to(self.device).float() # (batch size, channels, height, width)

		in_ = self.config.DATA.input

		if self.config.DATA.input != 'image':
			inputs[in_] = data[in_].repeat(1, 3, 1, 1).to(self.device).float()

		if self.config.SEMANTIC_2D_MODEL.stage == 1:
			output = self._semantic_2d_network.forward(inputs[in_])
		else:
			output = self._semantic_2d_network.forward(inputs['image'], inputs[in_])
	
		sem_histograms = torch.softmax(output[0], dim=1)
		sem_histograms = sem_histograms.permute(0, 2, 3, 1)

		return sem_histograms

	def _fusion(self, inputs, values):
		b, _, h, w = self._shape

		tsdf_est = self._fusion_network.forward(inputs)

		tsdf_est = tsdf_est.permute(0, 2, 3, 1)

		tsdf_est = tsdf_est[:, :, :, :self.n_points]
		tsdf_est = tsdf_est.view(b, h * w, self.n_points)

		return tsdf_est

	def _prepare_fusion_input(self, frame, values, semantics):

		# get frame shape
		b, _, h, w = self._shape
		
		inputs = {}

		# extracting data
		tsdf_input = values['fusion_values']
		tsdf_weights = values['fusion_weights']

		# reshaping data
		inputs['tsdf_values'] = tsdf_input.view(b, h, w, self.n_points)
		inputs['tsdf_weights'] = tsdf_weights.view(b, h, w, self.n_points)
		del tsdf_input, tsdf_weights

		inputs['tsdf_frame'] = frame.unsqueeze(-1)

		if self.config.FUSION_MODEL.use_semantics:
			assert semantics is not None
			semantics = semantics.unsqueeze(-1).float()
			semantics = (1 + semantics) / self.n_classes	# (0, 1]
			inputs['semantic_frame'] = semantics
			del semantics

		# permute input
		inputs = {k: v.permute(0, -1, 1, 2).contiguous() for k, v in inputs.items()}

		return inputs

	def _prepare_fusion_output(self, values, tsdf_est, filtered_frame=None, values_gt=None):
		b, _, h, w = self._shape

		# computing weighted updates for loss calculation
		tsdf_old = values['fusion_values']
		weights = values['fusion_weights']
		tsdf_new = torch.clamp(tsdf_est,
							   -self.config.DATA.init_value,
							   self.config.DATA.init_value)
		weights = weights.view(b, h * w, self.n_points)
		weights = torch.where(weights < 0, torch.zeros_like(weights), weights)

		tsdf_fused = (weights * tsdf_old + tsdf_new) / (weights + 1)

		if values_gt is None:
			return tsdf_fused
		else:
			assert filtered_frame is not None

			# masking invalid losses
			tsdf_fused = masking(tsdf_fused, filtered_frame.view(b, h * w, 1))

			tsdf_target = values_gt['fusion_values']
			tsdf_target = tsdf_target.view(b, h * w, self.n_points)
			tsdf_target = masking(tsdf_target, filtered_frame.view(b, h * w, 1))

			output = {
				'tsdf_est': tsdf_est,
				'tsdf_fused': tsdf_fused,
				'tsdf_target': tsdf_target
			}
			return output

	def _prepare_volume_update(self, values, tsdf_est, tsdf_frame, semantics, scores):
		b, _, h, w = self._shape
		tail_points = self.config.FUSION_MODEL.n_tail_points

		updates = {}

		depth = tsdf_frame.view(b, h * w, 1)

		valid = (depth != 0.)
		valid = valid.nonzero()[:, 1]
		
		values['points'] = values['points'][:, :, :self.n_points].contiguous()
		
		updates['points'] = values['points'][:, valid, :tail_points, :]
		updates['indices'] = values['indices'][:, valid, :tail_points, :, :]
		updates['weights'] = values['weights'][:, valid, :tail_points, :]
		updates['values'] = tsdf_est[:, valid, :tail_points]
		updates['values'] = torch.clamp(updates['values'],
									-self.config.DATA.init_value,
									self.config.DATA.init_value)

		if self.config.DATA.semantics:
			assert semantics is not None
			semantics = semantics.to(self.device).view(b, h * w, -1).contiguous()
			semantics = semantics.unsqueeze(-2).repeat(1, 1, tsdf_est.shape[2], 1)
			updates['semantics'] = semantics[:, valid, :tail_points, ...]

			assert scores is not None
			scores = scores.to(self.device).view(b, h * w, -1).contiguous()
			scores = scores.unsqueeze(-2).repeat(1, 1, tsdf_est.shape[2], 1)
			updates['scores'] = scores[:, valid, :tail_points, ...]
	
		del valid

		return updates

	def fuse(self, batch, database, device):

		self.device = device

		self._shape = batch['image'].shape
		b, _, h, w = self._shape

		# predict semantics or use gt
		if self.config.DATA.semantics:
			if self.config.DATA.semantic_strategy == 'predict':
				sem_histograms = self._segmentation(batch)
				scores, sem_ids = sem_histograms.max(dim=-1)
				del sem_histograms
			elif self.config.DATA.semantic_strategy == 'gt':
				sem_ids = batch['semantic_gt'].long()
				scores = torch.ones_like(sem_ids).float()
			else:
				ValueError('Error! Valid value for DATA.semantic_strategy are "gt" or "predict".')
		else:
			scores = None
			sem_ids = None

		frame = batch[self.config.DATA.input].squeeze_(1).to(self.device)
		filtered_frame = torch.where(batch['mask'], frame, torch.zeros_like(frame, device=self.device))

		# get current tsdf values
		scene_id = batch['frame_id'][0].split('/')[0]
		volume = database[scene_id]

		values = self._extractor.forward(frame,
										 batch['extrinsics'],
										 batch['intrinsics'],
										 volume['current'],
										 volume['weights'],
										 volume['origin'],
										 volume['resolution'])
		
		tsdf_input = self._prepare_fusion_input(frame, 
												values,
												sem_ids)

		tsdf_est = self._fusion(tsdf_input, values)

		if self.config.DATA.semantics:
			sem_ids = sem_ids.type(torch.uint8)
			scores_volume = volume['scores']
			semantic_volume = volume['ids_est']
		else:
			sem_ids = None
			scores_volume = None
			semantic_volume = None

		updates = self._prepare_volume_update(values, 
											  tsdf_est, 
											  filtered_frame, 
											  sem_ids,
											  scores)

		del values, sem_ids, scores, filtered_frame, tsdf_est

		values, weights, sem_ids, scores = self._integrator.forward(updates,
															  volume['current'],
															  volume['weights'],
															  scores_volume,
															  semantic_volume)

		database.state[scene_id] = True
		database.scenes_est[scene_id].volume = values
		database.fusion_weights[scene_id] = weights
		if self.config.DATA.semantics:
			database.ids_est[scene_id].volume = sem_ids
			database.scores[scene_id].volume = scores

		del values, weights, sem_ids, scores

		return


	def fuse_training(self, batch, database, device):

		"""
			Learned real-time depth map fusion pipeline

			:param batch:
			:param extractor:
			:param routing_model:
			:param tsdf_model:
			:param database:
			:param device:
			:param config:
			:param routing_config:
			:param mode:
			:return:
			"""

		self.device = device

		self._shape = batch['image'].shape
		b, _, h, w = self._shape

		sem_ids = None
		sem_histograms = None
		scores_volume = None

		# predict semantics
		if self.config.DATA.semantics:
			if self.config.DATA.semantic_strategy == 'predict':
				with torch.no_grad():
					sem_histograms = self._segmentation(batch)
					scores, sem_ids = sem_histograms.max(dim=-1)
					sem_ids = sem_ids.type(torch.uint8)
					del sem_histograms
			elif self.config.DATA.semantic_strategy == 'gt':
				sem_ids = batch['semantic_gt'].type(torch.uint8)
				scores = torch.ones_like(sem_ids).float()
			else:
				ValueError('Error! Valid value for DATA.semantic_strategy are "gt" or "predict".')
		else:
			sem_ids = None
			scores = None

		frame = batch[self.config.DATA.input].squeeze_(1)
		filtered_frame = torch.where(batch['mask'], frame, torch.zeros_like(frame, device=self.device))

		# get current tsdf values
		scene_id = batch['frame_id'][0].split('/')[0]
		volume = database[scene_id]

		values = self._extractor.forward(frame,
										 batch['extrinsics'],
										 batch['intrinsics'],
										 volume['current'],
										 volume['weights'],
										 volume['origin'],
										 volume['resolution'])

		values_gt = self._extractor.forward(frame,
											batch['extrinsics'],
											batch['intrinsics'],
											volume['gt'],
											volume['weights'],
											volume['origin'],
											volume['resolution'])

		# prepare input for fusion net
		tsdf_input = self._prepare_fusion_input(frame, 
												values, 
												sem_ids)
		# train fusion net
		tsdf_est = self._fusion(tsdf_input, values)
		del tsdf_input

		# prepare output for loss calculation (if values_gt --> return output, else return only tsdf update)
		output = self._prepare_fusion_output(values, 
											 tsdf_est, 
											 filtered_frame, 
											 values_gt)

		if self.config.DATA.semantics:
			sem_ids = sem_ids.type(torch.uint8)
			scores_volume = volume['scores']
			semantic_volume = volume['ids_est']
		else:
			sem_ids = None
			scores_volume = None
			semantic_volume = None
		
		# prepare updates (dict)
		updates = self._prepare_volume_update(values, 
											  tsdf_est, 
											  filtered_frame, 
											  sem_ids,
											  scores)

		del values, values_gt, sem_ids, scores, filtered_frame, tsdf_est

		# integrate updates in the volumes
		# if semantic volume and scores are not None--> update semantics
		# during training do not update semantic volume to speed up
		values, weights, sem_ids, scores = self._integrator.forward(updates,
															  volume['current'],
															  volume['weights'],
															  scores_volume,
															  semantic_volume,
															  test=False)

		database.state[scene_id] = True
		database.scenes_est[scene_id].volume = values.detach()
		database.fusion_weights[scene_id] = weights.detach()

		return output

def masking(x, values, threshold=0., option='ueq'):

	if option == 'leq':

		if x.dim() == 2:
			valid = (values <= threshold)[0, :, 0]
			xvalid = valid.nonzero()[:, 0]
			xmasked = x[:, xvalid]
		if x.dim() == 3:
			valid = (values <= threshold)[0, :, 0]
			xvalid = valid.nonzero()[:, 0]
			xmasked = x[:, xvalid, :]

	if option == 'geq':

		if x.dim() == 2:
			valid = (values >= threshold)[0, :, 0]
			xvalid = valid.nonzero()[:, 0]
			xmasked = x[:, xvalid]
		if x.dim() == 3:
			valid = (values >= threshold)[0, :, 0]
			xvalid = valid.nonzero()[:, 0]
			xmasked = x[:, xvalid, :]

	if option == 'eq':

		if x.dim() == 2:
			valid = (values == threshold)[0, :, 0]
			xvalid = valid.nonzero()[:, 0]
			xmasked = x[:, xvalid]
		if x.dim() == 3:
			valid = (values == threshold)[0, :, 0]
			xvalid = valid.nonzero()[:, 0]
			xmasked = x[:, xvalid, :]

	if option == 'ueq':
		valid = (values != threshold)[0, :, 0]
		xvalid = valid.nonzero()[:, 0]
		xmasked = x[:, xvalid, ...]

	return xmasked
