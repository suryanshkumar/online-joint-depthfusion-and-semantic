import torch
import torch.nn.functional as F
import numpy as np


class CrossEntropy(torch.nn.Module):

	def __init__(self, weight=None, reduction='mean'):
		self.weight = weight
		self.reduction = reduction

	def forward(self, inputs, target):
		return F.cross_entropy(inputs, target, weight=self.weight, reduction=self.reduction, ignore_index=0)


class BootstrappedCrossEntropy(torch.nn.Module):

	def __init__(self, min_K, loss_th, weight=None, ignore_index=-100):
		self.min_K = min_K
		self.loss_th = loss_th
		self.weight = weight
		self.ignore_index = ignore_index

	def _bootstrap_xentropy_single(self, inputs, target):
			n, c, h, w = inputs.size()
			inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
			target = target.view(-1)
			loss = F.cross_entropy(inputs, target, weight=self.weight, reduction='none', ignore_index=self.ignore_index)
			sorted_loss, _ = torch.sort(loss, descending=True)
			
			if sorted_loss[self.min_K] > self.loss_th:
				loss = sorted_loss[sorted_loss > self.loss_th]
			else:
				loss = sorted_loss[:self.min_K]
			reduced_topk_loss = torch.mean(loss)

			return reduced_topk_loss

	def forward(self, inputs, target):
		batch_size = inputs.size()[0]
		loss = 0.0
		# Bootstrap from each image not entire batch
		for i in range(batch_size):
			loss += self._bootstrap_xentropy_single(torch.unsqueeze(inputs[i], 0), torch.unsqueeze(target[i], 0))
		return loss / float(batch_size)


class MultiScaleCrossEntropy(torch.nn.Module):

	def __init__(self, min_K, loss_th, weight=None, reduction='mean', scale_weight=[1.0, 0.4]):
		self.criterion1 = CrossEntropy(weight=weight, reduction=reduction)
		self.criterion2 = BootstrappedCrossEntropy(min_k=min_K, loss_th=loss_th, weight=weight)

	def forward(self, inputs, target):
		if not isinstance(inputs, tuple):
			return self.criterion1.forward(inputs, target)

		loss = 0.0
		for i, inp in enumerate(inputs):
			loss += self.scale_weight[i] * self.criterion2.forward(inp, target)

		return loss	


class FusionLoss(torch.nn.Module):

	def __init__(self, reduction='none', w_l1=1., w_l2=10., w_cos=0.1):
		super(FusionLoss, self).__init__()

		self.criterion1 = torch.nn.L1Loss(reduction=reduction)
		self.criterion2 = torch.nn.MSELoss(reduction=reduction)
		self.criterion3 = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean') # we use mean because the cosineembedingloss otherwise gives the wrong loss
		# when we take the sum and divide by the numbe of elements - at the core, the problem is that the cosineembeddingloss gives too many copies of the same output

		self.lambda1 = w_l1 if w_l1 is not None else 0.
		self.lambda2 = w_l2 if w_l2 is not None else 0.
		self.lambda3 = w_cos if w_cos is not None else 0.

	def forward(self, est, target):

		if est.shape[1] == 0:
		   return torch.ones_like(est).sum().clamp(min=1)

		x1 = torch.sign(est)
		x2 = torch.sign(target)

		x1 = x1.reshape([x1.shape[0], x1.shape[2], x1.shape[1]]) # we reshape to compute the cosine loss over the rays at a spatial location
		# if no reshaping is done, the loss is computed for a constant extraction depth over some spatial location.
		x2 = x2.reshape([x2.shape[0], x2.shape[2], x2.shape[1]])

		label = torch.ones_like(x1)

		l1 = self.criterion1.forward(est, target)
		l2 = self.criterion2.forward(est, target)
		l3 = self.criterion3.forward(x1, x2, label)

		normalization = torch.ones_like(l1).sum()

		l1 = l1.sum() / normalization
		l2 = l2.sum() / normalization

		l = self.lambda1*l1 + self.lambda2*l2 + self.lambda3*l3

		return l