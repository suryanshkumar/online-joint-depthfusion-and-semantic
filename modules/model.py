import torch
from torch import nn

class Block(nn.Module):
	def __init__(self, in_channels, out_channels):

		super(Block, self).__init__()

		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(),
			nn.Dropout2d(p=0.2),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(),
			nn.Dropout2d(p=0.2)
		)

	def forward(self, x):
		return self.block(x)


class Pred(nn.Module):
	def __init__(self, in_channels, out_channels, n_points=None):

		super(Pred, self).__init__()

		self.pred = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(),
			nn.Dropout2d(p=0.2),
			nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(),
			nn.Dropout2d(p=0.2)
		)
		if n_points is not None:
			self.pred = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
				nn.BatchNorm2d(out_channels),
				nn.LeakyReLU(),
				nn.Dropout2d(p=0.2),
				nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
				nn.LeakyReLU(),
				nn.Conv2d(out_channels, n_points, kernel_size=1, padding=0),
				nn.Tanh()
			)

	def forward(self, x):
		return self.pred(x)


class FusionNet_v1(nn.Module):
	def __init__(self, config):

		super(FusionNet, self).__init__()

		self.config = config
		self.scale = config.output_scale

		self.n_channels = 2 * config.n_points + 1 + int(config.use_semantics)
		self.n_points = config.n_points

		self.block = nn.ModuleList(
			[Block((i + 1) * self.n_channels, self.n_channels) 
			for i in range(4)])

		self.pred1 = Pred(5 * self.n_channels, 4 * self.n_channels)
		self.pred2 = Pred(4 * self.n_channels, 3 * self.n_channels)
		self.pred3 = Pred(3 * self.n_channels, 2 * self.n_channels)
		self.pred4 = Pred(2 * self.n_channels, 1 * self.n_channels, self.n_points)

	def forward(self, data):
		"""
		params: x = list of input tensors (tsdf_values, tsdf_weights, tsdf_frame, confidence, semantics)
		"""
		x = torch.cat([data['tsdf_values'], data['tsdf_weights'], data['tsdf_frame']], dim=1)
		if self.config.use_semantics:
			x = torch.cat([x, data['semantic_frame']], dim=1)

		x1 = self.block[0].forward(x)
		x1 = torch.cat([x, x1], dim=1)
		x2 = self.block[1].forward(x1)
		x2 = torch.cat([x1, x2], dim=1)
		x3 = self.block[2].forward(x2)
		x3 = torch.cat([x2, x3], dim=1)
		x4 = self.block[3].forward(x3)
		y  = torch.cat([x3, x4], dim=1)

		y = self.pred1(y)
		y = self.pred2(y)
		y = self.pred3(y)
		y = self.scale * self.pred4(y)

		return y


class VortexPooling(nn.Module):
	def __init__(self, in_chs, mid_chs, out_chs, feat_res):

		super(VortexPooling, self).__init__()
		
		rates = [1, 3, 9, 27]

		self.gave_pool = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_chs, out_chs, kernel_size=1, padding=0),
			nn.Upsample(size=feat_res, mode='bilinear', align_corners=True),
			nn.BatchNorm2d(num_features=out_chs)
		)

		self.pool1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
		self.pool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
		self.pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

		self.branches = nn.ModuleList([])
		for rate in rates:
			# branch 2-4
			branch = nn.Sequential(
				nn.Conv2d(in_chs, mid_chs, kernel_size=1),
				nn.BatchNorm2d(mid_chs),
				nn.ReLU(),
				nn.Conv2d(mid_chs, mid_chs, kernel_size=3, dilation=rate, padding=rate),
				nn.BatchNorm2d(mid_chs),
				nn.ReLU(),
				nn.Conv2d(mid_chs, mid_chs, kernel_size=3, dilation=rate, padding=rate),
				nn.BatchNorm2d(mid_chs),
				nn.ReLU(),
				nn.Conv2d(mid_chs, out_chs, kernel_size=1),
				nn.BatchNorm2d(out_chs),
				nn.ReLU(),
			)
			self.branches.append(branch)

		self.final = nn.Sequential(
			nn.Conv2d(5 * out_chs, out_chs, kernel_size=1, padding=0),
			nn.BatchNorm2d(num_features=out_chs),
			nn.Dropout2d(p=0.2, inplace=True)
		)

	def forward(self, x):

		x1 = self.gave_pool(x)
		x2 = self.branches[0](x)

		xp = self.pool1(x)
		x3 = self.branches[1](xp)
		
		xp = self.pool2(xp)
		x4 = self.branches[2](xp)
		
		xp = self.pool3(xp)
		x5 = self.branches[3](xp)

		out = torch.cat([x1, x2, x3, x4, x5], dim=1)

		out = self.final(out)

		return out


class FusionNet_v2(nn.Module):
	def __init__(self, config):

		super(FusionNet_v2, self).__init__()

		self.config = config
		self.scale = config.output_scale
		self.n_points = config.n_points
		self.resy = config.resy
		self.resx = config.resx

		self.n_channels = config.n_points * 2 + 1 + int(config.use_semantics)
		self.gf = config.growth_factor - 1

		pool_in = self.n_channels * (self.gf + 1)

		self.block = nn.ModuleList([
			Block((i + 1) * self.n_channels, self.n_channels) 
			for i in range(self.gf)
		])

		self.vortex = VortexPooling(pool_in, self.n_channels, pool_in, (self.resy, self.resx))
		self.vortex_final = VortexPooling(pool_in, self.n_channels, pool_in, (self.resy, self.resx))

		self.pred = nn.ModuleList([])
		for i in range(self.gf):
			points = self.n_points if i == (self.gf - 1) else None
			self.pred.append(Pred((self.gf + 1 - i) * self.n_channels, (self.gf - i) * self.n_channels, points))  
		self.pred = nn.Sequential(*self.pred)


	def forward_blocks(self, x, blocks):
		for block in blocks:
			x_old = x
			y = block.forward(x)
			x = torch.cat([x_old, y], dim=1)
		return x

	def forward(self, x):
		"""
		:param x: list of input tensors (tsdf_values, tsdf_weights, tsdf_frame, confidence, semantics)
		"""
		if self.config.use_semantics:
			x = torch.cat([x['tsdf_values'], x['tsdf_weights'], x['tsdf_frame'], x['semantic_frame']], dim=1)
		else:
			x = torch.cat([x['tsdf_values'], x['tsdf_weights'], x['tsdf_frame']], dim=1)

		y = self.forward_blocks(x, self.block)
		y = self.vortex(y)
		y = self.vortex_final(y)
		y = self.pred(y) * self.scale

		return y


class FusionNet_v3(nn.Module):
	def __init__(self, config):

		super(FusionNet_v3, self).__init__()

		self.config = config
		self.scale = config.output_scale
		self.n_points = config.n_points
		self.resy = config.resy
		self.resx = config.resx

		self.n_channels = config.n_points * 2 + 1
		self.gf = config.growth_factor - 1

		pool_in = self.n_channels * (self.gf + 1)
		heads = 1

		self.block0 = nn.ModuleList([
			Block((i + 1) * self.n_channels, self.n_channels) 
			for i in range(self.gf)
		])
		self.vortex0 = VortexPooling(pool_in, self.n_channels, pool_in, (self.resy, self.resx))

		if self.config.use_semantics:
			heads += 1
			self.block2 = nn.ModuleList([
				Block((i + 1) * self.n_channels, self.n_channels) 
				for i in range(self.gf)
			])
			self.vortex2 = VortexPooling(pool_in, self.n_channels, pool_in, (self.resy, self.resx))

		self.vortex3 = VortexPooling(heads * pool_in, self.n_channels, pool_in, (self.resy, self.resx))

		self.pred = nn.ModuleList([])
		for i in range(self.gf):
			points = self.n_points if i == (self.gf - 1) else None
			self.pred.append(Pred((self.gf + 1 - i) * self.n_channels, (self.gf - i) * self.n_channels, points))  
		self.pred = nn.Sequential(*self.pred)

	def forward_blocks(self, x, blocks):
		for block in blocks:
			x_old = x
			y = block.forward(x)
			x = torch.cat([x_old, y], dim=1)
		return x

	def forward(self, x):
		"""
		:param x: list of input tensors (tsdf_values, tsdf_weights, tsdf_frame, confidence, semantics)
		"""
		x_tsdf = torch.cat([x['tsdf_values'], x['tsdf_weights'], x['tsdf_frame']], dim=1)
		y = self.forward_blocks(x_tsdf, self.block0)
		y = self.vortex0(y)

		if self.config.use_semantics:
			x_sem = torch.cat([x['tsdf_values'], x['tsdf_weights'], x['semantic_frame']], dim=1)
			y1 = self.forward_blocks(x_sem, self.block2)
			y1 = self.vortex2(y1)

			y = torch.cat([y, y1], dim=1)

		y = self.vortex3(y)
		y = self.pred(y) * self.scale

		return y