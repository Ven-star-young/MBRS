import torch
import torch.nn as nn
import torch.nn.functional as F

from network.blocks.SENet import BasicBlock, BottleneckBlock, BottleneckBlock

from network.blocks.SENet import BasicBlock

class ChannelSelfAttention(nn.Module):
	
    def __init__(self, channels, r=8, heads=4):
        super(ChannelSelfAttention, self).__init__()
        self.heads = heads
        self.inter_channels = channels // r
        self.head_dim = self.inter_channels // heads
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.to_qkv = nn.Linear(channels, self.inter_channels * 2 + channels, bias=False)
        self.scale = self.head_dim ** -0.5
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()
		
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x).view(b, c)
        qkv = self.to_qkv(y)
		
		#TODO 1
        q_raw, k_raw, v_raw = qkv[:, :self.inter_channels], qkv[:, self.inter_channels:self.inter_channels * 2], qkv[:, self.inter_channels * 2:self.inter_channels * 2 + c]
		#TODO 1 END
		
        q = q_raw.view(b, self.heads, 1, self.head_dim)
        k = k_raw.view(b, self.heads, 1, self.head_dim)
        v = v_raw.view(b, self.heads, 1, -1)
		
        #TODO 2
        attn = (q * k).sum(dim=-1) * self.scale
        attn = F.softmax(attn, dim=-1)
		#TODO 2 END
		
        out = torch.matmul(attn, v).view(b, c)
		
        #TODO 3
        scale = self.fc(out)
        scale = self.sigmoid(scale).view(b, c, 1, 1)
        #TODO 3 END
		
        return x * scale

class selfBasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate):
		super(selfBasicBlock, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
					  stride=drop_rate, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.se = ChannelSelfAttention(out_channels, r, heads=4)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.se.forward(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x


class selfBottleneckBlock(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate):
		super(selfBottleneckBlock, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
					  stride=drop_rate, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.se = ChannelSelfAttention(out_channels, r, heads=4)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.se.forward(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x


class selfSENet(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
		super(selfSENet, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(out_channels, out_channels, r, drop_rate)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class selfSENet_decoder(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
		super(selfSENet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1)
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate)
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
