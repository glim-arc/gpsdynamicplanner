import torch
import torch.nn as nn

# Model-Path Generator
class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(5120, 2560),nn.PReLU(),nn.Dropout(),
		nn.BatchNorm1d(2560),
		nn.Linear(2560, 1280), nn.PReLU(), nn.Dropout(),
		nn.BatchNorm1d(1280),
		nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(),
		nn.BatchNorm1d(1024),
		nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(),
		nn.BatchNorm1d(896),
		nn.Linear(896, 768),nn.PReLU(),nn.Dropout(),
		nn.BatchNorm1d(768),
		nn.Linear(768, 512),nn.PReLU(),nn.Dropout(),
		nn.BatchNorm1d(512),
		nn.Linear(512, 384),nn.PReLU(),nn.Dropout(),
		nn.BatchNorm1d(384),
		nn.Linear(384, 256),nn.PReLU(), nn.Dropout(),
		nn.BatchNorm1d(256),
		nn.Linear(256, 256),nn.PReLU(), nn.Dropout(),
		nn.BatchNorm1d(256),
		nn.Linear(256, 128),nn.PReLU(), nn.Dropout(),
		nn.BatchNorm1d(128),
		nn.Linear(128, 64),nn.PReLU(), nn.Dropout(),
		nn.BatchNorm1d(64),
		nn.Linear(64, 32),nn.PReLU(),
		nn.BatchNorm1d(32),
		nn.Linear(32, output_size))
        
	def forward(self, x):
		out = self.fc(x)
		return out

 
