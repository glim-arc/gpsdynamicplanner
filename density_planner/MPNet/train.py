import argparse
import os
import torch
from torch import nn
from mpnet_data.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np
import hyperparams
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpnet_data.plot import plot
from MPNet.data_loader import load_dataset

def main(args):
	load_dataset(args, False)

	seed = 10
	torch.manual_seed(seed)

	encoder = Encoder()
	decoder = Decoder()

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
		encoder.to(device)
		decoder.to(device)

	params = list(encoder.parameters()) + list(decoder.parameters())
	optimizer = torch.optim.Adam(params)
	avg_loss_list = []

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	env_list = torch.ones((args.total_env_num - 100, 1, 241, 401)).to(device)

	print("load env ")
	for env_num in range(args.total_env_num - 100):
		# load environment
		env_grid = load_env(env_num).permute(2, 0, 1).unsqueeze(dim=0).to(device)
		env_list[env_num] = env_grid

	dataset = TensorDataset(env_list)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
	print("env loaded")

	print("training starts")
	ctr = 0
	for epoch in range(args.num_epochs):
		print("epoch" + str(epoch))
		avg_loss = 0

		for batch_idx, batch in enumerate(dataloader):
			optimizer.zero_grad()
			decoder.zero_grad()
			encoder.zero_grad()
			cur_grid_batch = batch[0].to(device)

			# ===================forward=====================
			latent_space = encoder(cur_grid_batch)
			output = decoder(latent_space)
			keys = encoder.state_dict().keys()
			W = encoder.state_dict()[
				'encoder.15.weight']  # regularize or contracting last layer of encoder. Print keys to displace the layers name.
			loss = loss_function(W, cur_grid_batch, output, latent_space)
			avg_loss = avg_loss + loss.data
			# ===================backward====================
			loss.backward()
			optimizer.step()

		if ctr == 0:
			ctr = 1

		print("--average loss:")
		avg_loss = avg_loss.cpu().numpy() / args.batch_size / ctr
		print(avg_loss)
		avg_loss_list.append(avg_loss)

	print("validation starts")
	env_list = torch.zeros((args.total_env_num - 100, 1, 1, 241, 401)).to(device)

	idx = 0
	for env_num in range(args.total_env_num - 100, args.total_env_num):
		# load environment
		print("load env " + str(env_num))
		env_grid = load_env(env_num).permute(2, 0, 1).unsqueeze(dim=0).to(device)
		env_list[idx] = env_grid
		idx += 1
	print("Env loaded")

	avg_loss = 0
	start_val_env = args.total_env_num - 100
	for i in range(start_val_env, args.total_env_num):
		optimizer.zero_grad()
		decoder.zero_grad()
		encoder.zero_grad()

		# ===================forward=====================
		cur_grid_batch = env_list[i - start_val_env]
		print(cur_grid_batch.shape)
		latent_space = encoder(cur_grid_batch)
		output = decoder(latent_space)
		keys = encoder.state_dict().keys()
		W = encoder.state_dict()[
			'encoder.15.weight']  # regularize or contracting last layer of encoder. Print keys to displace the layers name.
		loss = loss_function(W, cur_grid_batch, output, latent_space)
		avg_loss = avg_loss + loss.data

	print("--Validation average loss:")
	avg_loss = avg_loss.cpu().numpy() / 100
	print(avg_loss)

	avg_loss_list = np.array(avg_loss_list)
	val_loss = np.array(avg_loss)

	torch.save(encoder.state_dict(), os.path.join(args.model_path, 'cae_obs_encoder.model'))
	torch.save(decoder.state_dict(), os.path.join(args.model_path, 'cae_obs_decoder.model'))
	np.save(os.path.join(args.model_path, 'obs_avg_loss_list.npy'), avg_loss_list)
	np.save(os.path.join(args.model_path, 'obs_val_loss.npy'), val_loss)

	plot(args.model_path)

if __name__ == '__main__':
	start_size = 5
	goal_size = 5
	obs_latent_size = 28
	gps_latent_size = 2048
	input_size = gps_latent_size + obs_latent_size + start_size + goal_size  # gps space + obs space + start + goal
	upsize = 10
	output_size = upsize

	# mpnet training data generation
	parser.add_argument('--gps_training_data_generation', type=bool, default=True)
	parser.add_argument('--gps_training_env_path', type=str, default="mpnet_data/env/")
	parser.add_argument('--total_env_num', type=int, default=500)
	parser.add_argument('--training_env_num', type=int, default=500)
	parser.add_argument('--training_traj_num', type=int, default=10)
	parser.add_argument('--model_path', type=str, default='./mpnet_data/models/', help='path for saving trained models')
	parser.add_argument('--data_path', type=str, default='./mpnet_data/', help='path for saving data')
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	parser.add_argument('--start_size', type=int, default=start_size)
	parser.add_argument('--goal_size', type=int, default=goal_size)
	parser.add_argument('--obs_latent_size', type=int, default=obs_latent_size)
	parser.add_argument('--gps_latent_size', type=int, default=gps_latent_size)
	parser.add_argument('--input_size', type=int, default=input_size)
	parser.add_argument('--output_size', type=int, default=output_size)
	args = parser.parse_args()
	main(args)

