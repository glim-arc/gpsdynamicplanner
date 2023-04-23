import argparse
import os
import torch
from torch import nn
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from mpnet_data.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np
import hyperparams
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from MPNet.data_loader import load_dataset
from model import MLP
from mpnet_data.plot import planner_plot

def main(args):
	dataloader, val_dataloader = load_dataset(args, True)
	print("data loaded")

	seed = 10
	torch.manual_seed(seed)

	planner = MLP(args.input_size, args.output_size)

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
		planner.to(device)
	
	print(device)

	optimizer = torch.optim.Adam(planner.parameters(), lr = 0.001, weight_decay = 0.9)
	avg_loss_list = []

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	print("training starts")

	avg_loss = []
	mse_loss = nn.MSELoss()

	for epoch in range(args.num_epochs):
		print("epoch" + str(epoch))
		cur_batch_loss = []
		for batch_idx, batch in enumerate(dataloader):
			optimizer.zero_grad()
			planner.zero_grad()

			cur_batch = batch[0].to(device)
			cur_batch_output =  batch[1].to(device)

			# ===================forward=====================
			output = planner(cur_batch)

			loss = mse_loss(output, cur_batch_output)
			print(loss)
			cur_batch_loss.append(loss.cpu().detach().numpy())
			# ===================backward====================
			loss.backward()
			optimizer.step()

		print("--average loss:")
		avg_loss.append(sum(cur_batch_loss) / len(cur_batch_loss))
		print(avg_loss[-1])
		avg_loss_list.append(avg_loss)

	torch.save(planner.state_dict(), os.path.join(args.data_path, 'planner.model'))
	np.save(os.path.join(data_path, 'planner_avg_loss_list.npy'), avg_loss_list)

	print("validation starts")
	avg_loss = []
	for batch_idx, batch in enumerate(val_dataloader):
		batch = dataloader[batch_idx]

		optimizer.zero_grad()
		planner.zero_grad()

		cur_batch = batch[0].to(device)
		cur_batch_output = batch[1].to(device)

		# ===================forward=====================
		output = planner(cur_batch)

		loss = mse_loss(output, cur_batch_output)
		avg_loss.append(loss.cpu().detach().numpy())
		print(batch_idx, " loss :", loss)

	print("--Validation average loss:")
	avg_loss = sum(avg_loss) / len(avg_loss)
	print(avg_loss[-1])

	val_loss = np.array(avg_loss)

	np.save(os.path.join(args.model_path, 'planner_val_loss.npy'), val_loss)
	planner_plot(os.path.join(os.getcwd(), "data"))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	start_size = 5
	goal_size = 5
	obs_latent_size = 28
	gps_latent_size = 2048
	input_size = gps_latent_size + obs_latent_size + start_size + goal_size  # gps space + obs space + start + goal
	upsize = 10
	output_size = upsize
	load = False
	num_epochs = 50
  
	# mpnet training data generation
	parser.add_argument('--load', type=bool, default=load)
	parser.add_argument('--gps_training_data_generation', type=bool, default=True)
	parser.add_argument('--gps_training_env_path', type=str, default="mpnet_data/env/")
	parser.add_argument('--num_epochs', type=int, default=num_epochs)
	parser.add_argument('--total_env_num', type=int, default=500)
	parser.add_argument('--training_env_num', type=int, default=500)
	parser.add_argument('--training_traj_num', type=int, default=10)
	parser.add_argument('--validation_env_num', type=int, default=100)
	parser.add_argument('--model_path', type=str, default='./mpnet_data/models/', help='path for saving trained models')
	parser.add_argument('--data_path', type=str, default='./mpnet_data/', help='path for saving data')
	parser.add_argument('--batch_size', type=int, default=5000)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	parser.add_argument('--start_size', type=int, default=start_size)
	parser.add_argument('--goal_size', type=int, default=goal_size)
	parser.add_argument('--obs_latent_size', type=int, default=obs_latent_size)
	parser.add_argument('--gps_latent_size', type=int, default=gps_latent_size)
	parser.add_argument('--input_size', type=int, default=input_size)
	parser.add_argument('--output_size', type=int, default=output_size)
	parser.add_argument('--seed', type=int, default=10)
	args = parser.parse_args()
	main(args)