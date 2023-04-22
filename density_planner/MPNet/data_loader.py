import numpy as np
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
import os.path
import csv
import torch
import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse
from AE.CAE_obs import encode_obs
from AE.CAE_gps import encode_gps


def load_dataset(args, load_pickle):
	pickle_path = os.path.join(args.data_path, "planning_data.pickle")
	val_pickle_path = os.path.join(args.data_path, "planning_val_data.pickle")

	if load_pickle:
		with open(pickle_path, 'rb') as f:
			data = pickle.load(f)
			print("parsed data loaded")
   
		with open(val_pickle_path, 'rb') as f:
			val_data = pickle.load(f)
			print("parsed val data loaded")
		
		return data, val_pickle_path

	encoded_obs = encode_obs(args)
	encoded_gps = encode_gps(args)

	seed = args.seed
	torch.manual_seed(seed)
	np.random.seed(seed)

	data = []
	output = []

	print("start loading traj data")
	### load env and datasets
	for i in range(args.total_env_num):
		print("load env num ",i)
		directory = "./mpnet_data/env/" + str(i)

		env_file = ""
		start_goal_file = ""

		files = os.listdir(directory)

		for file in files:
			if file[-3:] == "npy":
				if file.find("start") != -1:
					start_goal_file = file
				elif file.find("env_" + str(i) + ".npy") != -1:
					env_file = file

		start_goal_list = np.load(directory + "/" + start_goal_file, allow_pickle=True)
		start_goal_list_torch = []

		for j, cur_start_goal in enumerate(start_goal_list):
			cur_start_goal = cur_start_goal[0].tolist() + cur_start_goal[1].tolist()
			input = torch.from_numpy(np.array(cur_start_goal))
			start_goal_list_torch.append(input)

		traj_list = []
		for j in range(20):
			traj = np.load(directory + "/" + "traj_output_" + str(j)+".npy", allow_pickle=True)
			traj = torch.from_numpy(traj[0])
			traj_list.append(traj)

		obs = encoded_obs[i]
		gps = encoded_gps[i]

		for j in range(len(traj_list)):
			cur_start_goal = start_goal_list_torch[j]
			cur_traj = traj_list[j][0][0].flatten()

			cur_data = torch.zeros(args.input_size)
			cur_data[0:len(gps)] = gps
			cur_data[len(gps):len(gps)+len(obs)] = obs
			cur_data[len(gps)+len(obs):len(gps)+len(obs)+len(cur_start_goal)] = cur_start_goal

			data.append(cur_data)
			output.append(cur_traj)

	data = torch.stack(data, dim=0)
	output = torch.stack(output, dim=0)

	loaded_data = TensorDataset(data[:-100], output[:-100])

	input_data = DataLoader(loaded_data, batch_size=args.batch_size, shuffle=True)

	with open(pickle_path, 'wb') as f:
		pickle.dump(input_data, f, pickle.HIGHEST_PROTOCOL)

	loaded_data = TensorDataset(data[-100:], output[-100:])

	val_data = DataLoader(loaded_data, batch_size=args.batch_size, shuffle=True)

	with open(val_pickle_path, 'wb') as f:
		pickle.dump(val_data, f, pickle.HIGHEST_PROTOCOL)
  
	print("end")

	return input_data, val_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	start_size = 5
	goal_size = 5
	obs_latent_size = 28
	gps_latent_size = 2048
	input_size = gps_latent_size + obs_latent_size + start_size + goal_size  # gps space + obs space + start + goal
	upsize = 10
	output_size = upsize
	load = True

	# mpnet training data generation
	parser.add_argument('--load', type=bool, default=load)
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
	parser.add_argument('--seed', type=int, default=10)
	args = parser.parse_args()
	load_dataset(args, False)