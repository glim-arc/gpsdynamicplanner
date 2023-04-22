import numpy as np
import os
import os.path
import csv
import torch
import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from AE.CAE_obs import encode_obs
from AE.CAE_gps import encode_gps

def load_dataset(args, load_pickle):
	pickle_path = os.path.join(args.data_path, "planning_data.pickle")

	if load_pickle:
		with open(pickle_path, 'rb') as f:
			data = pickle.load(f)
			print("parsed bag loaded")
			return data

	encoded_obs = encode_obs(args)
	encoded_gps = encode_gps(args)

	seed = args.random_seed
	torch.manual_seed(seed)
	np.random.seed(seed)

	data = []
	output = []

	print("start loading traj data")
	### load env and datasets
	for i in range(args.total_env_num):
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
			start, goal = cur_start_goal
			xref0 = torch.from_numpy(start).reshape(1, -1, 1).type(torch.FloatTensor)
			# goal point
			xrefN = torch.from_numpy(goal).reshape(1, -1, 1)

			input = torch.cat([xref0, xrefN], dim = 0)
			start_goal_list_torch.append(input)

		traj_list = []
		for j in range(20):
			traj = np.load(directory + "/" + "traj_output_" + str(j), allow_pickle=True)
			traj = torch.from_numpy(traj)
			traj_list.append(traj)

		obs = encoded_obs[i]
		gps = encoded_gps[i]

		for j in range(traj_list):
			cur_start_goal = start_goal_list_torch[j]
			cur_traj = traj_list[j]

			cur_data = [gps, obs, cur_start_goal]
			cur_data = torch.cat(cur_data, dim = 0)

			data.append(cur_data)
			output.append(cur_traj)

	data = torch.stack(data, dim=0)
	output = torch.stack(output, dim=0)

	loaded_data = TensorDataset(data, output)

	data = DataLoader(loaded_data, batch_size=args.batchsize, shuffle=True)

	with open(pickle_path, 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

	return data

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
	parser.add_argument('--seed', type=int, default=10)
	args = parser.parse_args()
	main(args)