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
from MPNet.AE.CAE_obs import encode_obs
from MPNet.AE.CAE_gps import encode_gps
from gps_MPNet_hyperparams import parse_args

def load_dataset(args, load_pickle, shuffle=True):
    if shuffle == True:
        pickle_path = os.path.join(args.data_path, "planning_data.pickle")
        val_pickle_path = os.path.join(args.data_path, "planning_val_data.pickle")
    else:
        pickle_path = os.path.join(args.data_path, "planning_data_unshuffle.pickle")
        val_pickle_path = os.path.join(args.data_path, "planning_val_data_unshuffle.pickle")

    if load_pickle:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            print("parsed data loaded")

        with open(val_pickle_path, 'rb') as f:
            val_data = pickle.load(f)
            print("parsed val data loaded")

        return data, val_data

    encoded_obs = encode_obs(args)
    #encoded_gps = encode_gps(args)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = []
    output = []

    print("start loading traj data")
    ### load env and datasets
    for i in range(args.total_env_num):
        print("load env num ", i)
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
            traj = np.load(directory + "/" + "traj_output_" + str(j) + ".npy", allow_pickle=True)
            traj = torch.from_numpy(traj[0])
            traj_list.append(traj)

        obs = encoded_obs[i]
        #gps = encoded_gps[i]

        for j in range(len(traj_list)):
            cur_start_goal = start_goal_list_torch[j]
            cur_traj = traj_list[j][0].flatten()

            cur_data = torch.zeros(args.input_size)
            cur_data[0:len(gps)] = gps
            cur_data[len(gps):len(gps) + len(obs)] = obs
            cur_data[len(gps) + len(obs):len(gps) + len(obs) + len(cur_start_goal)] = cur_start_goal

            data.append(cur_data)
            output.append(cur_traj)

    data = torch.stack(data, dim=0)
    output = torch.stack(output, dim=0)

    print(data.shape)

    input_data = TensorDataset(data[:-args.validation_env_num * 20], output[:-args.validation_env_num * 20])

    with open(pickle_path, 'wb') as f:
        pickle.dump(input_data, f, pickle.HIGHEST_PROTOCOL)

    val_data = TensorDataset(data[-args.validation_env_num * 20:], output[-args.validation_env_num * 20:])

    with open(val_pickle_path, 'wb') as f:
        pickle.dump(val_data, f, pickle.HIGHEST_PROTOCOL)

    print("end")

    return input_data, val_data

def load_dataset_recur(args, load_pickle, shuffle=True):
    pickle_path = os.path.join(args.data_path, "planning_data_recur.pickle")
    val_pickle_path = os.path.join(args.data_path, "planning_val_data_recur.pickle")

    if load_pickle:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            print("parsed data loaded")

        with open(val_pickle_path, 'rb') as f:
            val_data = pickle.load(f)
            print("parsed val data loaded")

        return data, val_data

    encoded_obs = encode_obs(args)
    #encoded_gps = encode_gps(args)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = []
    output = []

    print("start loading traj data")
    ### load env and datasets
    for i in range(args.total_env_num):
        print("load env num ", i)
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
            cur_start_goal = cur_start_goal[0][:-1].tolist() + cur_start_goal[1][:-1].tolist()
            input = torch.from_numpy(np.array(cur_start_goal))
            start_goal_list_torch.append(input)

        traj_list = []
        for j in range(20):
            traj = np.load(directory + "/" + "traj_output_" + str(j) + ".npy", allow_pickle=True)
            traj = torch.from_numpy(traj[0])
            traj_list.append(traj)

        obs = encoded_obs[i]
        #gps = encoded_gps[i]

        for j in range(len(traj_list)):
            cur_start_goal = start_goal_list_torch[j]
            cur_traj = traj_list[j][0].T

            cur_data = torch.zeros(args.input_size)
            cur_data[0:len(obs)] = obs
            cur_data[len(obs):len(obs) + len(cur_start_goal)] = cur_start_goal

            data.append(cur_data)
            output.append(cur_traj)

        # for j in range(len(traj_list)):
        #     curup = traj_list[j][:][0].flatten()
        #     upsize = len(curup)
        #     for k in range(traj_list.shape[2]):
        #         goal = goal_list[j]
        #
        #         cur_data = torch.zeros(args.input_size)
        #         idx = args.obs_latent_size
        #         cur_data[0:idx] = obs
        #         #cur_data[len(gps):len(gps) + len(obs)] = obs
        #         cur_data[idx:idx+len(goal)] = goal
        #         idx += len(goal)
        #         cur_data[idx:idx + upsize] = curup
        #
        #         data.append(cur_data)
        #         curup = traj_list[j][:][k].flatten() #next up
        #         output.append(curup)

    data = torch.stack(data, dim=0)
    output = torch.stack(output, dim=0)

    print(data.shape)

    input_data = TensorDataset(data[:-args.validation_env_num * 20], output[:-args.validation_env_num * 20])

    with open(pickle_path, 'wb') as f:
        pickle.dump(input_data, f, pickle.HIGHEST_PROTOCOL)

    val_data = TensorDataset(data[-args.validation_env_num * 20:], output[-args.validation_env_num * 20:])

    with open(val_pickle_path, 'wb') as f:
        pickle.dump(val_data, f, pickle.HIGHEST_PROTOCOL)

    print("end")

    return input_data, val_data


if __name__ == '__main__':
    args = parse_args()
    load_dataset_recur(args, False)
    #load_dataset(args, False)