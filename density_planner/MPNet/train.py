import os
import torch
from torch import nn
import sys

sys.path.append(os.getcwd())
from gps_MPNet_hyperparams import parse_args
from mpnet_data.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np
import hyperparams
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from MPNet.data_loader import load_dataset, load_dataset_recur
from MPNet.model import MLP
from mpnet_data.plot import planner_plot


def save_generated_traj(args):
    load_pickle = args.load
    load_pickle = True

    loaded_data, val_loaded_data = load_dataset_recur(args, load_pickle=load_pickle)
    shuffle = False

    dataloader = DataLoader(loaded_data, batch_size=1, shuffle=shuffle)
    val_dataloader = DataLoader(val_loaded_data, batch_size=1, shuffle=shuffle)
    print("data loaded")

    seed = args.seed
    torch.manual_seed(seed)

    planner = MLP(args.input_size, args.output_size, args.up_size, args.lstm_hidden_size, args.lstm_layer_size).eval()

    device = "cpu"

    planner.to(device)

    print(device)

    model = torch.load(os.path.join(args.model_path, args.planner_model_name))
    planner.load_state_dict(model)

    avg_loss_list = []

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print("training data accuracy")
    mse_loss = nn.MSELoss()
    training_output_up_list = []
    avg_loss = []

    for batch_idx, batch in enumerate(dataloader):
        cur_batch = batch[0].to(device)
        cur_batch_output = batch[1].to(device)

        planner.zero_grad()

        cur_batch = batch[0].to(device)
        cur_batch_output = batch[1].to(device)
        cur_output = []

        for i in range(args.timesteps):
            # ===================forward=====================
            up = torch.zeros((cur_batch.shape[0], i + 1, args.up_size)).to(device)  # initial up = 0
            if i > 0:
                up[:, 1:] = cur_batch_output[:, :i]
            output = planner(cur_batch, up, device)
            loss = mse_loss(output, cur_batch_output[:, i])
            avg_loss.append(loss.cpu().detach().numpy())
            cur_output.append(output.cpu().detach().numpy())
        npoutput = np.array(cur_output)[:,0,:].T
        training_output_up_list.append(npoutput)

    print("--Training data average loss:")
    avg_loss = sum(avg_loss) / len(avg_loss)
    print(avg_loss)

    training_output_up_list = np.array(training_output_up_list)
    training_output_path = os.path.join(args.data_path, 'training_output_up_list.npy')
    np.save(training_output_path, training_output_up_list)

    print("validation data accuracy")
    avg_loss = []
    validation_output_up_list = []

    for batch_idx, batch in enumerate(val_dataloader):
        cur_batch = batch[0].to(device)
        cur_batch_output = batch[1].to(device)
        cur_output = []

        for i in range(args.timesteps):
            # ===================forward=====================
            up = torch.zeros((cur_batch.shape[0], i + 1, args.up_size)).to(device)  # initial up = 0
            if i > 0:
                up[:, 1:] = cur_batch_output[:, :i]
            output = planner(cur_batch, up, device)
            loss = mse_loss(output, cur_batch_output[:, i])
            avg_loss.append(loss.cpu().detach().numpy())
            cur_output.append(output.cpu().detach().numpy())

        npoutput = np.array(cur_output).T
        validation_output_up_list.append(npoutput)

    print("--Validation average loss:")
    avg_loss = sum(avg_loss) / len(avg_loss)
    print(avg_loss)

    validation_output_up_list = np.array(validation_output_up_list)
    validation_output_path = os.path.join(args.data_path, 'validation_output_up_list.npy')
    np.save(validation_output_path, validation_output_up_list)

    print(len(training_output_up_list))
    print(len(validation_output_up_list))

    print("start saving traj data")
    ### load env and datasets
    idx = 0
    for i in range(args.total_env_num):
        print("load env num ", i)
        directory = "./mpnet_data/env/" + str(i)

        if idx < len(training_output_up_list) - 1:
            for j in range(20):
                cur_train_traj_path = os.path.join(directory, "traj_output_" + str(j) + "_gen.npy")
                # print(idx)
                print(training_output_up_list[idx])
                np.save(cur_train_traj_path, training_output_up_list[idx])
                # os.remove(cur_train_traj_path)
                idx += 1
        else:
            for j in range(20):
                cur_val_traj_path = os.path.join(directory, "traj_output_" + str(j) + "_gen.npy")
                np.save(cur_val_traj_path, validation_output_up_list[idx - len(training_output_up_list)])
                # os.remove(cur_val_traj_path)
                idx += 1

    print("end")


def main(args):
    load_pickle = args.load
    load_pickle = True

    loaded_data, val_loaded_data = load_dataset_recur(args, load_pickle=load_pickle)

    dataloader = DataLoader(loaded_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_loaded_data, batch_size=args.batch_size, shuffle=True)

    print("data loaded")

    torch.manual_seed(args.seed)

    planner = MLP(args.input_size, args.output_size, args.up_size, args.lstm_hidden_size, args.lstm_layer_size)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    planner.to(device)
    print(device)

    optimizer = torch.optim.Adam(planner.parameters(), lr=args.learning_rate)
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
            cur_batch_output = batch[1].to(device)
            for i in range(args.timesteps):
                # ===================forward=====================
                up = torch.zeros((cur_batch.shape[0],i +1, args.up_size)).to(device) #initial up = 0
                if i > 0:
                    up[:,1:] = cur_batch_output[:,:i]
                output = planner(cur_batch, up, device)
                loss = mse_loss(output, cur_batch_output[:,i])
                # ===================backward====================
                loss.backward(retain_graph=True)
                optimizer.step()
                loss = loss.cpu().detach().numpy()
                cur_batch_loss.append(loss)

        print("--average loss:")
        avg_loss_list.append(sum(cur_batch_loss) / len(cur_batch_loss))
        print(avg_loss_list[-1])

    torch.save(planner.state_dict(), os.path.join(args.model_path, 'planner.model'))
    np.save(os.path.join(args.model_path, 'planner_avg_loss_list.npy'), avg_loss_list)

    # model = torch.load(os.path.join(args.model_path, 'planner.model'), map_location=torch.device('cpu'))
    # planner.load_state_dict(model)

    print("validation starts")
    avg_loss = []
    mse_loss = nn.MSELoss()

    for batch_idx, batch in enumerate(val_dataloader):
        planner.zero_grad()

        cur_batch = batch[0].to(device)
        cur_batch_output = batch[1].to(device)

        for i in range(args.timesteps):
            # ===================forward=====================
            up = torch.zeros((cur_batch.shape[0], i + 1, args.up_size)).to(device)  # initial up = 0
            if i > 0:
                up[:, 1:] = cur_batch_output[:, :i]
            output = planner(cur_batch, up, device)
            loss = mse_loss(output, cur_batch_output[:, i])
            loss = loss.cpu().detach().numpy()
            avg_loss.append(loss)

    print("--Validation average loss:")
    avg_loss = sum(avg_loss) / len(avg_loss)
    avg_loss = avg_loss
    print(avg_loss)

    val_loss = np.array(avg_loss)

    np.save(os.path.join(args.data_path, 'planner_val_loss.npy'), val_loss)
    planner_plot(args.model_path, val_loss)
    planner_plot(args.model_path, np.array(avg_loss_list[-1]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    save_generated_traj(args)