import argparse
import os
import torch
from torch import nn
from gps_planning.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np
import hyperparams
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from mpnet_data.plot import plot


# https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # convolutional layer
        self.encoder = nn.Sequential(
            # convolutional layer
            nn.Conv3d(1, 8, 2, stride=2, padding=1),
            nn.PReLU(),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 16, 2, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(),
            nn.Conv3d(16, 32, 2, stride=2, padding=0),
            nn.PReLU(),

            nn.Flatten(start_dim=1),

            # linear layer
            nn.Linear(32 * 2 * 15 * 25, 10240), nn.PReLU(),# nn.BatchNorm1d(10240),
            nn.Linear(10240, 8192), nn.PReLU(),# nn.BatchNorm1d(8192),
            nn.Linear(8192, 4096), nn.PReLU(), nn.Linear(4096, 2048)
        )

    # #convolutional layer #1*240*400
    # self.conv = self.encoder_cnn = nn.Sequential(
    # 	#1*101*240*400
    # 	nn.Conv3d(1, 8, 2, stride=2, padding=1),
    # 	nn.PReLU(),
    # 	nn.BatchNorm3d(8),
    # 	nn.Conv3d(8, 16, 2, stride=2, padding=1),
    # 	nn.BatchNorm3d(16),
    # 	nn.PReLU(),
    # 	nn.Conv3d(16, 32, 2, stride=2, padding=0),
    # 	nn.PReLU()
    # ) #32*1*8*13
    #
    # #flatten
    # self.flatten = nn.Flatten(start_dim=1)
    #
    # #linear
    # self.lin = nn.Sequential(nn.Linear(32*2*15*25, 10240),nn.PReLU(),nn.Linear(10240, 4096),nn.PReLU(),nn.Linear(4096, 2048),nn.PReLU(),
    # 						 nn.Linear(2048, 1024),nn.PReLU(),nn.Linear(1024, 512))

    def forward(self, x):
        x = self.encoder(x)
        # x = self.conv(x)
        # x = self.flatten(x)
        # x = self.lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            # linear layer
            nn.Linear(2048, 4096), nn.PReLU(),  nn.Linear(4096, 8192),  nn.PReLU(), #nn.BatchNorm1d(8192),
            nn.Linear(8192, 10240), nn.PReLU(), #nn.BatchNorm1d(10240),
            nn.Linear(10240, 32 * 2 * 15 * 25),

            # flatten
            nn.Unflatten(dim=1, unflattened_size=(32, 2, 15, 25)),

            # de-convolutional layer
            nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.PReLU(),
            nn.ConvTranspose3d(16, 8, 2, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm3d(8),
            nn.PReLU(),
            nn.ConvTranspose3d(8, 1, 2, stride=2,
                               padding=1, output_padding=0),
            nn.PReLU(),
            nn.Upsample(size=(10, 120, 200))
        )

    def forward(self, x):
        x = self.decoder(x)
        # x = self.lin(x)
        # x = self.unflatten(x)
        # x = self.deconv(x)
        return x


mse_loss = nn.MSELoss()
lam = 1e-3


def loss_function(W, x, recons_x, h):
    mse = mse_loss(recons_x, x)
    """
	W is shape of N_hidden x N. So, we do not need to transpose it as opposed to http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
	"""
    # dh = h*(1-h) # N_batch x N_hidden
    # contractive_loss = torch.sum(W**2, dim=1).sum().mul_(lam)
    return mse


def load_env(i):
    ### load hyperparameters
    args = hyperparams.parse_args()

    directory = "./mpnet_data/env/" + str(i)

    env_file = ""
    start_goal_file = ""

    files = os.listdir(directory)

    for file in files:
        if file[-3:] == "npy":
            if file.find("env_" + str(i) + ".npy") != -1:
                env_file = file

    obslist = np.load(directory + "/" + env_file, allow_pickle=True)

    objects = []
    for i in range(len(obslist)):
        obs, gps_dynamics = obslist[i]
        map = DynamicObstacle(args, name="gpsmaps%d" % i, coord=obs, velocity_x=gps_dynamics[1],
                              velocity_y=gps_dynamics[2],
                              gps_growthrate=gps_dynamics[0], isgps=True)
        objects.append(map)

    environment = Environment(objects, args)
    environment.forward_occupancy(step_size=100)

    return environment.gps_grid


def main(args):
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

    discrete_time = 10
    env_list = torch.ones((args.total_env_num - args.validation_env_num, 1, discrete_time, 120, 200)).to("cpu")

    print("load env ")
    # for env_num in range(args.total_env_num - args.validation_env_num):
    # 	print("env ", env_num)
    # 	# load environment
    # 	env_grid = load_env(env_num).permute(2, 0, 1)
    # 	env_grid = TF.resize(env_grid, (120, 200)).to("cpu")

    # 	for i in range(discrete_time):
    # 		env_list[env_num][0][i] = env_grid[i * discrete_time]

    # save in np
    # np.save(os.path.join(args.model_path,'gps_env_list.npy'), env_list.numpy())

    env_list = np.load(os.path.join(args.model_path, 'gps_env_list.npy'))
    env_list = torch.from_numpy(env_list).to("cpu")

    dataset = TensorDataset(env_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("env loaded")

    print("training starts")
    for epoch in range(args.num_epochs):
        print("epoch" + str(epoch))
        avg_loss = 0
        ctr = 0
        for batch_idx, batch in enumerate(dataloader):
            ctr += 1
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
            ctr =1
            
        print("--average loss:", ctr)
        avg_loss = avg_loss.cpu().numpy() / args.batch_size / ctr
        print(avg_loss)
        avg_loss_list.append(avg_loss)

    print("validation starts")
    env_list = torch.ones((args.validation_env_num, 1, discrete_time, 120, 200)).to(device)

    # idx = 0
    # for env_num in range(args.total_env_num - args.validation_env_num, args.total_env_num):
    # 	print("env ", env_num)
    # 	# load environment
    # 	env_grid = load_env(env_num).permute(2, 0, 1)
    # 	env_grid = TF.resize(env_grid, (120, 200)).to(device)

    # 	for i in range(discrete_time):
    # 		env_list[idx][0][i] = env_grid[i * discrete_time]

    # 	idx += 1
    # print("Env loaded")

    # np.save(os.path.join(args.model_path, 'gps_env_list_val.npy'), env_list.cpu().numpy())
    env_list = np.load(os.path.join(args.model_path, 'gps_env_list_val.npy'))
    env_list = torch.from_numpy(env_list).to("cuda")

    avg_loss = 0
    start_val_env = args.total_env_num - 100

    for i in range(start_val_env, args.total_env_num):
        optimizer.zero_grad()
        decoder.zero_grad()
        encoder.zero_grad()

        # ===================forward=====================
        cur_grid_batch = env_list[i - start_val_env].unsqueeze(dim=0)
        latent_space = encoder(cur_grid_batch)
        output = decoder(latent_space)
        keys = encoder.state_dict().keys()
        W = encoder.state_dict()[
            'encoder.15.weight']  # regularize or contracting last layer of encoder. Print keys to displace the layers name.
        loss = loss_function(W, cur_grid_batch, output, latent_space)
        avg_loss = avg_loss + loss.data

    print("--Validation average loss:")
    print(avg_loss / 100)

    avg_loss_list = np.array(avg_loss_list)
    val_loss = np.array(avg_loss.cpu().numpy() / 100)

    torch.save(encoder.state_dict(), os.path.join(args.model_path, 'cae_gps_encoder.model'))
    torch.save(decoder.state_dict(), os.path.join(args.model_path, 'cae_gps_decoder.model'))
    np.save(os.path.join(args.model_path, 'gps_avg_loss_list.npy'), avg_loss_list)
    np.save(os.path.join(args.model_path, 'gps_val_loss.npy'), val_loss)


def encode_gps(args):
    seed = args.seed
    torch.manual_seed(seed)

    encoder = Encoder()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        encoder.to(device)
        
    model = torch.load(os.path.join(args.model_path,'cae_gps_encoder.model'), map_location=torch.device('cpu'))
    encoder.load_state_dict(model)

    discrete_time = 10
    env_list = torch.ones((args.total_env_num, 1, discrete_time, 120, 200)).to("cpu")

    print("load env ")

    if args.load:
        env_list = np.load(os.path.join(args.model_path, 'gps_env_list_planning.npy'))
        env_list = torch.from_numpy(env_list).to("cpu")
    else:
        for env_num in range(args.total_env_num):
            print("env ", env_num)
            # load environment
            env_grid = load_env(env_num).permute(2, 0, 1)
            env_grid = TF.resize(env_grid, (120, 200)).to("cpu")

            for i in range(discrete_time):
                env_list[env_num][0][i] = env_grid[i * discrete_time]

        ##save in np
        np.save(os.path.join(args.model_path, 'gps_env_list_planning.npy'), env_list.numpy())

    dataset = TensorDataset(env_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("env loaded")

    latent_space_list = []

    for batch_idx, batch in enumerate(dataloader):
        cur_grid_batch = batch[0].to(device)

        # ===================forward=====================
        latent_space = encoder(cur_grid_batch)
        latent_space_list.append(latent_space)

    latent_space_list = torch.cat(latent_space_list, dim=0).to("cpu")

    return latent_space_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # mpnet training data generation
    parser.add_argument('--gps_training_data_generation', type=bool, default=True)
    parser.add_argument('--gps_training_env_path', type=str, default="mpnet_data/env/")
    parser.add_argument('--total_env_num', type=int, default=500)
    parser.add_argument('--training_env_num', type=int, default=500)
    parser.add_argument('--validation_env_num', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='./mpnet_data/models/', help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    # test(args)
    main(args)