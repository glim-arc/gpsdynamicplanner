from gps_planning.simulation_objects_gps import Environment
import numpy as np
from gps_planning.utils import find_start_goal, check_start_goal
from gps_planning.simulation_objects_gps import EgoVehicle
from plots.plot_functions import plot_grid
from env.environment import Environment as Env
import matplotlib.pyplot as plt
import torch
import logging
import os
from scipy.ndimage import gaussian_filter

def create_mp_task(args, seed):
    """
    function to load the environment and sample the initial state and the goal position

    :param args:    settings
    :param seed:    random seed
    :return: ego:   object for the ego vehicle which contains the start and goal position and the environment
                        occupation map
    """
    logging.info("")
    logging.info("###################################################################")
    logging.info("###################################################################")
    ### create environment and motion planning problem

    # generate random environment
    env = create_environment(args, timestep=0, stationary=args.mp_stationary)
    logging.info("Loading Simulated Environment (seed %d)" % (seed))

    # start point
    xref0 = torch.tensor([-20, -15, 1, 0, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
    # goal point
    xrefN = torch.tensor([0,-10, 1, 0, 0]).reshape(1, -1, 1)

    logging.info("Start State: [%.1f, %.1f, %.1f, %.1f]" % (xref0[0, 0, 0], xref0[0, 1, 0], xref0[0, 2, 0], xref0[0, 3, 0]))
    logging.info("Goal Position: [%.1f, %.1f]" % (xrefN[0, 0, 0], xrefN[0, 1, 0]))

    # create the ego vehicle
    ego = EgoVehicle(xref0, xrefN, env, args, video=args.mp_video)
    return ego


def create_environment(args, object_str_list=None, name="environment", timestep=0, stationary=False):
    """
    create random environment

    :param args:            settings
    :param object_str_list: list of objects (if None: objects will be randomly generated)
    :param name:            name of the environment
    :param timestep:        time duration of the prediction
    :param stationary:      True if environment contains only stationary obstacles
    :return: environment
    """
    logging.info("create random environment")

    gpsenvlist = os.listdir(args.gps_env_path)

    obslist = []
    gps_map_list = []

    for i,file in enumerate(gpsenvlist):
        if file[-3:] == "npy":
            if file.find("heat") != -1:
                gps_map_list.append(file)
            if file.find("obs") != -1:
                obslist.append(file)

    obs_maps = []
    gps_maps = []
    obs_maps_non_gaus = []
    gps_maps_non_gaus = []
    spread = 1.5
    discount = 1
    gps_discount = 0.7

    for i, map in enumerate(obslist):
        temp = np.load(args.gps_env_path + "/" + map)

        # apply gaussian
        temp_gaus = gaussian_filter(temp * discount, sigma=spread)

        obs_maps.append(torch.FloatTensor(temp_gaus).T)
        obs_maps_non_gaus.append(torch.FloatTensor(temp).T)

        # plt.imshow(temp_gaus)
        # plt.show()

    for i, map in enumerate(gps_map_list):
        temp = np.load(args.gps_env_path + "/" + map)/40
        # filter = obs_maps_non_gaus[i] >0
        # temp[filter] = 0

        #filter gps
        filter = temp < 0.3
        temp[filter] = 0

        # apply gaussian
        temp_gaus = gaussian_filter(temp*gps_discount, sigma=spread)

        gps_maps.append(torch.FloatTensor(temp_gaus).T)
        gps_maps_non_gaus.append(torch.FloatTensor(temp).T)

    gps_maps = torch.stack(gps_maps, dim = 2)
    obs_maps = torch.stack(obs_maps, dim = 2)
    gps_maps_non_gaus = torch.stack(gps_maps_non_gaus, dim=2)
    obs_maps_non_gaus = torch.stack(obs_maps_non_gaus, dim=2)

    timestep = len(obslist)
    environment = Environment(obs_maps, gps_maps,obs_maps_non_gaus,gps_maps_non_gaus,  args, name=name)

    return environment