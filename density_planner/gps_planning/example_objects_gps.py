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
    env = create_environment(args, timestep=100, stationary=args.mp_stationary)
    logging.info("Loading Simulated Environment (seed %d)" % (seed))
    gps_test_case = args.gps_test_case

    if gps_test_case == True:
        # start
        xref0 = torch.tensor([-5, -28, 1.5, 3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
        # goal
        xrefN = torch.tensor([0., 0, 4, 1, 0]).reshape(1, -1, 1)
    else:
        #start
        xref0 = torch.tensor([-5, -28, 1.5, 3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
        #goal
        xrefN = torch.tensor([0., 0, 4, 1, 0]).reshape(1, -1, 1)



    # # side
    # # start point
    # xref0 = torch.tensor([-20, -15, 3, -3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
    # # goal point
    # xrefN = torch.tensor([0., -10, 1, -4, 0]).reshape(1, -1, 1)

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
    logging.info("Load GPS environment")

    gpsenvlist = os.listdir(args.gps_real_env_path)

    obslist = []
    gps_map_list = []

    obslist.sort()
    gps_map_list.sort()

    for i, file in enumerate(gpsenvlist):
        if file[-3:] == "npy":
            if file.find("heat") != -1:
                gps_map_list.append(file)
            if file.find("obs") != -1:
                obslist.append(file)

    obs_maps = []
    gps_maps = []
    obs_maps_non_gaus = []
    gps_maps_non_gaus = []
    spread = 5
    discount = 1
    gps_discount = 0.7
    gps_test_case = args.gps_test_case
    shift = 100

    for i, map in enumerate(obslist):
        temp = np.load(args.gps_real_env_path + "/" + map)

        if gps_test_case == True:
            temp = temp.T

        temp = temp[shift:shift+241][:]

        # apply gaussian
        temp_gaus = gaussian_filter(temp * discount, sigma=spread)

        # obs_maps.append(torch.FloatTensor(temp_gaus).T)
        # obs_maps_non_gaus.append(torch.FloatTensor(temp).T)

        obs_maps.append(temp_gaus)
        obs_maps_non_gaus.append(temp)

        # plt.imshow(temp_gaus)
        # plt.show()

    for i, map in enumerate(gps_map_list):
        temp = np.load(args.gps_real_env_path + "/" + map) / 40

        if gps_test_case == True:
            temp = temp.T

        temp = temp[shift:shift+241][:]

        #Remove obstacles
        # filter = obs_maps_non_gaus[i] >0
        # temp[filter] = 0

        # filter gps
        filter = temp < 0.3
        temp[filter] = 0

        #interpolate by adding 5 more intermediate timesteps
        if i > 0:
            prev = gps_maps_non_gaus[i-1]
            diff = temp - prev
            step = diff / 6
            newtemp = []
            newtempgaus = []

            for i in range(6):
                newtemp.append(prev + step)

            for newmap in newtemp:
                # apply gaussian
                temp_gaus = gaussian_filter(newmap * gps_discount, sigma=spread)
                newtempgaus.append(temp_gaus)

            gps_maps += newtempgaus
            gps_maps_non_gaus += newtemp

        else:
            # apply gaussian
            temp_gaus = gaussian_filter(temp * gps_discount, sigma=spread)

            gps_maps.append(temp_gaus)
            gps_maps_non_gaus.append(temp)

    # plt.imshow(gps_maps[0])
    # plt.show()
    # plt.imshow(obs_maps[0])
    # plt.show()

    for i in range(3):
        obs_maps += [np.copy(map) for map in obs_maps]
        obs_maps_non_gaus += [np.copy(map) for map in obs_maps_non_gaus]

    gps_maps = np.array(gps_maps[:101])
    obs_maps = np.array(obs_maps[:101])
    gps_maps_non_gaus = np.array(gps_maps_non_gaus[:101])
    obs_maps_non_gaus = np.array(obs_maps_non_gaus[:101])

    if args.without_gps_map == True:
        gps_maps = np.zeros_like(gps_maps)
        gps_maps_non_gaus = np.zeros_like(gps_maps_non_gaus)

    gps_maps = torch.FloatTensor(gps_maps).permute(1,2,0)
    obs_maps = torch.FloatTensor(obs_maps).permute(1,2,0)
    gps_maps_non_gaus = torch.FloatTensor(gps_maps_non_gaus).permute(1,2,0)
    obs_maps_non_gaus = torch.FloatTensor(obs_maps_non_gaus).permute(1,2,0)

    environment = Environment(obs_maps, gps_maps, obs_maps_non_gaus, gps_maps_non_gaus, args, name=name)
    return environment