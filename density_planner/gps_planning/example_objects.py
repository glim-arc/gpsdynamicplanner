from gps_planning.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np
from gps_planning.utils import find_start_goal, check_start_goal
from gps_planning.simulation_objects import EgoVehicle
from plots.plot_functions import plot_grid
from env.environment import Environment as Env
import torch
import logging


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

    # start point
    xref0 = torch.tensor([0, -25, 1.5, 3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
    # goal point
    xrefN = torch.tensor([0., 8, 4, 1, 0]).reshape(1, -1, 1)

    if args.mp_plot_envgrid:
        plot_grid(env, args, timestep=1, save=True)
        for t in [1, 20, 40, 60, 80, 100]:
            plot_grid(env, args, timestep=t, save=False)
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

    wide = 3
    height = 8
    obs1 = np.array([-3, -3 + wide, 0,0 + height, 1, 1])
    obs2 = np.array([-3, -3 + wide, -12, -12 + height, 1, 1])
    obs3 = np.array([3, 3 + wide, 0, 0 + height, 1, 1])
    obs4 = np.array([3, 3 + wide, -12, -12 + height, 1, 1])
    obs5 = np.array([0, 0 + wide, -20, -20 + height/2, 1, 1])

    objects = []

    obslist = [obs1, obs2, obs3, obs4, obs5]
    # gps_growthrates = [0.05, 0, 0.05, 0, 0]
    # gps_meanvel = [0.05, 0, -0.05, 0, 0]

    #both_stationary
    gps_growthrates = [0.08, 0, 0.08, 0, 0]
    gps_meanvel = [0, 0, 0, 0, 0]

    #one side
    gps_growthrates = [0, 0, -0.05, 0, 0]
    gps_meanvel = [0, 0, 0.05, 0, 0]

    # one side static
    # gps_growthrates = [0, 0, 0, 0, 0]
    # gps_meanvel = [0, 0, 0, 0, 0]

    for i in range(len(obslist)):
        obs = obslist[i]
        obj = StaticObstacle(args, name="staticObs%d" % i, coord=obs, isgps = True)
        map = DynamicObstacle(args, name="gpsmaps%d" % i, coord=obs, velocity_x=gps_meanvel[i],gps_growthrate=gps_growthrates[i], isgps = True)
        #objects.append(obj)
        objects.append(map)

    environment = Environment(objects, args, name=name)
    if timestep > 0:
        environment.forward_occupancy(step_size=timestep)
    return environment