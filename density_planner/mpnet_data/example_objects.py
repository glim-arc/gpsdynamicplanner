from gps_planning.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np
from gps_planning.utils import find_start_goal, check_start_goal
from gps_planning.simulation_objects import EgoVehicle
from plots.plot_functions import plot_grid
from env.environment import Environment as Env
import torch
import logging


def create_mp_task(args, seed, env, cur_start_goal):
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
    env = create_environment(args, env, timestep=100, stationary=args.mp_stationary)
    logging.info("Loading Simulated Environment (seed %d)" % (seed))

    # start point
    start, goal = cur_start_goal
    xref0 = torch.from_numpy(start).reshape(1, -1, 1).type(torch.FloatTensor)
    # goal point
    xrefN = torch.from_numpy(goal).reshape(1, -1, 1)

    logging.info("Start State: [%.1f, %.1f, %.1f, %.1f]" % (xref0[0, 0, 0], xref0[0, 1, 0], xref0[0, 2, 0], xref0[0, 3, 0]))
    logging.info("Goal Position: [%.1f, %.1f]" % (xrefN[0, 0, 0], xrefN[0, 1, 0]))

    # create the ego vehicle
    ego = EgoVehicle(xref0, xrefN, env, args, video=args.mp_video)
    return ego


def create_environment(args, env, object_str_list=None, name="environment", timestep=0, stationary=False):
    """
    create random environment

    :param args:            settings
    :param object_str_list: list of objects (if None: objects will be randomly generated)
    :param name:            name of the environment
    :param timestep:        time duration of the prediction
    :param stationary:      True if environment contains only stationary obstacles
    :return: environment
    """
    logging.info("create training environment")

    objects = []
    obslist = env

    for i in range(len(obslist)):
        obs, gps_dynamics = obslist[i]
        map = DynamicObstacle(args, name="gpsmaps%d" % i, coord=obs, velocity_x=gps_dynamics[1],
                              velocity_y=gps_dynamics[2],
                              gps_growthrate=gps_dynamics[0], isgps=True)
        objects.append(map)

    environment = Environment(objects, args)

    if timestep > 0:
        environment.forward_occupancy(step_size=timestep)
    return environment