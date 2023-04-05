from mpnet_data.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np
from gps_planning.utils import find_start_goal, check_start_goal
from gps_planning.simulation_objects import EgoVehicle
from plots.plot_functions import plot_grid
from env.environment import Environment as Env
import torch
import logging
import hyperparams
import os
from gps_planning.utils import pos2gridpos, traj2grid, shift_array, \
    pred2grid, get_mesh_sample_points, sample_pdf, enlarge_grid, compute_gradient, make_path
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def get_grid(args, obslist):
    objects = []
    for i in range(len(obslist)):
        obs, gps_dynamics = obslist[i]
        map = DynamicObstacle(args, name="gpsmaps%d" % i, coord=obs, velocity_x=gps_dynamics[1], velocity_y=gps_dynamics[2],
                              gps_growthrate=gps_dynamics[0], isgps=True)
        objects.append(map)

    environment = Environment(objects, args)

    return environment.grid

def plot_grid(object,start, goal, args, timestep=None, cmap='binary', name=None,
              save=True, show=True, filename=None, include_date=False, folder=None):
    """
    function for occupation map of the environment
    """

    if save:
        plt.close("all")
    if torch.is_tensor(object):
        if object.dim() == 3:
            if timestep is None:
                grid = object[:, :, -1]
                str_timestep = f"\n at timestep={object.shape[2] - 1}"
            else:
                grid = object[:, :, timestep]
                str_timestep = f"\n at timestep={timestep}"
        else:
            grid = object
            str_timestep = ""
        if name is None:
            name = "grid"
    else:
        if timestep is None:
            timestep = object.current_timestep
        str_timestep = f"\n at timestep={timestep}"
        grid = object.grid[:, :, timestep]
        if name is None:
            name = object.name
    x_wide = max(np.abs((args.environment_size[1] - args.environment_size[0])) / 10, 3) *2
    y_wide = np.abs((args.environment_size[3] - args.environment_size[2])) / 10 *2.5
    plt.figure(figsize=(x_wide, y_wide), dpi=200)
    plt.pcolormesh(grid.T, cmap=cmap, norm=None)
    plt.axis('scaled')

    gridpos_x, gridpos_y = pos2gridpos(args, pos_x=[start[0], goal[0]], pos_y=[start[1], goal[1]])

    plt.scatter(gridpos_x[0], gridpos_y[0], c="red", marker='o', s=80, label="Start $\mathbf{x}_0$")
    plt.scatter(gridpos_x[1], gridpos_y[1], c="red", marker='x', s=100, label="Goal $\mathbf{x}_{goal}$")

    ticks_x = np.concatenate(
        (np.arange(0, args.environment_size[1] + 1, 10), np.arange(-10, args.environment_size[0] - 1, -10)), 0)
    ticks_y = np.concatenate(
        (np.arange(0, args.environment_size[3] + 1, 10), np.arange(-10, args.environment_size[2] - 1, -10)), 0)
    ticks_x_grid, ticks_y_grid = pos2gridpos(args, ticks_x, ticks_y)
    plt.xticks(ticks_x_grid, ticks_x)
    plt.yticks(ticks_y_grid, ticks_y)

    plt.title(f"{name}")
    plt.tight_layout()
    plt.savefig(folder + name, dpi=200)
    plt.clf()

def grid_visualizer(args, start, goal, folder, grid, name):
    # create colormap
    greys = cm.get_cmap('Greys')
    grey_col = greys(range(0, 256))
    greens = cm.get_cmap('Greens')
    green_col = greens(range(0, 256))
    blue = np.array([[0.212395, 0.359683, 0.55171, 1.]])
    # yellow = np.array([[0.993248, 0.906157, 0.143936, 1.      ]])
    colorarray = np.concatenate((grey_col[::2, :], green_col[::2, :], blue))
    cmap = ListedColormap(colorarray)

    grid_env_sc = 127 * grid.clone().detach()
    grid_all = torch.clamp(grid_env_sc[:, :, 0], 0, 256)
    plot_grid(grid_all,start, goal, args, name=name, cmap=cmap, show=False, save=True, folder=folder)


if __name__ == '__main__':
    ### load hyperparameters
    args = hyperparams.parse_args()

    env_num = args.training_env_num
    traj_num = args.training_traj_num
    seed = args.trainig_datagen_seed

    np.random.seed(seed)

    for i in range(env_num):
        print("env: ", i)
        obslist = []
        obj_num = np.random.randint(2, 8)
        max_total_area = (40 * 24) * 0.3
        cur_total_area = 0

        for _ in range(obj_num):
            flag = True
            while flag:
                wide = np.random.randint(1, 8)

                if wide < 4:
                    height = np.random.randint(1, wide+1)
                else:
                    height = np.random.randint(wide-1, 8)

                x = np.random.randint(-12, 12 - wide)
                y = np.random.randint(-30, 10 - height)
                obs = [x, x + wide, y, y + height, 1, 1]

                gps_growthrates = np.random.uniform(-0.1, 0.1)
                gps_meanvel_x = np.random.uniform(-0.1, 0.1)
                gps_meanvel_y = np.random.uniform(-0.1, 0.1)
                gps_dynamics = [gps_growthrates, gps_meanvel_x, gps_meanvel_y]

                object_info = [obs, gps_dynamics]

                cur_area = wide * height
                if cur_area + cur_total_area < max_total_area:
                    obslist.append(object_info)
                    flag = False
                    cur_total_area += cur_area

        obslist = np.array(obslist)

        directory = "./mpnet_data/env/" + str(i)

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory + "/env_" + str(i), obslist)

        start_goal_list = []
        grid_plot = get_grid(args, obslist)
        grid = grid_plot.squeeze()
        check = torch.tensor(grid)

        for k in range(traj_num):
            flag = True

            while flag:
                start = [0, -25, 1.5, 3, 0]
                goal = [0, 8, 4, 1, 0]

                goal_x_step = np.random.randint(-9, 9)
                goal_y_step = np.random.randint(-8, 1)
                start_x_step = np.random.randint(-9, 9)
                start_y_step = np.random.randint(-5, 5)

                start[0] += start_x_step
                start[1] += start_y_step
                goal[0] += goal_x_step
                goal[1] += goal_y_step

                gridpos_x, gridpos_y = pos2gridpos(args, pos_x=[start[0], goal[0]],
                                                   pos_y=[start[1], goal[1]])

                gridpos_x = gridpos_x.numpy()
                gridpos_y = gridpos_y.numpy()

                continue_flag = False

                for q in range(gridpos_y[0], gridpos_y[0]+ 40):
                    if check[gridpos_x[0]][q] == 1:
                        continue_flag = True

                if continue_flag == True:
                    continue

                if grid[gridpos_x[0]][gridpos_y[0]] == 0 and grid[gridpos_x[1]][gridpos_y[1]] == 0:
                    start_goal = [start, goal]
                    start_goal_list.append(start_goal)
                    flag = False
                    name = "env_" + str(i) + "_" + str(k)
                    grid_visualizer(args, start, goal, directory + "/", grid_plot, name)
                    grid[gridpos_x[0]][gridpos_y[0]] = 1
                    grid[gridpos_x[1]][gridpos_y[1]] = 1

        start_goal_list = np.array(start_goal_list)
        np.save(directory + "/env_" + str(i) + "_start_goal", start_goal_list)

    print("done")