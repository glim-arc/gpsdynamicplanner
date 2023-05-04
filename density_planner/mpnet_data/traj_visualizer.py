import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
import hyperparams
import torch
import pickle
from gps_planning.utils import initialize_logging, get_cost_table, get_cost_increase
from gps_planning.MotionPlannerGrad import MotionPlannerGrad
import numpy as np
from mpnet_data.example_objects import create_mp_task
import os
from plots.plot_functions import plot_ref, plot_grid, plot_traj
from systems.sytem_CAR import Car
import matplotlib
import matplotlib.pyplot as plt
from gps_planning.utils import make_path, pos2gridpos, traj2grid, pred2grid, convert_color
import matplotlib.colors as colors

### settings
plt.style.use('seaborn-paper')
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['text.usetex'] = False
plt.rcParams['legend.fontsize'] = 15
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

###colors
MITRed = np.array([163 / 256, 31 / 256, 52 / 256])
TUMBlue = np.array([0, 101 / 256, 189 / 256])
TUMBlue_med = np.array([153 / 256, 193 / 256, 229 / 256])
TUMBlue_light = np.array([230 / 256, 240 / 256, 249 / 256])
TUMGray = np.array([128 / 256, 128 / 256, 128 / 256])
TUMGray_light = np.array([204 / 256, 204 / 256, 198 / 256])
TUMOrange_acc = np.array([227 / 256, 114 / 256, 34 / 256])
TUMGreen_acc = np.array([162 / 256, 113 / 256, 0 / 256])
Green = np.array([0 / 256, 100 / 256, 0 / 256])

def visualize_xref(args,ego_dict, xref_traj, uref_traj=None, show=True, save=False, include_date=True,
                   name='Reference Trajectory', folder=None, x_traj=None):
    """
    plot the reference trajectory in the occupation map of the environment

    :param xref_traj:   reference state trajectory
    :param uref_traj:   reference input trajectory
    :param show:        True if plot should be shown
    :param save:        True if plot should be saved
    :param include_date:True if filename includes the date
    :param name:        name for the plot
    :param folder:      folder where the plot should be saved
    """

    if x_traj is not None:
        mp_methods = ["sys", "ref"]
        mp_results = {"sys": {"x_traj": [x_traj]}, "ref": {"x_traj": [xref_traj]}}
    else:
        mp_methods = ["ref"]
        mp_results = {"ref": {"x_traj": [xref_traj]}}

    plot_traj(ego_dict, mp_results, mp_methods, args, folder=folder, traj_idx=0, animate=False,
              include_density=False, name=name)

def plot_traj(ego_dict, mp_results, mp_methods, args, folder=None, traj_idx=None, animate=False, include_density=False,
              name=None):
    """
    function for plotting the reference trajectories planned by different motion planners in the occupation grid
    """

    colorarray = np.concatenate((convert_color(TUMBlue),
                                 convert_color(TUMGray),
                                 convert_color(TUMGreen_acc),
                                 convert_color(TUMOrange_acc),
                                 convert_color(TUMBlue_light),
                                 convert_color(Green),
                                 convert_color(TUMBlue_med),convert_color(MITRed)
                                 ))

    if mp_methods[0] == "sys":
        colorarray = np.concatenate((convert_color(TUMGray),
                                     convert_color(TUMBlue),
                                     convert_color(TUMGreen_acc),
                                     convert_color(TUMOrange_acc),
                                     convert_color(TUMBlue_light),
                                     convert_color(TUMGray_light),
                                     convert_color(TUMBlue_med)))
    col_start = convert_color(MITRed)
    plt.rcParams['legend.fontsize'] = 15
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    legend = True

    plt.close("all")
    if args.mp_use_realEnv:
        div = 6
    else:
        div = 4
    x_wide = np.abs((args.environment_size[1] - args.environment_size[0])) / div
    y_wide = np.abs((args.environment_size[3] - args.environment_size[2])) / div

    if animate:
        folder = make_path(folder, "GridTraj%d" % traj_idx)

    if traj_idx is None:
        traj_idx = len(mp_results[mp_methods[0]]["x_traj"]) - 1
    x_traj_list = []
    rho_traj_list = []
    x_trajs_list = []
    for method in mp_methods:
        if mp_results[method]["x_traj"][traj_idx] is None:
            x_traj_list.append(None)
            rho_traj_list.append(None)
            x_trajs_list.append(None)
            continue
        x_traj = np.array(mp_results[method]["x_traj"][traj_idx].detach())
        x0 = ego_dict["start"]
        idx_old = np.linspace(0, x_traj.shape[2] - 1, x_traj.shape[2])
        idx_new = np.linspace(0, x_traj.shape[2] - 1, (x_traj.shape[2] - 1) * 10 + 1)
        x_traj_long = np.zeros((x_traj.shape[0], x_traj.shape[1], (x_traj.shape[2] - 1) * 10 + 1))
        for j in range(x_traj.shape[1]):
            for traj_i in range(x_traj.shape[0]):
                x_traj_long[traj_i, j, :] = np.interp(idx_new, idx_old, x_traj[traj_i, j, :])
        x_traj_list.append(torch.from_numpy(x_traj_long))
        if include_density:
            rho_traj_list.append(np.array(mp_results[method]["rho_traj"][traj_idx].detach()))
            x_trajs_list.append(np.array(mp_results[method]["x_trajs"][traj_idx].detach()))

    if animate:
        start_idx = 0
    else:
        start_idx = x_traj.shape[2] - 1

    for t_idx in range(start_idx, x_traj.shape[2]):
        grid = ego_dict["grid"][:, :, [t_idx]]
        grid_all = 1 - np.repeat(grid, 4, axis=2)
        grid_all[:, :, 3] = 1
        if mp_methods[0] == "sys":
            plt.figure(figsize=(x_wide, y_wide + 1.5), dpi=100)
        elif args.mp_use_realEnv or name is not None:
            plt.figure(figsize=(x_wide, y_wide + 1), dpi=100)
        elif name is not None:
            plt.figure(figsize=(x_wide, y_wide + 1.5), dpi=100)
        elif not legend:
            plt.figure(figsize=(x_wide + 1, y_wide), dpi=100)
        else:
            plt.figure(figsize=(1.85 * x_wide, y_wide), dpi=100)  # plt.figure(figsize=(x_wide, y_wide + 2.5), dpi=100)

        for i, x_traj in enumerate(x_traj_list):
            if mp_methods[i] == "grad" and "search" in mp_methods:
                label = "Gradient-based \n Method"  # "Density planner"
            elif mp_methods[i] == "ref":
                label = "Reference trajectory"
            elif mp_methods[i] == "sys":
                label = "System trajectories"
            elif mp_methods[i] == "search":
                label = "Search-based \n Method"
            elif mp_methods[i] == "sampl":
                label = "Sampling-based \n Method"
            elif mp_methods[i] == "grad":
                label = "Density planner"
            else:
                label = mp_methods[i]

            if i >= len(colorarray):
                i = 0

            traj_colors = []

            if x_traj is None:
                continue
            for num_traj in range(x_traj.shape[0]):
                grid_traj = traj2grid(x_traj[[num_traj], :, :t_idx * 10 + 1], args)
                idx = grid_traj != 0
                grid_idx = grid_all[idx]
                traj_color = colorarray[[-3], :]*0.8
                if num_traj == 0:
                    traj_color = colorarray[[-1], :]

                traj_colors.append(traj_color)

                grid_idx[:, :] = torch.from_numpy(traj_color*1.05**num_traj)
                grid_all[idx] = grid_idx

            plt.plot(0, 0, "-", color=traj_colors[0], label="Gradient based reference trajectory")
            plt.plot(0, 0, "-", color=traj_colors[1], label="MPnet based generated trajectories")
            plt.imshow(torch.transpose(grid_all, 0, 1), origin="lower")

            if x_traj is None or x_traj.shape[2] < t_idx * 10:
                continue
            gridpos_x, gridpos_y = pos2gridpos(args, pos_x=[x_traj[0, 0, t_idx * 10]],
                                               pos_y=[x_traj[0, 1, t_idx * 10]])

            plt.scatter(gridpos_x[0], gridpos_y[0], c="red", marker='o', s=10)

        plt.imshow(torch.transpose(grid_all, 0, 1), origin="lower")
        gridpos_x, gridpos_y = pos2gridpos(args, pos_x=[x0[0, 0, 0], ego_dict["goal"][0, 0, 0]],
                                           pos_y=[x0[0, 1, 0], ego_dict["goal"][0, 1, 0]])
        if mp_methods[0] == "ref":
            plt.scatter(gridpos_x[0], gridpos_y[0], c=col_start, marker='o', s=80, label="Start $\mathbf{x}_0$")
        elif mp_methods[0] == "sys":
            col_start2 = col_start.copy()
            col_start2[0, 3] = 0.2
            plt.scatter(gridpos_x[0], gridpos_y[0], c=col_start2, marker='s', s=800,
                        label="Initial density distribution")
        else:
            plt.scatter(gridpos_x[0], gridpos_y[0], c=col_start, marker='o', s=80, label="Start")
        plt.scatter(gridpos_x[1], gridpos_y[1], c=col_start, marker='x', s=100, label="Goal $\mathbf{x}_{goal}$")
        plt.axis('scaled')
        if legend:
            if args.mp_use_realEnv == False:
                if name is not None:
                    plt.legend(bbox_to_anchor=(0.5, -0.09), loc="upper center")
                else:
                    plt.legend(bbox_to_anchor=(1.4, 0.5), loc="center", labelspacing=1.1)
            elif args.mp_recording == 26:
                plt.legend(loc="upper right")
            else:
                plt.legend(loc="upper left")
        if name is not None:
            if isinstance(name, str):
                plt.title(name, fontsize=24)
            else:
                plt.title("Iteration %d" % name, fontsize=24)
        else:
            plt.title("$t_k=%.2fs$" % (t_idx / 10.), fontsize=24)
        ticks_x = np.concatenate(
            (np.arange(0, args.environment_size[1] + 1, 10), np.arange(-10, args.environment_size[0] - 1, -10)), 0)
        ticks_y = np.concatenate(
            (np.arange(0, args.environment_size[3] + 1, 10), np.arange(-10, args.environment_size[2] - 1, -10)), 0)
        ticks_x_grid, ticks_y_grid = pos2gridpos(args, ticks_x, ticks_y)
        plt.xticks(ticks_x_grid, ticks_x)
        plt.yticks(ticks_y_grid, ticks_y)
        plt.xlabel("$p_x$ [m]")
        plt.ylabel("$p_y$ [m]")

        # plt.title(f"{name}" + str_timestep)
        plt.tight_layout()
        if folder is None:
            folder = args.path_plot_grid
        if name is None:
            filename = "GridTraj%d" % traj_idx + "_%d" % t_idx + ".jpg"
        else:
            if isinstance(name, str):
                filename = name
            else:
                filename = "iter%d" % name

        filename += ".jpg"
        plt.savefig(folder + filename, dpi=100)
        plt.clf()

if __name__ == '__main__':
    ### load hyperparameters
    args = hyperparams.parse_args()

    env_num = args.training_env_num
    traj_num = args.training_traj_num

    #args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.device(args.device)

    ### choose methods
    opt_methods = ["grad"]
    mp_methods = ["grad"]

    seed = args.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create logging folder
    path_log = initialize_logging(args, "up_datagen")

    print("start generating training traj")

    avg_difference = []

    ### load env and datasets
    for i in range(env_num):
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

        env = np.load(directory + "/" + env_file, allow_pickle=True)
        start_goal_list = np.load(directory + "/" + start_goal_file, allow_pickle=True)

        ego = create_mp_task(args, seed, env, [np.zeros((1,5)), np.zeros((1,5))])
        planner = MotionPlannerGrad(ego, name=env_file[:-4], path_log=path_log)

        for j, cur_start_goal in enumerate(start_goal_list):
            print(env_file, j)
            ### create environment and motion planning problem

            start, goal = cur_start_goal
            xref0 = torch.from_numpy(start).reshape(1, -1, 1).type(torch.FloatTensor)
            # goal point
            xrefN = torch.from_numpy(goal).reshape(1, -1, 1).type(torch.FloatTensor)

            ego.xref0 = xref0
            ego.xrefN = xrefN

            real_traj_path = ""
            gen_traj_path = ""

            for file in files:
                if file.find("traj_output_" + str(j)) != -1:
                    if file[-3:] == "npy":
                        if file.find("gen") != -1:
                            gen_traj_path = file
                        else:
                            real_traj_path = file

            #load traj output
            print(os.path.join(directory, real_traj_path))
            real_data = np.load(os.path.join(directory, real_traj_path), allow_pickle=True)
            real_up = torch.from_numpy(real_data[0])
            gen_data = torch.from_numpy(np.load(os.path.join(directory, gen_traj_path)))
            gen_up = torch.zeros_like(real_up)
            gen_up[0][0] = gen_data[0,:]
            gen_up[0][1] = gen_data[1,:]
            print(gen_up)

            #visualize trajectory
            real_uref_traj, real_xref_traj = ego.system.up2ref_traj(xref0.repeat(1, 1, 1),
                                                                    real_up, args, short=True)
            gen_uref_traj, gen_xref_traj = ego.system.up2ref_traj(xref0.repeat(gen_up.shape[0], 1, 1),
                                                                  gen_up, args, short=True)


            real_cost, real_dict = planner.get_up_cost(real_up, real_uref_traj, real_xref_traj)
            gen_cost, gen_dict = planner.get_up_cost(gen_up, gen_uref_traj, gen_xref_traj)

            avg_difference.append((gen_cost - real_cost)/real_cost*100)

            plotfolder = directory + "/"

            xref_traj = torch.cat([real_xref_traj, gen_xref_traj], dim = 0)

            trajname = "traj_output_" + str(j)+"_gen"

            ego_dict = {"grid": torch.clamp(ego.env.grid + ego.env.gps_grid, 0, 1),
                        "start": xref0,
                        "goal": xrefN,
                        "args": args}

            visualize_xref(args, ego_dict, xref_traj=xref_traj, save=True, show=False, name= trajname, folder=plotfolder)

    print("Average cost difference percentage: ", sum(avg_difference) / len(avg_difference))
    print("end")
