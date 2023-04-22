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


def plot_traj(self, ego_dict, mp_results, mp_methods, args, folder=None, traj_idx=None, animate=False,
              include_density=False,
              name=None):
    """
    function for plotting the reference trajectories planned by different motion planners in the occupation grid
    """

    gps_colormap = colors.LinearSegmentedColormap.from_list("",
                                                            ["black", "green", "yellow", "orange", "red", "darkred"])

    colorarray = np.concatenate((convert_color(TUMBlue),
                                 convert_color(TUMGray),
                                 convert_color(TUMGreen_acc),
                                 convert_color(TUMOrange_acc),
                                 convert_color(TUMBlue_light),
                                 convert_color(TUMGray_light),
                                 convert_color(TUMBlue_med)))

    if mp_methods[0] == "sys":
        colorarray = np.concatenate((convert_color(TUMGray),
                                     convert_color(TUMBlue),
                                     convert_color(TUMGreen_acc),
                                     convert_color(TUMOrange_acc),
                                     convert_color(TUMBlue_light),
                                     convert_color(TUMGray_light),
                                     convert_color(TUMBlue_med)))
    col_start = convert_color(MITRed)
    plt.rcParams['legend.fontsize'] = 22
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    legend = True

    plt.close("all")
    if args.mp_use_realEnv:
        div = 6
    else:
        div = 4
    x_wide = (np.abs((args.environment_size[1] - args.environment_size[0])) / div) * 2
    y_wide = (np.abs((args.environment_size[3] - args.environment_size[2])) / div) * 2.5

    dpi = 80
    axis_tick = 1

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
        # grid = ego_dict["grid"][:, :, [t_idx]]
        # grid_all = 1 - np.repeat(grid, 4, axis=2)
        # grid_all[:, :, 3] = 1
        grid_all = ego_dict["grid"][:, :, [t_idx]] * 40
        building = ego_dict["building"][:, :, [t_idx]]
        buildding_filter = building > 0
        grid_all[buildding_filter] = 50
        grid_all = grid_all.squeeze().T
        trjgrid = torch.zeros_like(grid_all)

        if mp_methods[0] == "sys":
            plt.figure(figsize=(x_wide, y_wide + 1.5), dpi=dpi)
        elif args.mp_use_realEnv or name is not None:
            plt.figure(figsize=(x_wide, y_wide + 1), dpi=dpi)
        elif name is not None:
            plt.figure(figsize=(x_wide, y_wide + 1.5), dpi=dpi)
        elif not legend:
            plt.figure(figsize=(x_wide + 1, y_wide), dpi=dpi)
        else:
            plt.figure(figsize=(1.85 * x_wide, y_wide),
                       dpi=dpi)  # plt.figure(figsize=(x_wide, y_wide + 2.5), dpi=100)

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
            elif mp_methods[i] == "oracle":
                label = "Oracle"
            elif mp_methods[i] == "tube2MPC":
                if args.mp_use_realEnv == False:
                    label = "MPC with \n $r_\\textrm{tube}=0.5m$"
                else:
                    label = "MPC with $r_\\textrm{tube}=0.5m$"
            elif mp_methods[i] == "tube3MPC":
                label = "MPC with \n $r_\\textrm{tube}=1m$"
            else:
                label = mp_methods[i]

            if i >= len(colorarray):
                i = 0

            plt.plot(0, 0, "-", color=colorarray[i, :], label=label)

            if x_traj is None:
                continue
            for num_traj in range(x_traj.shape[0]):
                grid_traj = traj2grid(x_traj[[num_traj], :, :t_idx * 10 + 1], args)
                # idx = grid_traj != 0
                # grid_idx = grid_all[idx]
                # grid_idx[:, :] = torch.from_numpy(colorarray[[i - 1], :])  # .unsqueeze(0)
                # grid_all[idx] = grid_idx

                grid_traj = grid_traj.T
                trajidx = grid_traj != 0
                grid_all[trajidx] = -15

            if include_density:
                rho_traj = rho_traj_list[i]
                x_trajs = x_trajs_list[i]
                gridpos_x, gridpos_y = pos2gridpos(args, pos_x=x_trajs[:, 0, t_idx],
                                                   pos_y=x_trajs[:, 1, t_idx])
                plt.scatter(gridpos_x[:], gridpos_y[:], c=colorarray[[i - 1], :], marker='.',
                            s=500 * rho_traj[:, 0, t_idx])
            if x_traj is None or x_traj.shape[2] < t_idx * 10:
                continue
            gridpos_x, gridpos_y = pos2gridpos(args, pos_x=[x_traj[0, 0, t_idx * 10]],
                                               pos_y=[x_traj[0, 1, t_idx * 10]])

            plt.scatter(gridpos_x[0], gridpos_y[0], c="red", marker='o', s=10)

        plt.imshow(grid_all, origin="lower", cmap=gps_colormap)
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
        plt.axis('on')

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
            (np.arange(0, args.environment_size[1] + 1, axis_tick),
             np.arange(-axis_tick, args.environment_size[0] - 1, -axis_tick)), 0)
        ticks_y = np.concatenate(
            (np.arange(0, args.environment_size[3] + 1, axis_tick),
             np.arange(-axis_tick, args.environment_size[2] - 1, -axis_tick)), 0)
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
        plt.savefig(folder + filename, dpi=dpi)
        plt.clf()

if __name__ == '__main__':
    ### load hyperparameters
    args = hyperparams.parse_args()

    env_num = args.training_env_num
    traj_num = args.training_traj_num

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        for j, cur_start_goal in enumerate(start_goal_list):
            print(env_file, j)
            ### create environment and motion planning problem
            ego = create_mp_task(args, seed, env, cur_start_goal)

            planner = MotionPlannerGrad(ego, name=env_file[:-4], path_log=path_log)
            up, cost = planner.up_datagen()
            traj_output = np.array([up.detach().numpy(), cost.detach().numpy()])

            #save traj output
            trajname = "traj_output_" + str(j)

            #visualize trajectory
            uref_traj, xref_traj = ego.system.up2ref_traj(ego.xref0.repeat(up.shape[0], 1, 1),
                                                               up, ego.args, short=True)

            plotfolder = directory + "/"
            ego.visualize_xref(xref_traj, save=True, show=False, name= trajname, folder=plotfolder)
            np.save(plotfolder + trajname, traj_output)

    print("end")