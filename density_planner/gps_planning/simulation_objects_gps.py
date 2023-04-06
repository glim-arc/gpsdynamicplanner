import numpy as np
import torch
from gps_planning.utils import pos2gridpos, traj2grid, shift_array, \
    pred2grid, get_mesh_sample_points, sample_pdf, enlarge_grid, compute_gradient, make_path
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap
from plots.plot_functions import plot_ref, plot_grid, plot_traj
from matplotlib import cm
from matplotlib.colors import ListedColormap
from systems.sytem_CAR import Car
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from gps_planning.utils import make_path, pos2gridpos, traj2grid, pred2grid, convert_color
import matplotlib.colors as colors

### settings
plt.style.use('seaborn-paper')
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{mathrsfs}}'
# plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams['legend.fontsize'] = 18
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

###colors
MITRed = np.array([163 / 256, 31 / 256, 52 / 256])
TUMBlue = np.array([0, 101 / 256, 189 / 256])
TUMBlue_med = np.array([153 / 256, 193 / 256, 229 / 256])
TUMBlue_light = np.array([230 / 256, 240 / 256, 249 / 256])
TUMGray = np.array([128 / 256, 128 / 256, 128 / 256])
TUMGray_light = np.array([204 / 256, 204 / 256, 198 / 256])
TUMOrange_acc = np.array([227 / 256, 114 / 256, 34 / 256])
TUMGreen_acc = np.array([162 / 256, 113 / 256, 0 / 256])


class Environment:
    """
    Environment class which contains the occupation map
    """

    def __init__(self, obs_map, gps_maps,obs_maps_non_gaus,gps_maps_non_gaus, args, name="environment", timestep=0):
        # self.time = time
        self.args = args
        print("device", args.device)
        self.device = args.device
        self.gpsgridvisualize = args.gpsgridvisualize
        self.custom_cost = args.gps_cost
        self.grid = obs_map
        self.gps_grid = gps_maps
        self.grid_non_gaus = obs_maps_non_gaus
        self.gps_grid_non_gaus = gps_maps_non_gaus
        self.grid_size = args.grid_size
        self.current_timestep = timestep
        self.name = name
        self.grid_enlarged = None
        self.grid_gradientX = None
        self.grid_gradientY = None
        self.gps_grid_gradientX = None
        self.gps_grid_gradientY = None

        if self.gpsgridvisualize == True:
            self.grid_visualizer()

    def grid_visualizer(self):
        folder = make_path(self.args.path_plot_motion, "_gps_dynamic")

        # create colormap
        greys = cm.get_cmap('Greys')
        grey_col = greys(range(0, 256))
        greens = cm.get_cmap('Greens')
        green_col = greens(range(0, 256))
        blue = np.array([[0.212395, 0.359683, 0.55171, 1.]])
        # yellow = np.array([[0.993248, 0.906157, 0.143936, 1.      ]])
        colorarray = np.concatenate((grey_col[::2, :], green_col[::2, :], blue))
        cmap = ListedColormap(colorarray)

        grid_env_sc = 127 * self.gps_grid_non_gaus.clone().detach()

        for i in range(self.gps_grid_non_gaus.shape[2]):
            grid_all = torch.clamp(grid_env_sc[:, :, i], 0, 256)
            plot_grid(grid_all, self.args, name="iter%d" % i, cmap=cmap, show=False, save=True, folder=folder)

        if self.custom_cost == False:
            grid_env_sc = 127 * self.grid_non_gaus.clone().detach()
        else:
            grid_env_sc = torch.clamp(self.grid_non_gaus + self.gps_grid_non_gaus, 0, 1)
            grid_env_sc = 127 * grid_env_sc.clone().detach()

        folder = make_path(self.args.path_plot_motion, "gps+grid")

        for i in range(self.grid_non_gaus.shape[2]):
            grid_all = torch.clamp(grid_env_sc[:, :, i], 0, 256)
            plot_grid(grid_all, self.args, name="iter%d" % i, cmap=cmap, show=False, save=True, folder=folder)

    def enlarge_shape(self, table=None):
        """
        enlarge the shape of all obstacles and update the grid to do motion planning for a point
        """
        if table is None:
            table = [[0, 10, 25], [10, 30, 20], [30, 50, 10], [50, 101, 5]]
        grid_enlarged = self.grid.clone().detach()
        for elements in table:
            if elements[0] >= self.grid.shape[2]:
                continue
            timesteps = torch.arange(elements[0], min(elements[1], self.grid.shape[2]))
            grid_enlarged[:, :, timesteps] = enlarge_grid(self.grid[:, :, timesteps], elements[2])
        self.grid_enlarged = grid_enlarged

    def get_gradient(self):
        """
        compute the gradients of the occupation grid
        """
        if self.grid_gradientX is None:
            grid_gradientX, grid_gradientY = compute_gradient(self.grid, step=1)
            s = 5
            missingGrad = torch.logical_and(self.grid != 0, torch.logical_and(grid_gradientX == 0, grid_gradientY == 0))
            while torch.any(missingGrad):
                idx = missingGrad.nonzero(as_tuple=True)
                grid_gradientX_new, grid_gradientY_new = compute_gradient(self.grid, step=s)
                grid_gradientX[idx] += s * grid_gradientX_new[idx]
                grid_gradientY[idx] += s * grid_gradientY_new[idx]
                s += 10
                missingGrad = torch.logical_and(self.grid != 0, torch.logical_and(grid_gradientX == 0, grid_gradientY == 0))

        self.grid_gradientX = grid_gradientX
        self.grid_gradientY = grid_gradientY

        if self.gps_grid_gradientX is None:
            gps_grid_gradientX, gps_grid_gradientY = compute_gradient(self.gps_grid, step=1)
            s = 5
            missingGrad = torch.logical_and(self.gps_grid != 0,
                                            torch.logical_and(gps_grid_gradientX == 0, gps_grid_gradientY == 0))
            while torch.any(missingGrad):
                idx = missingGrad.nonzero(as_tuple=True)
                gps_grid_gradientX_new, gps_grid_gradientY_new = compute_gradient(self.gps_grid, step=s)
                gps_grid_gradientX[idx] += s * gps_grid_gradientX_new[idx]
                gps_grid_gradientY[idx] += s * gps_grid_gradientY_new[idx]
                s += 10
                missingGrad = torch.logical_and(self.grid != 0,
                                                torch.logical_and(gps_grid_gradientX == 0, gps_grid_gradientY == 0))

            self.gps_grid_gradientX = gps_grid_gradientX
            self.gps_grid_gradientY = gps_grid_gradientY

class EgoVehicle:
    """
    class for the ego vehicle which contains all information for the motion planning task
    """

    def __init__(self, xref0, xrefN, env, args, pdf0=None, name="egoVehicle", video=False):
        self.device = env.device
        self.xref0 = xref0
        self.xrefN = xrefN
        self.system = Car()
        self.name = name
        self.env = env
        self.args = args
        self.video = video
        if pdf0 is None:
            pdf0 = sample_pdf(self.system, args.mp_gaussians)
        self.initialize_predictor(pdf0)
        self.env.get_gradient()

    def initialize_predictor(self, pdf0):
        """
        sample initial states from the initial density distribution

        :param pdf0:    initial density distribution
        """
        self.model = self.load_predictor(self.system.DIM_X)
        if self.args.mp_sampling == 'random':
            xe0 = torch.rand(self.args.mp_sample_size, self.system.DIM_X, 1) * (
                    self.system.XE0_MAX - self.system.XE0_MIN) + self.system.XE0_MIN
        else:
            _, xe0 = get_mesh_sample_points(self.system, self.args)
            xe0 = xe0.unsqueeze(-1)
        rho0 = pdf0(xe0)
        mask = rho0 > 0
        self.xe0 = xe0[mask, :, :]
        self.rho0 = (rho0[mask] / rho0.sum()).reshape(-1, 1, 1)

    def load_predictor(self, dim_x):
        """
        load the density NN

        :param dim_x: dimensionaliy of the state
        :return: model of the NN
        """
        _, num_inputs = load_inputmap(dim_x, self.args)
        _, num_outputs = load_outputmap(dim_x)
        model, _ = load_nn(num_inputs, num_outputs, self.args, load_pretrained=True)
        model.eval()
        return model

    def predict_density(self, up, xref_traj, use_nn=True, xe0=None, rho0=None, compute_density=True):
        """
        predict the state and denisty trajectories

        :param up:              parameters of the reference trajectory
        :param xref_traj:       reference trajectory
        :param use_nn:          True if density NN is used for the predictions, otherwise LE
        :param xe0:             initial deviations of the reference trajectory
        :param rho0:            initial density values
        :param compute_density: True if density should be computed, otherwise just computation of the state trajectories
        :return: state and density trajectories
        """
        if xe0 is None:
            xe0 = self.xe0
        if rho0 is None:
            rho0 = self.rho0
            assert rho0.shape[0] == xe0.shape[0]

        if self.args.input_type == "discr10" and up.shape[2] != 10:
            N_sim = up.shape[2] * (self.args.N_sim_max // 10) + 1
            up = torch.cat((up, torch.zeros(up.shape[0], up.shape[1], 10 - up.shape[2])), dim=2, device=self.device)
        elif self.args.input_type == "discr5" and up.shape[2] != 5:
            N_sim = up.shape[2] * (self.args.N_sim_max // 5) + 1
            up = torch.cat((up, torch.zeros(up.shape[0], up.shape[1], 5 - up.shape[2])), dim=2, device=self.device)
        else:
            N_sim = self.args.N_sim

        if use_nn: # approximate with density NN
            xe_traj = torch.zeros(xe0.shape[0], xref_traj.shape[1], xref_traj.shape[2], device=self.device)
            rho_log_unnorm = torch.zeros(xe0.shape[0], 1, xref_traj.shape[2], device=self.device)
            t_vec = torch.arange(0, self.args.dt_sim * N_sim - 0.001, self.args.dt_sim * self.args.factor_pred)

            # 2. predict x(t) and rho(t) for times t
            for idx, t in enumerate(t_vec):
                xe_traj[:,:, [idx]], rho_log_unnorm[:, :, [idx]] = get_nn_prediction(self.model, xe0[:, :, 0], self.xref0[0, :, 0], t, up, self.args)
        else: # use LE
            uref_traj, _ = self.system.sample_uref_traj(self.args, up=up)
            xref_traj_long = self.system.compute_xref_traj(self.xref0, uref_traj, self.args)
            xe_traj_long, rho_log_unnorm = self.system.compute_density(xe0, xref_traj_long, uref_traj, self.args.dt_sim,
                                                   cutting=False, log_density=True, compute_density=True)
            xe_traj = xe_traj_long[:, :, ::self.args.factor_pred]

        if compute_density:
            rho_max, _ = rho_log_unnorm.max(dim=0)
            rho_unnorm = torch.exp(rho_log_unnorm - rho_max.unsqueeze(0)) * rho0.reshape(-1, 1, 1)
            rho_traj = rho_unnorm / rho_unnorm.sum(dim=0).unsqueeze(0)
            rho_traj = rho_traj #+ rho0.reshape(-1, 1, 1)
        else:
            rho_traj = None

        x_traj = xe_traj + xref_traj
        return x_traj, rho_traj

    def visualize_xref(self, xref_traj, uref_traj=None, show=True, save=False, include_date=True,
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

        # plt.imshow(self.env.grid_non_gaus.numpy()[:,:,0])
        # plt.show()

        if uref_traj is not None:
            plot_ref(xref_traj, uref_traj, 'Reference Trajectory', self.args, self.system, t=self.t_vec,
                     include_date=True)

        ego_dict = {"grid": self.env.gps_grid_non_gaus,
                    "building": self.env.grid_non_gaus,
                    "start": self.xref0,
                    "goal": self.xrefN,
                    "args": self.args}
        if x_traj is not None:
            mp_methods = ["sys", "ref"]
            mp_results = {"sys": {"x_traj": [x_traj]}, "ref": {"x_traj": [xref_traj]}}
        else:
            mp_methods = ["ref"]
            mp_results = {"ref": {"x_traj": [xref_traj]}}

        self.plot_traj(ego_dict, mp_results, mp_methods, self.args, folder=folder, traj_idx=0, animate=False,
                  include_density=False, name=name)

    def animate_traj(self, folder, xref_traj, x_traj=None, rho_traj=None):
        """
        plot the density and the states in the occupation map for each point in time

        :param folder:      name of the folder for saving the plots
        :param xref_traj:   reference state trajectory
        :param x_traj:      state trajectories
        :param rho_traj:    density trajectories
        """

        if x_traj is None:
            x_traj = xref_traj
        if rho_traj is None:
            rho_traj = torch.ones(x_traj.shape[0], 1, x_traj.shape[2]) / x_traj.shape[0]  # assume equal density

        # create colormap
        greys = cm.get_cmap('Greys')
        grey_col = greys(range(0, 256))
        greens = cm.get_cmap('Greens')
        green_col = greens(range(0, 256))
        blue = np.array([[0.212395, 0.359683, 0.55171, 1.]])
        # yellow = np.array([[0.993248, 0.906157, 0.143936, 1.      ]])
        colorarray = np.concatenate((grey_col[::2, :], green_col[::2, :], blue))
        cmap = ListedColormap(colorarray)

        if self.env.custom_cost == False:
            grid_env_sc = 127 * self.env.grid_non_gaus.clone().detach()
        else:
            grid_env_sc = torch.clamp(self.env.grid_non_gaus + self.env.gps_grid_non_gaus, 0, 1)
            grid_env_sc = 127 * grid_env_sc.clone().detach()

        for i in range(xref_traj.shape[2]):
            with torch.no_grad():
                # 3. compute marginalized density grid
                grid_pred = pred2grid(x_traj[:, :, [i]], rho_traj[:, :, [i]], self.args, return_gridpos=False)

            grid_pred_sc = 127 * torch.clamp(grid_pred/grid_pred.max(), 0, 1)
            grid_pred_sc[grid_pred_sc != 0] += 128
            grid_traj = traj2grid(xref_traj[:, :, :i + 1], self.args)
            grid_traj[grid_traj != 0] = 256
            grid_all = torch.clamp(grid_env_sc[:, :, i] + grid_traj + grid_pred_sc[:, :, 0], 0, 256)
            plot_grid(grid_all, self.args, name="iter%d" % i, cmap=cmap, show=False, save=True, folder=folder)

    def animate_trajs(self, folder, xref_traj, x_traj=None, rho_traj=None):
        """
        plot the density and the states in the occupation map for each point in time

        :param folder:      name of the folder for saving the plots
        :param xref_traj:   reference state trajectory
        :param x_traj:      state trajectories
        :param rho_traj:    density trajectories
        """

        if x_traj is None:
            x_traj = xref_traj
        if rho_traj is None:
            rho_traj = torch.ones(x_traj.shape[0], 1, x_traj.shape[2]) / x_traj.shape[0]  # assume equal density

        # create colormap
        greys = cm.get_cmap('Greys')
        grey_col = greys(range(0, 256))
        greens = cm.get_cmap('Greens')
        green_col = greens(range(0, 256))
        blue = np.array([[0.212395, 0.359683, 0.55171, 1.]])
        # yellow = np.array([[0.993248, 0.906157, 0.143936, 1.      ]])
        colorarray = np.concatenate((grey_col[::2, :], green_col[::2, :], blue))
        cmap = ListedColormap(colorarray)

        if self.env.custom_cost == False:
            grid_env_sc = 127 * self.env.grid_non_gaus.clone().detach()
        else:
            grid_env_sc = torch.clamp(self.env.grid_non_gaus + self.env.gps_grid_non_gaus, 0, 1)
            grid_env_sc = 127 * grid_env_sc.clone().detach()

        for i in range(xref_traj.shape[2]):
            with torch.no_grad():
                # 3. compute marginalized density grid
                grid_pred = pred2grid(x_traj[:, :, [i]], rho_traj[:, :, [i]], self.args, return_gridpos=False)

            grid_pred_sc = 127 * torch.clamp(grid_pred/grid_pred.max(), 0, 1)
            grid_pred_sc[grid_pred_sc != 0] += 128
            grid_traj = traj2grid(xref_traj[:, :, :i + 1], self.args)
            grid_traj[grid_traj != 0] = 256
            grid_all = torch.clamp(grid_env_sc[:, :, i] + grid_traj + grid_pred_sc[:, :, 0], 0, 256)
            plot_grid(grid_all, self.args, name="iter%d" % i, cmap=cmap, show=False, save=True, folder=folder)

    def set_start_grid(self):
        self.grid = pred2grid(self.xref0 + self.xe0, self.rho0, self.args)

    def plot_traj(self, ego_dict, mp_results, mp_methods, args, folder=None, traj_idx=None, animate=False,
                  include_density=False,
                  name=None):
        """
        function for plotting the reference trajectories planned by different motion planners in the occupation grid
        """

        gps_colormap = colors.LinearSegmentedColormap.from_list("", [ "black", "green", "yellow", "orange", "red", "darkred"])

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
        x_wide = (np.abs((args.environment_size[1] - args.environment_size[0])) / div) *2
        y_wide = (np.abs((args.environment_size[3] - args.environment_size[2])) / div) *2.5

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
            grid_all = ego_dict["grid"][:, :, [t_idx]]*40
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

            plt.imshow(grid_all, origin="lower", cmap = gps_colormap)
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
                (np.arange(0, args.environment_size[1] + 1, axis_tick), np.arange(-axis_tick, args.environment_size[0] - 1, -axis_tick)), 0)
            ticks_y = np.concatenate(
                (np.arange(0, args.environment_size[3] + 1, axis_tick), np.arange(-axis_tick, args.environment_size[2] - 1, -axis_tick)), 0)
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