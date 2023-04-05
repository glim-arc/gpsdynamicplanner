import hyperparams
import torch
import pickle
from gps_planning.utils import initialize_logging, get_cost_table, get_cost_increase
from gps_planning.MotionPlannerGrad import MotionPlannerGrad
import numpy as np
from mpnet_data.example_objects import create_mp_task
import os

if __name__ == '__main__':
    ### load hyperparameters
    args = hyperparams.parse_args()

    env_num = args.training_env_num
    traj_num = args.training_traj_num

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.device(args.device)

    ### load env and datasets
    for i in range(env_num):
        directory = "./mpnet_data/env/" + str(i)

        env_file = ""
        start_goal_file = ""

        files = os.listdir(directory)

        trajlist = []

        is_missing_env = False

        for j in range(traj_num):
            flag = False
            for file in files:
                if file[-3:] == "npy":
                    trajname = "traj_output_" + str(j)
                    if file.find(trajname) != -1:
                        flag = True

            if flag == False:
                is_missing_env = True

        if is_missing_env:
            print(i)
            continue

    print("end")