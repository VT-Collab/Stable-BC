import numpy as np
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import random
import os, sys
import torch
import math
from utils import *
import time





roll_max=0.4 # radians
pitch_max=0.4
f_g_diff_max=1.0 # max difference between thrust and gravity

noise_std = np.array([f_g_diff_max, roll_max, pitch_max]) * 0.1


if __name__ == "__main__":
    
    
    data_dict = pickle.load(open("sim10_quadrotor/data/data_0.pkl", "rb"))
    x_trajectories_list = data_dict['x_trajectories_list']
    controls_list = data_dict['controls_list']

    for i in range(len(x_trajectories_list)):
        x_trajectory = x_trajectories_list[i]
        controls = controls_list[i]

        # example ccil data augmentation for another system
        # state1 = np.copy(state)
        # state1[:2] += np.random.normal(0, 0.2, 2)
        # action1 = (state[:2] + u1) - state1[:2]
        # dataset.append(list(state1) + list(action1))

        # generate ccil data augmentation for quadrotor
        for j in range(len(x_trajectory)):
            state = x_trajectory[j]
            control = controls[j]
            # state1 = np.copy(state)
            # state1[3:6] += np.random.normal(0, noise_std, 3)
            # action1 = (state[3:6] + control) - state1[3:6]
            # x_trajectory[j] = state1
            # controls[j] = action1
        


    a = 1