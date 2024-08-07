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








if __name__ == "__main__":
    
    # load sim10_quadrotor/data/data.pkl
    data_dict = pickle.load(open("sim10_quadrotor/data/data_0.pkl", "rb"))
    x_trajectories_list = data_dict['x_trajectories_list']
    controls_list = data_dict['controls_list']

    all_states = np.concatenate(x_trajectories_list, axis=0)
    all_controls = np.concatenate(controls_list, axis=0)
    print(all_states.shape, all_controls.shape)

    for i in range(6):
        plt.hist(all_states[:,i])
        plt.savefig(f"sim10_quadrotor/data/histogram_state_{i}.png")
        plt.close()
    
    for i in range(3):
        plt.hist(all_controls[:,i])
        plt.savefig(f"sim10_quadrotor/data/histogram_control_{i}.png")
        plt.close()
    

    a = 1