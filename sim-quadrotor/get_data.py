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
from quadrotor_mppi import get_one_mppi_trajectory







###################################################################################
# ------------------ EXPERT TRAJECTORIES ------------------------------------------
###################################################################################

def get_one_trajectory_lean(initial_condition_array, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs ):
    seedEverything()
    trajectory, control, valid_trajectory_indicator = get_one_mppi_trajectory(initial_condition_array.reshape((1,6)), x_goal, obstacle_list, map_boundaries, lambda_, **kwargs)
    return trajectory, control, valid_trajectory_indicator



def generate_trajectories_sequential(initial_condition_array_list, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs ):
    success_trajectories_list = []
    success_controls_list = []
    fail_trajectories_list = []
    fail_controls_list = []

    i = 0
    for initial_condition_array in initial_condition_array_list:
        i = i+1
        print ('  ', i, f' / {len(initial_condition_array_list)}' , end="\r")
        trajectory, control, valid_trajectory_indicator = get_one_trajectory_lean(initial_condition_array, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs )
        if valid_trajectory_indicator:
            success_trajectories_list.append(trajectory)
            success_controls_list.append(control)
        else:
            fail_trajectories_list.append(trajectory)
            fail_controls_list.append(control)
    return success_trajectories_list, success_controls_list, fail_trajectories_list, fail_controls_list




def get_expert_trajectory_wrapper(argument_tuple):
    initial_condition_array, x_goal, obstacle_list, map_boundaries, lambda_, kwargs = argument_tuple
    trajectory, control, valid_trajectory_indicator = torch.empty((1,6)), torch.empty((3)), 0
    try:
        trajectory, control, valid_trajectory_indicator = get_one_trajectory_lean(initial_condition_array, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs )
    except:
        print(f"except at run with {initial_condition_array}")
    return trajectory, control, valid_trajectory_indicator

N_THREADS = 10

def get_expert_trajectories_multiprocessing( initial_condition_array_list, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs ):
    iterable_arguments = [ (initial_condition_array_list[i], x_goal, obstacle_list, map_boundaries, lambda_, kwargs ) for i in range(len(initial_condition_array_list))]
    valid_trajectories_list = []
    valid_controls_list = []
    collision_trajectories_list = []
    collision_controls_list = []
    i = 0
    print(' ---------- Multiprocess starting -----------')
    with multiprocessing.Pool(processes=N_THREADS) as pool:
        for result in pool.imap_unordered( get_expert_trajectory_wrapper, iterable_arguments):
            i = i+1
            print ('  ', i, f' / {len(initial_condition_array_list)}' , end="\r")
            if result[2]:
                valid_trajectories_list.append(result[0])
                valid_controls_list.append(result[1])
            else:
                collision_trajectories_list.append(result[0])
                collision_controls_list.append(result[1])
    print(' ---------- Multiprocess finished -----------')
    return valid_trajectories_list, valid_controls_list, collision_trajectories_list, collision_controls_list








def get_dataset(random_seed, n_training_initial_conditions, path_to_save):

    x_goal = np.array([4.0, 2.5, 2.5])
    obstacle_list = np.array([ 
        [2.0, 1.5, 0.5, 0.5], [2.0, 3.5, 0.5, 0.5], \
        [2.0, 0.5, 2.5, 0.5], [2.0, 2.5, 2.5, 0.5], [2.0, 4.5, 2.5, 0.5],\
        [2.0, 1.5, 4.5, 0.5], [2.0, 3.5, 4.5, 0.5]
    ])
    map_boundaries = np.array([5, 5, 5])
    lambda_ = 1
    kwargs = {'goal_cost_weight': 1.5, 'obstacle_cost_weight': 1e1, 'obstacle_cost_exponential_weight': 1e1, 'control_cost_weight': 1e-2}

    


    # sample initial conditions
    seedEverything(random_seed)
    y_range = [0.5, 4.5]
    z_range = [ 0.5, 4.5]
    all_training_initial_conditions_array = sample_initial_conditions_array( y_range, z_range,  num_of_sample = n_training_initial_conditions)

    # generate expert trajectories
    bc_rollout_success_trajectories_list, bc_rollout_success_controls_list, bc_rollout_fail_trajectories_list, bc_rollout_fail_controls_list = \
                get_expert_trajectories_multiprocessing(all_training_initial_conditions_array, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs )
    

    print('expert success rate:' , len(bc_rollout_success_trajectories_list) / n_training_initial_conditions)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)



    # plot
    all_iterations_rollout_plot_str = path_to_save + f'/successfull_rollout_trajectories.png'
    # plot the trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for trajectory in bc_rollout_success_trajectories_list:
        ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2])
    # plot the goal
    ax.scatter(x_goal[0], x_goal[1], x_goal[2])
    # plot the obstacles
    for obs in obstacle_list:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_obs = obs[0] + obs[3] * np.outer(np.cos(u), np.sin(v))
        y_obs = obs[1] + obs[3] * np.outer(np.sin(u), np.sin(v))
        z_obs = obs[2] + obs[3] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_obs, y_obs, z_obs, color='b', alpha=0.3)
    # set x, y, z limits
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    # set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig( all_iterations_rollout_plot_str, dpi=300)




    # plot
    all_iterations_rollout_plot_str = path_to_save + f'/failure_rollout_trajectories.png'
    # plot the trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for trajectory in bc_rollout_fail_trajectories_list:
        ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2])
    # plot the goal
    ax.scatter(x_goal[0], x_goal[1], x_goal[2])
    # plot the obstacles
    for obs in obstacle_list:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_obs = obs[0] + obs[3] * np.outer(np.cos(u), np.sin(v))
        y_obs = obs[1] + obs[3] * np.outer(np.sin(u), np.sin(v))
        z_obs = obs[2] + obs[3] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_obs, y_obs, z_obs, color='b', alpha=0.3)
    # set x, y, z limits
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    # set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig( all_iterations_rollout_plot_str, dpi=300)

    x_trajectories_list = bc_rollout_success_trajectories_list
    controls_list = bc_rollout_success_controls_list

    # save the data
    data_dict = {
        'x_trajectories_list': x_trajectories_list,
        'controls_list': controls_list
    }

    pickle.dump(data_dict, open(path_to_save + "/data_0.pkl", "wb"))


if __name__ == "__main__":
    random_seed = 0
    n_training_initial_conditions = 100
    path_to_save = 'sim-quadrotor/data'
    get_dataset(random_seed, n_training_initial_conditions, path_to_save)
