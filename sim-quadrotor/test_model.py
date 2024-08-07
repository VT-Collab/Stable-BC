import torch
import numpy as np
from models import MyModel
import matplotlib.pyplot as plt
import csv
import pickle
from utils import *
import sys, time, os
from datetime import datetime

from quadrotor_mppi import *










########################################################
# ------------------  GET IMITATION AGENT TRAJECTORIES  -----------------------
########################################################

def get_imitation_agent_action(imitating_agent, state_array, control_bounds, device ):
    upper_bound = control_bounds[1]
    lower_bound = control_bounds[0]
    action = imitating_agent.get_action( state_array, device)
    action = torch.minimum( action, upper_bound )
    action = torch.maximum( action, lower_bound )
    return action.unsqueeze(0)

def get_imitation_agent_action_sequential(imitating_agent, state_array_list, control_bounds, device ):
    action_list = []
    for state_array in state_array_list:
        action = get_imitation_agent_action(imitating_agent, state_array, control_bounds, device )
        action_list.append(action)
    return action_list


def get_imitating_agent_trajectory(imitating_agent, state_array, x_goal, obstacle_list, map_boundaries, control_bounds, device = 'cpu'):

    # start every trajectory with the same seed
    seedEverything()
    state_array = state_array.reshape(1,6)
    distance2goal = get_dist_to_goal(state_array, x_goal)
    trajectory_list = []
    control_list = []
    valid_trajectory_indicator = 1

    while distance2goal > 0.5 and valid_trajectory_indicator:
        
        action = get_imitation_agent_action(imitating_agent, state_array, control_bounds, device )
        
        action = action.detach().numpy()
        control_list.append(action)
        trajectory_list.append(state_array)
        state_array = get_next_step_state(state_array, action, DT)
        distance2goal = get_dist_to_goal(state_array, x_goal)
        
        if collision_check(state_array, obstacle_list, map_boundaries) or len(trajectory_list) > 1500:
            valid_trajectory_indicator = 0
        
    trajectory = np.concatenate(trajectory_list)
    control = np.concatenate(control_list)
    return trajectory, control, valid_trajectory_indicator



def generate_imitating_agent_trajectories_sequential(initial_condition_array_list, imitating_agent, x_goal, obstacle_list, map_boundaries, control_bounds, device = 'cpu' ):
    success_trajectories_list = []
    success_controls_list = []
    fail_trajectories_list = []
    fail_controls_list = []
    for i in range(len(initial_condition_array_list)):

        print ('---', i + 1, f' / {len(initial_condition_array_list)}' , end="\r")
        trajectory, control, valid_trajectory_indicator = get_imitating_agent_trajectory(imitating_agent, initial_condition_array_list[i], x_goal, obstacle_list, map_boundaries, control_bounds, device )
        if valid_trajectory_indicator:
            success_trajectories_list.append(trajectory)
            success_controls_list.append(control)
        else:
            fail_trajectories_list.append(trajectory)
            fail_controls_list.append(control)
    return success_trajectories_list, success_controls_list, fail_trajectories_list, fail_controls_list
    



def get_imitating_agent_trajectory_action_noise(imitating_agent, state_array, x_goal, obstacle_list, map_boundaries, control_bounds, device = 'cpu'):

    # start every trajectory with the same seed
    seedEverything()
    state_array = state_array.reshape(1,6)
    distance2goal = get_dist_to_goal(state_array, x_goal)
    trajectory_list = []
    control_list = []
    valid_trajectory_indicator = 1

    noise_std = np.array([f_g_diff_max, roll_max, pitch_max]) * 0.1

    while distance2goal > 0.5 and valid_trajectory_indicator:
        
        action = get_imitation_agent_action(imitating_agent, state_array, control_bounds, device )
        
        action = action.detach().numpy()
        control_list.append(action)
        trajectory_list.append(state_array)
        applied_action = action + np.random.normal( 0, noise_std )
        state_array = get_next_step_state(state_array, applied_action, DT)
        distance2goal = get_dist_to_goal(state_array, x_goal)
        
        if collision_check(state_array, obstacle_list, map_boundaries) or len(trajectory_list) > 1500:
            valid_trajectory_indicator = 0
        
    trajectory = np.concatenate(trajectory_list)
    control = np.concatenate(control_list)
    return trajectory, control, valid_trajectory_indicator


def generate_imitating_agent_trajectories_action_noise_sequential(initial_condition_array_list, imitating_agent, x_goal, obstacle_list, map_boundaries, control_bounds, device = 'cpu' ):
    success_trajectories_list = []
    success_controls_list = []
    fail_trajectories_list = []
    fail_controls_list = []
    for i in range(len(initial_condition_array_list)):

        print ('---', i + 1, f' / {len(initial_condition_array_list)}' , end="\r")
        trajectory, control, valid_trajectory_indicator = get_imitating_agent_trajectory_action_noise(imitating_agent, initial_condition_array_list[i], x_goal, obstacle_list, map_boundaries, control_bounds, device )
        if valid_trajectory_indicator:
            success_trajectories_list.append(trajectory)
            success_controls_list.append(control)
        else:
            fail_trajectories_list.append(trajectory)
            fail_controls_list.append(control)
    return success_trajectories_list, success_controls_list, fail_trajectories_list, fail_controls_list
    





def test_imitation_agent( num_dems, type, random_seed, test_name, base_path = None):
    # check to see whether training actually changed the system stability

    n_test_trajs = 100


    x_goal = np.array([4.0, 2.5, 2.5])
    obstacle_list = np.array([ 
        [2.0, 1.5, 0.5, 0.5], [2.0, 3.5, 0.5, 0.5], \
        [2.0, 0.5, 2.5, 0.5], [2.0, 2.5, 2.5, 0.5], [2.0, 4.5, 2.5, 0.5],\
        [2.0, 1.5, 4.5, 0.5], [2.0, 3.5, 4.5, 0.5]
    ])
    map_boundaries = np.array([5, 5, 5])


    ROLL_BOUND = [-roll_max, roll_max]
    PITCH_BOUND = [-pitch_max, pitch_max]
    THRUST_BOUND = [A_G - f_g_diff_max, A_G + f_g_diff_max]
    control_bounds = torch.tensor([ THRUST_BOUND, ROLL_BOUND, PITCH_BOUND]).T




    models_path= base_path + f'/{num_dems}dems/{random_seed}'



    model_path = models_path + f"/im_model{type}.pt"

    im_model = MyModel()
    im_model.load_state_dict(torch.load( model_path ))


    # test_name = 'unsafe_hard_states_test'

    if test_name == 'training_region':
        # sample initial conditions
        seedEverything(random_seed)
        y_range = [0.5, 4.5]
        z_range = [ 0.5, 4.5]
        test_initial_conditions_array = sample_initial_conditions_array( y_range, z_range,  num_of_sample = n_test_trajs)


    im_success_trajectories_list, im_success_controls_list, im_fail_trajectories_list, im_fail_controls_list = \
        generate_imitating_agent_trajectories_action_noise_sequential(test_initial_conditions_array, im_model, x_goal, obstacle_list, map_boundaries, control_bounds, device = 'cpu' )
    
    result_save_path = models_path + f'/{test_name}_noise0.1'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    
    
    # write into a text file
    with open( result_save_path + f'/im_test_results_{type}.txt', 'a') as f:
        f.write('\n---------------------------------------- \n')
        # write the date and time
        f.write(f'\n {datetime.now()} \n')
        f.write(f'\n {type} \n')
        f.write(f'\n {random_seed} \n')
        f.write(f'\n {num_dems} \n')
        success_rate = len(im_success_trajectories_list) / n_test_trajs
        f.write(f'\n success rate: {success_rate} \n')

    # save the test results
    pickle.dump( {'success_trajectories_list': im_success_trajectories_list, 'success_controls_list': im_success_controls_list, 
                'fail_trajectories_list': im_fail_trajectories_list, 'fail_controls_list': im_fail_controls_list, 'success_rate': len(im_success_trajectories_list) / n_test_trajs}, 
                open(result_save_path + f'/im_test_results_{type}.pkl', 'wb'))

    # plot
    all_iterations_rollout_plot_str = result_save_path + f'/{type}_im_success_rollout_trajectories.png'
    # plot the trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for trajectory in im_success_trajectories_list:
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
    all_iterations_rollout_plot_str = result_save_path + f'/{type}im_fail_rollout_trajectories.png'
    # plot the trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for trajectory in im_fail_trajectories_list:
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

    a = 1
    




if __name__ == "__main__":
    random_seed = 0
    num_dems = 11
    type = 0
    test_name = 'training_region'
    # test_imitation_agent(num_dems, type, random_seed, test_name)
    type = 1

    base_path = 'sim10_quadrotor/first_results_0.001lr_1000epoch/lamda_0.01'
    test_imitation_agent(num_dems, type, random_seed, test_name, base_path)

    a = 1