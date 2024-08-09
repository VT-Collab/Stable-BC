
import numpy as np
import matplotlib.pyplot as plt
import math

A_G = 9.81
HORIZON = 50
NUM_SAMPLES = 5000

roll_max=0.4 # radians
pitch_max=0.4
f_g_diff_max=1.0 # max difference between thrust and gravity

PERTURBATION_STD = np.array([ f_g_diff_max / 2, roll_max / 2, pitch_max / 2 ])


ROLL_BOUND = [-roll_max, roll_max]
PITCH_BOUND = [-pitch_max, pitch_max]
THRUST_BOUND = [A_G - f_g_diff_max, A_G + f_g_diff_max]

# 100Hz
DT = 0.01 # seconds

INITIAL_NOMINAL_CONTROL = np.broadcast_to(np.array([A_G, 0, 0]), (1, HORIZON, 3))

MAP_BOUNDARIES = np.array([5, 5, 5]) # meters

def x_dot(x, u):
    x_dot = np.zeros_like(x)
    x_dot[...,0] = x[...,3]
    x_dot[...,1] = x[...,4]
    x_dot[...,2] = x[...,5]
    x_dot[...,3] = A_G * np.tan(u[...,2])
    x_dot[...,4] = -A_G * np.tan(u[...,1])
    x_dot[...,5] = u[...,0] - A_G
    return x_dot

def get_next_step_state(x, u, dt):
    return x + x_dot(x, u) * dt

def get_cost(x, u, dist_to_goal, dist_to_obs, dist_to_map_boundaries, **kwargs):
    '''
    x: (number_of_samples, horizon, state_dim) (px, py, pz, vx, vy, vz)
    u: (number_of_samples, horizon, control_dim) ( thrust, roll, pitch )
    x_goal: (px, py, pz)
    obstacle_list: (number of obstacles, 4) (px, py, pz, radius)
    map_boundaries: (px_bound, py_bound, pz_bound)
    '''

    min_dist_to_obstacles = np.minimum(dist_to_obs, dist_to_map_boundaries)

    # goal cost
    goal_cost = np.sum(dist_to_goal * kwargs['goal_cost_weight'], axis=-1)

    # obstacle cost
    obstacle_cost = np.sum(np.exp(-min_dist_to_obstacles * kwargs['obstacle_cost_exponential_weight']) * kwargs['obstacle_cost_weight'], axis=-1)

    # control cost
    control_cost = np.sum(np.sum( ((u - np.array([A_G, 0, 0]))/ np.array([f_g_diff_max, roll_max, pitch_max]) ) **2, axis=-1) * kwargs['control_cost_weight'], axis=-1)   

    # total cost
    total_cost = goal_cost + obstacle_cost + control_cost
    return total_cost


def sample_control_perturbations( num_samples, std_dev, horizon, control_dim):
    perturbations = np.concatenate((np.random.normal(0, std_dev, (num_samples-1, horizon, control_dim)), \
                                   np.zeros((1, horizon, control_dim)))) # include the nominal control in the perturbations
    return perturbations

def sample_perturbed_controls(nominal_control ):
    perturbed_controls = nominal_control + sample_control_perturbations(NUM_SAMPLES, PERTURBATION_STD, HORIZON, nominal_control.shape[-1])
    perturbed_controls[...,0] = np.clip(perturbed_controls[...,0], THRUST_BOUND[0], THRUST_BOUND[1])
    perturbed_controls[...,1] = np.clip(perturbed_controls[...,1], PITCH_BOUND[0], PITCH_BOUND[1])
    perturbed_controls[...,2] = np.clip(perturbed_controls[...,2], ROLL_BOUND[0], ROLL_BOUND[1])
    return perturbed_controls


def update_nominal_control(perturbed_controls, cost_array, lambda_):
    '''
    Update the nominal control using the perturbed controls and the cost array
    :param perturbed_controls: (num_samples, horizon, control_dim)
    :param cost_array: (num_samples) 
    :param lambda_: The lambda value
    '''
    weights = np.exp(-cost_array / lambda_)
    weights /= np.sum(weights)
    weighted_perturbed_controls = perturbed_controls * weights[:, np.newaxis, np.newaxis]
    updated_nominal_control = np.sum(weighted_perturbed_controls, axis=0, keepdims=True)
    return updated_nominal_control


def get_dist_to_goal(x, x_goal):
    '''
    x_goal: (px, py, pz)
    '''
    return np.linalg.norm(x[...,:3] - x_goal, axis=-1)

def get_dist_to_obs(x, obstacle_list):
    '''
    obs: (px, py, pz, radius)
    '''
    min_dist = np.full(x[...,0].shape, np.inf)
    for obs in obstacle_list:
        dist = np.linalg.norm(x[...,:3] - obs[:3], axis=-1) - obs[3]
        min_dist = np.minimum(min_dist, dist)
    return min_dist

def get_dist_to_map_boundaries(x, map_boundaries):
    return np.min([x[..., 0], x[..., 1], x[..., 2], map_boundaries[0] - x[..., 0], map_boundaries[1] - x[..., 1], map_boundaries[2] - x[..., 2]], axis=0)

def collision_check(x, obstacle_list, map_boundaries):
    min_dist = np.minimum(get_dist_to_obs(x, obstacle_list), get_dist_to_map_boundaries(x, map_boundaries))
    return np.any(min_dist < 0)

def mppi(x, nominal_control, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs):

    # sample controls
    perturbed_controls = sample_perturbed_controls(nominal_control)

    # rollout perturbed controls
    x_rollouts = np.zeros((NUM_SAMPLES, HORIZON+1, 6))
    x_rollouts[:,0] = x[0]
    for i in range(1, HORIZON+1):
        x_rollouts[:,i] = get_next_step_state(x_rollouts[:,i-1], perturbed_controls[:,i-1], DT)

    # calculate cost
    dist_to_goal = get_dist_to_goal(x_rollouts, x_goal)
    dist_to_obs = get_dist_to_obs(x_rollouts, obstacle_list)
    dist_to_map_boundaries = get_dist_to_map_boundaries(x_rollouts, map_boundaries)
    cost_array = get_cost(x_rollouts, perturbed_controls, dist_to_goal, dist_to_obs, dist_to_map_boundaries, **kwargs)

    # update nominal control
    nominal_control = update_nominal_control(perturbed_controls, cost_array, lambda_)
    return nominal_control

def get_one_mppi_trajectory(x, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs):
    goal_distance = get_dist_to_goal(x, x_goal)
    nominal_control = INITIAL_NOMINAL_CONTROL
    trajectory_list = []
    control_list = []
    valid_trajectory_indicator = True
    state = x

    while goal_distance > 0.5 and valid_trajectory_indicator:
        if math.isnan(nominal_control[0,0,0]):
            break
        trajectory_list.append(state)
        nominal_control = mppi(state, nominal_control, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs)
        control2apply = nominal_control[:, 0]
        control_list.append(control2apply)
        state = get_next_step_state(state, control2apply, DT)
        goal_distance = get_dist_to_goal(state, x_goal)
        valid_trajectory_indicator = not collision_check(state, obstacle_list, map_boundaries) and len(trajectory_list) < 1500
    return np.concatenate(trajectory_list), np.concatenate(control_list), valid_trajectory_indicator




def get_one_mppi_trajectory_w_normal_noise(x, x_goal, obstacle_list, map_boundaries, lambda_, covariance, **kwargs):
    goal_distance = get_dist_to_goal(x, x_goal)
    nominal_control = INITIAL_NOMINAL_CONTROL
    trajectory_list = []
    control_list = []
    valid_trajectory_indicator = True
    state = x

    while goal_distance > 0.5 and valid_trajectory_indicator:
        if math.isnan(nominal_control[0,0,0]):
            break
        trajectory_list.append(state)
        nominal_control = mppi(state, nominal_control, x_goal, obstacle_list, map_boundaries, lambda_, **kwargs)
        expert_control = nominal_control[:, 0]
        control2apply = expert_control + np.random.multivariate_normal(np.zeros(3), covariance)
        control_list.append(expert_control)
        state = get_next_step_state(state, control2apply, DT)
        goal_distance = get_dist_to_goal(state, x_goal)
        valid_trajectory_indicator = not collision_check(state, obstacle_list, map_boundaries) and len(trajectory_list) < 1500
    return np.concatenate(trajectory_list), np.concatenate(control_list), valid_trajectory_indicator



