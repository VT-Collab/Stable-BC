
from quadrotor_mppi import get_one_mppi_trajectory_w_normal_noise
from test_model import get_imitation_agent_action_sequential
from train_model import StateImitationDataset
from models import MyModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys
import multiprocessing
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from quadrotor_mppi import roll_max, pitch_max, A_G, f_g_diff_max


def get_one_trajectory_lean(initial_condition_array, x_goal, obstacle_list, map_boundaries, lambda_, covariance, **kwargs ):
    seedEverything()
    trajectory, control, valid_trajectory_indicator = get_one_mppi_trajectory_w_normal_noise(initial_condition_array.reshape((1,6)), x_goal, obstacle_list, map_boundaries, lambda_, covariance, **kwargs)
    return trajectory, control, valid_trajectory_indicator


def generate_trajectories_sequential(initial_condition_array_list, x_goal, obstacle_list, map_boundaries, lambda_, covariance, **kwargs ):
    success_trajectories_list = []
    success_controls_list = []
    fail_trajectories_list = []
    fail_controls_list = []

    i = 0
    for initial_condition_array in initial_condition_array_list:
        i = i+1
        print ('  ', i, f' / {len(initial_condition_array_list)}' , end="\r")
        trajectory, control, valid_trajectory_indicator = get_one_trajectory_lean(initial_condition_array, x_goal, obstacle_list, map_boundaries, lambda_, covariance, **kwargs )
        if valid_trajectory_indicator:
            success_trajectories_list.append(trajectory)
            success_controls_list.append(control)
        else:
            fail_trajectories_list.append(trajectory)
            fail_controls_list.append(control)
    return success_trajectories_list, success_controls_list, fail_trajectories_list, fail_controls_list




def get_expert_trajectory_wrapper(argument_tuple):
    initial_condition_array, x_goal, obstacle_list, map_boundaries, lambda_, covariance, kwargs = argument_tuple
    trajectory, control, valid_trajectory_indicator = torch.empty((1,6)), torch.empty((3)), 0
    try:
        trajectory, control, valid_trajectory_indicator = get_one_trajectory_lean(initial_condition_array, x_goal, obstacle_list, map_boundaries, lambda_, covariance, **kwargs )
    except:
        print(f"except at run with {initial_condition_array}")
    return trajectory, control, valid_trajectory_indicator

N_THREADS = 10

def get_expert_trajectories_multiprocessing( initial_condition_array_list, x_goal, obstacle_list, map_boundaries, lambda_, covariance, **kwargs ):
    iterable_arguments = [ (initial_condition_array_list[i], x_goal, obstacle_list, map_boundaries, lambda_, covariance, kwargs ) for i in range(len(initial_condition_array_list))]
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



def iterative_il( path_to_save, n_demonstrations_list, random_seed=0 ):



    n_training_initial_conditions = n_demonstrations_list[-1]



    x_goal = np.array([4.0, 2.5, 2.5])
    obstacle_list = np.array([ 
        [2.0, 1.5, 0.5, 0.5], [2.0, 3.5, 0.5, 0.5], \
        [2.0, 0.5, 2.5, 0.5], [2.0, 2.5, 2.5, 0.5], [2.0, 4.5, 2.5, 0.5],\
        [2.0, 1.5, 4.5, 0.5], [2.0, 3.5, 4.5, 0.5]
    ])
    map_boundaries = np.array([5, 5, 5])
    lambda_ = 1
    kwargs = {'goal_cost_weight': 1.5, 'obstacle_cost_weight': 1e1, 'obstacle_cost_exponential_weight': 1e1, 'control_cost_weight': 1e-2}


    ROLL_BOUND = [-roll_max, roll_max]
    PITCH_BOUND = [-pitch_max, pitch_max]
    THRUST_BOUND = [A_G - f_g_diff_max, A_G + f_g_diff_max]
    control_bounds = torch.tensor([ THRUST_BOUND, ROLL_BOUND, PITCH_BOUND]).T
    


    # sample initial conditions
    seedEverything(random_seed)
    y_range = [0.5, 4.5]
    z_range = [ 0.5, 4.5]
    all_training_initial_conditions_array = sample_initial_conditions_array( y_range, z_range,  num_of_sample = n_training_initial_conditions)


    visited_states = np.empty((0, 6), dtype=np.float32)
    expert_actions = np.empty( (0, 3), dtype=np.float32)
    rollout_trajectories_list = []


    previous_n_demonstrations = 0
    covariance = np.zeros((3,3))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for iteration, n_demonstrations in enumerate(n_demonstrations_list):

        path_to_save_this_iteration = path_to_save + f'/{n_demonstrations}dems/{random_seed}'


        n_demonstrations_this_iteration = n_demonstrations - previous_n_demonstrations
        # get the training initial conditions for this iteration
        training_initial_conditions_array_this_iteration = all_training_initial_conditions_array[previous_n_demonstrations:previous_n_demonstrations + n_demonstrations_this_iteration]

        previous_n_demonstrations = n_demonstrations

        # get the training trajectories for this iteration
        rollout_success_trajectories_list_this_iter, rollout_success_controls_list_this_iter, rollout_fail_trajectories_list_this_iter, rollout_fail_controls_list_this_iter =\
        get_expert_trajectories_multiprocessing( training_initial_conditions_array_this_iteration, x_goal, obstacle_list, map_boundaries, lambda_, covariance, **kwargs )

        
        rollout_trajectories_list_this_iter = [ *rollout_success_trajectories_list_this_iter, *rollout_fail_trajectories_list_this_iter ]
        rollout_controls_list_this_iter = [ *rollout_success_controls_list_this_iter, *rollout_fail_controls_list_this_iter ]

        # collect all rollout trajectories
        rollout_trajectories_list = [ *rollout_trajectories_list, *rollout_trajectories_list_this_iter ]

        states_this_iteration = np.concatenate( rollout_trajectories_list_this_iter )

        actions_this_iteration = np.concatenate( rollout_controls_list_this_iter )
        
        # add the new data to the training data
        visited_states = np.concatenate( (visited_states, states_this_iteration), axis=0)
        expert_actions = np.concatenate( (expert_actions, actions_this_iteration), axis=0)

        # save the data        

        # plot training rollouts
        all_iterations_rollout_plot_str = path_to_save_this_iteration + f'/dart_rollout_trajectories_{n_demonstrations}dems.png'
        # plot the trajectories
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for trajectory in rollout_trajectories_list_this_iter:
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



        EPOCH = 1000
        LR = 0.001

        model = MyModel()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        # prepare training data
        valid_size = 0.20
        train_states, valid_states, train_actions,  test_actions = train_test_split( visited_states, expert_actions, test_size = valid_size, shuffle=True )
  
        train_dataset = StateImitationDataset(train_states, train_actions)
        test_dataset = StateImitationDataset(valid_states, test_actions)

        batch_size = int( len(train_dataset) / 10)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


        for epoch in range(EPOCH):

            # validation
            model.eval()
            total_test_loss = 0
            for i, data in enumerate(valid_dataloader):
                states = data[0]
                actions = data[1]
                states = states.to(device)
                actions = actions.to(device)
                outputs = model(states)
                test_loss = model.loss_func(actions, outputs)
                total_test_loss += test_loss.item()
            validdation_loss_per_sample = total_test_loss / len(valid_dataloader)
            print(f"Epoch {epoch} Test Loss: { validdation_loss_per_sample }")

            model.train()
            total_loss = 0
            
            train_bar = tqdm(train_dataloader, position = 0, leave = True)

            for batch, data in enumerate(train_bar):
                states = data[0]
                actions = data[1]
                states = states.to(device)
                actions = actions.to(device)

                # get mse loss
                loss = model.loss_func(actions, model(states))

                # update model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                total_loss += loss.item()
                train_bar.set_description('Train iteration (epoch {}): [{}] Loss: {:.4f}'.format(epoch, batch,
                                total_loss / (batch + 1)))

            
        
        # get new cov estimation
        model = model.cpu()

        cov = np.zeros((3,3))
        for i in range( len(rollout_trajectories_list_this_iter) ):
            sup_states = rollout_trajectories_list_this_iter[i]
            sup_actions = rollout_controls_list_this_iter[i]
            sup_states_this_iteration_list = []
            for i in range( len( sup_states ) ):
                sup_states_this_iteration_list.append( sup_states[i]) 
            lnr_actions_list = get_imitation_agent_action_sequential(model, sup_states_this_iteration_list, control_bounds, 'cpu' )
            lnr_actions = torch.concat( lnr_actions_list ).cpu().numpy()
            diff = sup_actions - lnr_actions
            cov = cov + np.dot(diff.T, diff) / float(len(sup_actions))
        cov = cov / float(len(rollout_trajectories_list_this_iter))
        covariance = cov

        # save the model
        if not os.path.exists(path_to_save_this_iteration):
            os.makedirs(path_to_save_this_iteration)
        savename = f"im_model2.pt"
        torch.save(model.state_dict(), path_to_save_this_iteration + '/' + savename)



if __name__ == '__main__':
    

    random_seed_list = [ i for i in range( 10 )]
    n_demonstrations_list = [5, 10, 20, 40, 60, 100]


    path_to_save = 'sim-quadrotor/results_0.001lr_1000epoch/lamda_0.0001'

    for random_seed in random_seed_list:
        iterative_il( path_to_save, n_demonstrations_list, random_seed )
