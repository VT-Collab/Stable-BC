import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
from utils import *
import sys, time, os
from datetime import datetime

from scipy.stats import ttest_ind




if __name__ == "__main__":

    metric = 'success_rate'

    n_dems_list = [5, 10, 20, 40, 60, 100] # [ i for i in range( 1, 10 )]
    random_seed_list = [ i for i in range( 10 )]
    type_list = [0, 1, 2]

    n_dems_list_str = str(n_dems_list).replace(' ', '')
    random_seed_list_str = str(random_seed_list).replace(' ', '')
    type_list_str = str(type_list).replace(' ', '')

    test_name = 'training_region'  + '_noise0.1' 


    base_path = 'first_results_0.001lr_1000epoch/lamda_0.0001'

    # base_path_list = [
    #     'sim10_quadrotor/first_results_0.001lr_1000epoch/lamda_0.01',
    #     'sim10_quadrotor/first_results_0.001lr_1000epoch/lamda_0.0001',
    #     'sim10_quadrotor/first_results_0.001lr_1000epoch/lamda_1.0',

    #     'sim10_quadrotor/first_results_0.0001lr_1000epoch/lamda_0.01',
    #     'sim10_quadrotor/first_results_0.0001lr_1000epoch/lamda_0.0001',
    #     'sim10_quadrotor/first_results_0.0001lr_1000epoch/lamda_1.0',

    #     'sim10_quadrotor/first_results_0.0001lr_2000epoch/lamda_0.01',
    #     'sim10_quadrotor/first_results_0.0001lr_2000epoch/lamda_0.0001',
    #     'sim10_quadrotor/first_results_0.0001lr_2000epoch/lamda_1.0'
    # ]

    mean_success_rate_vs_num_dems = np.empty( ( len(type_list), len(n_dems_list)) )
    ci95_success_rate_vs_num_dems = np.empty( ( len(type_list), len(n_dems_list)) ) 
    all_success_rate_vs_num_dems = np.empty( ( len(type_list), len(n_dems_list), len(random_seed_list)) )
    for type_i, type in enumerate(type_list):
        for i, n_dems in enumerate(n_dems_list):
            success_rate_per_seed_array = np.empty(len(random_seed_list) )
            for seed in random_seed_list:
            
                models_path= base_path + f'/{n_dems}dems/{seed}'
                result_save_path = models_path + f'/{test_name}'

                # load the result dict
                print(pickle.DEFAULT_PROTOCOL)
                result_dict = pickle.load(open( result_save_path + f'/im_test_results_{type}.pkl', 'rb'))
                success_rate = result_dict[ metric ]

                success_rate_per_seed_array[seed] = success_rate
            all_success_rate_vs_num_dems[type_i][i] = success_rate_per_seed_array
            mean_success_rate_vs_num_dems[type_i][i] = np.mean( success_rate_per_seed_array )
            ci95_success_rate_vs_num_dems[type_i][i] = 1.96 * np.std( success_rate_per_seed_array ) / np.sqrt( len(random_seed_list) )
    

    if len(n_dems_list) > 1:
        # plot the mean_success_rate_vs_num_dems
        for type_i, type in enumerate(type_list):
            plt.plot( n_dems_list, mean_success_rate_vs_num_dems[type_i], '.-', label=f'type{type}' )
            plt.fill_between( n_dems_list, mean_success_rate_vs_num_dems[type_i] - ci95_success_rate_vs_num_dems[type_i], mean_success_rate_vs_num_dems[type_i] + ci95_success_rate_vs_num_dems[type_i], alpha=0.2 )
        plt.xticks( n_dems_list )
        plt.xlabel('Number of demonstration trajectories')

    else:
        # plot a histogram 
        for type_i, type in enumerate(type_list):
            plt.bar( type_i, mean_success_rate_vs_num_dems[type_i], yerr=ci95_success_rate_vs_num_dems[type_i], label=f'type{type}' )
        plt.xticks( type_list )
        plt.xlabel('Type of imitation')

        # get the significance test for type 1 bigger than type 0
        greater_p_value = ttest_ind( all_success_rate_vs_num_dems[1][0], all_success_rate_vs_num_dems[0][0], alternative='greater' )


    plt.legend()
    plt.ylabel(metric)
    plt.title( test_name)

    plt_save_path = base_path + f'/{metric}_vs_{test_name}_{type_list_str}_{n_dems_list_str}_{len(random_seed_list)}.png'

    plt.savefig( plt_save_path, dpi=200 )

    if len(n_dems_list) == 1:
        print(f'{metric} for {test_name} with {n_dems_list[0]} demonstration trajectories: {mean_success_rate_vs_num_dems} +- {ci95_success_rate_vs_num_dems}')
        print(f'p_value: {greater_p_value}')

    a = 1


