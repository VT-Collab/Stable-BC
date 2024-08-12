from test_model import test_imitation_agent
import time
import os

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    n_dems_list = [ 5, 10, 20, 40, 60, 100 ] # [ 5, 10, 15, 20, 25 ] # [ i for i in range( 1, 10 )]
    random_seed_list = [ i for i in range( 10 )]
    test_name_list = [ 'training_region' ]

    base_path_list = [ 'sim-quadrotor/results_0.001lr_1000epoch/lamda_0.0001' ]

    for base_path in base_path_list:
        for n_dems in n_dems_list:
            for test_name in test_name_list:
                for seed in random_seed_list:
                
                    start_time = time.time()
                    
                    test_imitation_agent(n_dems, 2, seed, test_name, base_path)
                    test_imitation_agent(n_dems, 1, seed, test_name, base_path)
                    test_imitation_agent(n_dems, 0, seed, test_name, base_path)

                    end_time = time.time()
                    iteration_time = (end_time - start_time) / 60
                    print('n_dems: ', n_dems, 'seed: ', seed, 'test_name: ', test_name, 'iteration_time: ', iteration_time)