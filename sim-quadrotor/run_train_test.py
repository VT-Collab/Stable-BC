from train_model import train_imitation_agent
from test_model import test_imitation_agent

import os


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    n_dems_list = [5, 10, 20, 40, 60] # [ 5, 10, 15, 20, 25 ] # [ i for i in range( 1, 10 )]
    random_seed_list = [ i for i in range(  5 )]
    test_name_list = [ 'training_region' ]

    for seed in random_seed_list:
        # train
        for n_dems in n_dems_list:
            train_imitation_agent(n_dems, 1, seed)
            train_imitation_agent(n_dems, 0, seed)

        # test
        for n_dems in n_dems_list:
            for test_name in test_name_list:
                    test_imitation_agent(n_dems, 1, seed, test_name)
                    test_imitation_agent(n_dems, 0, seed, test_name)

    
