from train_model import train_imitation_agent
import os

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    n_dems_list = [5, 10, 20, 40, 60, 100]
    random_seed_list = [ i for i in range(  10 )]

    for seed in random_seed_list:
        # train
        for n_dems in n_dems_list:
            train_imitation_agent(n_dems, 1, seed)
            train_imitation_agent(n_dems, 0, seed)