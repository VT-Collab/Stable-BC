import numpy as np
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import random
import os
import torch
import math




# taken from https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 42

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

def sample_initial_conditions_array(y_range, z_range, num_of_sample):
    initial_x = 0.2
    initial_vx = 0
    initial_vy = 0
    initial_vz = 0
    initial_conditions_arrray = np.random.uniform( (initial_x, y_range[0], z_range[0], initial_vx, initial_vy, initial_vz), (initial_x, y_range[1], z_range[1], initial_vx, initial_vy, initial_vz), (num_of_sample, 6) )
    return initial_conditions_arrray
