import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MyModel, Autoencoder
import csv
import argparse



def rollout(cfg):
    # load relevant models
    loadname = 'model_{}.pt'.format(cfg.alg)
    model = MyModel()
    model.load_state_dict(torch.load('data/' + loadname))
    model.eval()
    ae = Autoencoder()
    ae.load_state_dict(torch.load('data/autoencoder.pt'))
    ae.eval()
    # main loop
    error = []
    n_rollouts = cfg.n_rollouts
    n_timesteps = 30
    for _ in range(n_rollouts):
        # get environment
        robot_pos = np.random.uniform(-10, 10, 2)
        goal_pos = np.random.uniform(-10, 10, 2)
        goal_pixel = (np.round(goal_pos) + 10).astype(int)
        img = np.zeros((21, 21)).astype(int)
        img[goal_pixel[0], goal_pixel[1]] = 255
        y = torch.FloatTensor(img.flatten())
        z = ae.encoder(y.unsqueeze(0)).detach()
        # rollout policy
        for _ in range(n_timesteps):
            x = torch.FloatTensor(robot_pos)
            robot_action = model(x.unsqueeze(0), z).detach().numpy()[0]
            if np.linalg.norm(robot_action) > 1.0:
                robot_action /= np.linalg.norm(robot_action)
            robot_pos += robot_action
        # get error
        error.append(np.linalg.norm(goal_pos - robot_pos))
    error = np.array(error)
    return np.mean(error)


def rollout_policy(cfg):
    savename = 'data/results_{}_{}.csv'.format(cfg.alg, cfg.num_dp)
    # get error for each model
    error = rollout(cfg)
    print("average error:", np.round(error, 2))
    with open(savename,'a') as myfile:
        datarow = error
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list(datarow))