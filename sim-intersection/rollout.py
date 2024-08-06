import torch
import numpy as np
from models import MyModel
import matplotlib.pyplot as plt
from get_demos import agent
import json
import csv

def new_agent(y, goal, beta=20.0):
    U = np.random.uniform(-1, 1, (100, 2))
    P = []
    for u in U:
        if np.linalg.norm(u) > 1.0:
            u /= np.linalg.norm(u)
        C = np.linalg.norm((y + u) - goal) - np.linalg.norm(y - goal)
        P.append(np.exp(-beta * C))
    P /= np.sum(P)
    idx = np.random.choice(len(U), p=P)
    return U[idx, :]

# rollout the learned policy
def rollout(cfg):
    test_case = cfg.test_case
    model = MyModel()
    model.load_state_dict(torch.load('data/model_{}.pt'.format(cfg.alg)))
    model.eval() 
    tau = []
    total_cost = 0.
    
    start_x = np.random.uniform([-10, -10], [0, 10], 2)
    start_y = np.random.uniform([-10, -10], [10, 0], 2)
    if test_case == 3:
        # test diestibution has new starting states for the robot
        if np.random.rand() < 0.5:
            start_x = np.random.uniform([-10, +10], [0, +15], 2)
        else:
            start_x = np.random.uniform([-10, -15], [0, -10], 2)

    start_state = np.array([start_x[0], start_x[1], start_y[0], start_y[1]])
    state = np.copy(start_state)
    goal_x = np.array([10., 0.])
    goal_y = np.array([0., 10.])

    for idx in range(20):
        tau.append(list(state))
        # robot is controlled by behavior cloned policy
        u1 = model(torch.FloatTensor(state)).detach().numpy()
        if np.linalg.norm(u1) > 1.0:
            u1 /= np.linalg.norm(u1)

        u2 = agent(state[2:4], state[0:2], goal_y)
        if test_case == 2:
            # human moves to goal while ignoring robot
            u2 = new_agent(state[2:4], goal_y)
            
        # computing cost
        x, y = state[0:2], state[2:4]
        C_goal = np.linalg.norm((x + u1) - goal_x) - np.linalg.norm(x - goal_x)
        C_avoid = np.linalg.norm(x - y) - np.linalg.norm((x + u1) - y)
        total_cost += C_goal + 0.75 * C_avoid
        # end cost compute

        state[0:2] += u1
        state[2:4] += u2
    return np.array(tau), total_cost


def rollout_policy(cfg):
    n_rollouts = cfg.num_rollouts
    save_name = 'results_{}_{}'.format(cfg.alg, cfg.test_case)
    cost_arr = []
    for idx in range(n_rollouts):
        tau, cost = rollout(cfg)
        cost_arr.append(cost)
    print("average cost for {}:".format(cfg.alg), np.round(np.mean(cost_arr), 2))
    json.dump(cost_arr, open('data/{}.json'.format(save_name), 'w'))
    with open('data/{}.csv'.format(save_name), 'a') as myfile:
        datarow = np.mean(cost_arr)
        writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        writer.writerow([datarow])