import numpy as np
import json
import matplotlib.pyplot as plt



# agent demonstration policy
def agent(x, y, goal, beta=20.0):
    U = np.random.uniform(-1, 1, (100, 2))
    P = []
    for u in U:
        if np.linalg.norm(u) > 1.0:
            u /= np.linalg.norm(u)
        C_goal = np.linalg.norm((x + u) - goal) - np.linalg.norm(x - goal)
        C_avoid = np.linalg.norm(x - y) - np.linalg.norm((x + u) - y)
        C = C_goal + 0.75 * C_avoid
        P.append(np.exp(-beta * C))
    P /= np.sum(P)
    idx = np.random.choice(len(U), p=P)
    return U[idx, :]



# get demonstrations
def get_dataset(cfg):
    num_trajectories = cfg.num_demos
    dataset = []
    dataset_ccil = []
    goal_x = np.array([10., 0.])
    goal_y = np.array([0., 10.])
    for _ in range(num_trajectories):
        x = np.random.uniform([-10, -10], [0, 10], 2)
        y = np.random.uniform([-10, -10], [10, 0], 2)
        tau = []
        for idx in range(20):
            u1 = agent(x, y, goal_x)
            u2 = agent(y, x, goal_y)
            state = np.array([x[0], x[1], y[0], y[1]])
            tau.append(list(state))
            dataset.append(list(state) + list(u1))

            # upsampled dataset for minimal CCIL implementation      
            dataset_ccil.append(list(state) + list(u1))
            for _ in range(3):
                state1 = np.copy(state)
                state1[:2] += np.random.normal(0, 0.2, 2)
                action1 = (state[:2] + u1) - state1[:2]
                dataset_ccil.append(list(state1) + list(action1))
            x += u1
            y += u2

    json.dump(dataset, open("data/data.json", "w"))
    json.dump(dataset_ccil, open("data/data_ccil.json", "w"))
    print("dataset has this many state-action pairs:", len(dataset))
    print("CCIL dataset has this many state-action pairs:", len(dataset_ccil))
