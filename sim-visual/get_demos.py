import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse



def get_dataset(cfg):
    num_datapoints = cfg.num_dp
    dataset = []
    for _ in range(num_datapoints):
        robot_pos = np.random.uniform(-10, 10, 2)
        goal_pos = np.random.uniform(-10, 10, 2)
        robot_action = goal_pos - robot_pos
        if np.linalg.norm(robot_action) > 1.0:
            robot_action /= np.linalg.norm(robot_action)

        goal_pixel = (np.round(goal_pos) + 10).astype(int)
        img = np.zeros((21, 21)).astype(int)
        img[goal_pixel[0], goal_pixel[1]] = 255

        dataset.append(list(robot_pos) + list(img.flatten()) + list(robot_action))

    pickle.dump(dataset, open("data/data.pkl", "wb"))
    print("dataset has this many state-action pairs:", len(dataset))

