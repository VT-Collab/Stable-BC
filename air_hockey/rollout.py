import numpy as np
import torch
import time
from models import MyModel
from utils import FR3, Joystick, camera
import json


def rollout_policy(cfg):
    START = [-0.0100159, 0.209284, 0.0545003, -2.64384, -0.0114203, 2.87026, 0.550335]

    # Connect to the control interface
    robot = FR3()
    conn = robot.connect(8080)
    robot.go2position(conn, START)
    mode = "v"

    interface = Joystick()
    cam = camera()

    model = MyModel()
    model.load_state_dict(torch.load('data/user_{}/{}_dp/model_{}.pt'.format(cfg.user, cfg.num_dp, cfg.alg)))
    model.eval()

    start_state = robot.readState(conn)['x']
    step_time = 0.01
    xdot = np.array([0.]*6)
    record = False
    prev_puck_pose = None

    dataset = []

    print("[*] Press A to rollout the policy")

    # Main rollout loop
    while True:
        _, a_button, b_button, start_button = interface.input()
        if a_button:
            record = True
            time.sleep(0.1)
            while prev_puck_pose is None:
                prev_puck_pose = cam.get_target()
            start_time = time.time()
        if b_button and record:
            record = False
            print("[*] Press A to record another rollout or START to save the recorded rollout")
            time.sleep(0.2)
            xdot = np.array([0.]*6)
        if start_button:
            json.dump(dataset, open('data/user_{}/{}_dp/eval{}_{}.json'.format(cfg.user, cfg.num_dp, cfg.alg, cfg.eval_num), 'w'))
            print("[*] Done!")
            return

        # Get predicted velocities, compute bounds and send velocity command to robot
        state = robot.readState(conn)
        xdot = robot.get_bounds(xdot, state, start_state, record)
        qdot = robot.xdot2qdot(xdot, state)
        robot.send2robot(conn, qdot, mode, limit=1.5)

        cur_time = time.time()
        # Get puck position and predicted robot velocities
        if record and cur_time - start_time > step_time:
            xdot = np.array([0.]*6)
            puck_pos = cam.get_target()
            if puck_pos is None:
                continue
            input = torch.FloatTensor(list(state['x'][:2]) + list(puck_pos) + list(prev_puck_pose))
            xdot[:2] += model(input.unsqueeze(0)).detach().numpy()[0]

            xdot[0] *= 2.0
            xdot[0] = np.clip(xdot[0], -1.0, 1.0)
            xdot[1] = np.clip(xdot[1], -0.25, 0.25)

            dataset.append(list(state['x'][:2]) + list(puck_pos) + list(prev_puck_pose) + list(xdot[:2]))

            start_time = cur_time
            prev_puck_pose = puck_pos
            record = True

        