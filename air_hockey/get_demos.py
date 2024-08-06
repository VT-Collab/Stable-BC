import numpy as np
import json
import time

from utils import FR3, Joystick, camera


def get_dataset(cfg):
    cam = camera()

    robot = FR3()
    control_mode = "v"

    print('[*] Connecting to robot...')
    conn = robot.connect(8080)
    print('[*] Connection complete')

    interface = Joystick()

    START = [-0.0100159, 0.209284, 0.0545003, -2.64384, -0.0114203, 2.87026, 0.550335]
    robot.go2position(conn, START)

    record = False
    step_time = 0.01
    dataset = []
    start_state = robot.readState(conn)['x']
    print("[*] Press A to start recording demos")
    while True:

        z, a_button, b_button, start_button = interface.input()
        if a_button and not record:
            record = True
            print("[*] Recording Started")
            time.sleep(0.1)
            start_time = time.time()
        if b_button and record:
            record = False
            print("Datapoints collected = ", len(dataset))
            print("[*] Press A to record another demo or START to save the recorded demos")
            time.sleep(1.0)
        if start_button:
            json.dump(dataset, open('data/user_{}/demo.json'.format(cfg.user), 'w'))
            print("[*] Saved {} datapoints".format(len(dataset)))
            return

        state = robot.readState(conn)
        

        xdot = robot.get_vel(z, state, start_state, record)
        
        qdot = robot.xdot2qdot(xdot, state)
        robot.send2robot(conn, qdot, control_mode, limit=1.5)

        curr_time = time.time()
        if record and curr_time - start_time >= step_time:
            print(curr_time - start_time)

            puck_pos = cam.get_target()
            if puck_pos is None:
                continue
            dataset.append(list(state['x'][:2]) + list(puck_pos) + list(xdot[:2]))
            start_time = curr_time

255, 0
458, 388