import numpy as np
import socket
import time
import pygame
import cv2

# Connect to the gamepad and read selected joystick and button inputs
class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1

    def input(self):
        pygame.event.get()
        z1 = self.gamepad.get_axis(3)
        z2 = self.gamepad.get_axis(4)
        z3 = self.gamepad.get_axis(0)
        if abs(z1) < self.deadband:
            z1 = 0.0
        if abs(z2) < self.deadband:
            z2 = 0.0
        if abs(z3) < self.deadband:
            z3 = 0.0
        A_pressed = self.gamepad.get_button(0)
        B_pressed = self.gamepad.get_button(1)
        START_pressed = self.gamepad.get_button(7)
        return [z1, z2, z3], A_pressed, B_pressed, START_pressed
    

class camera():
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        time.sleep(0.1)
        self.vs.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Function to get the position of the puck in pixel space
    def get_target(self):
        _, frame = self.vs.read()
        frame = cv2.flip(frame, 1)

        # Crop the camera image show just the air hockey table
        roi = (255, 0, 203, 385)

        clone = frame.copy()
        img = clone[int(roi[1]):int(roi[1] + roi [3]), \
                    int(roi[0]):int(roi[0] + roi[2])]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Color range to track the puck
        red_lower = np.array([170, 100, 100], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)

        red = cv2.inRange(hsv, red_lower, red_upper)
        kernal = np.ones ((10, 10), "uint8")
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernal)

        (contoursred, hierarchy) =cv2.findContours (red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contourred in enumerate (contoursred):
            area = cv2.contourArea (contourred) 
            if (area > 10):
                x, y, w, h = cv2.boundingRect (contourred)
                img = cv2.rectangle (img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img,"RED",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
        
        if len(contoursred) > 0:
            # Find the biggest contour
            biggest_contour = max(contoursred, key=cv2.contourArea)

            # Find center of contour and draw filled circle
            moments = cv2.moments(biggest_contour)
            centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cv2.circle(img, centre_of_contour, 2, (0, 0, 255), -1)
            # Save the center of contour so we draw line tracking it
            center_points1 = centre_of_contour
            r1 = center_points1[0]
            c1 = center_points1[1]
            cv2.imshow('Image window', img)
            cv2.waitKey(1)

            return (c1, r1)
        return None

# Class to connect and communicate with the Franka Emika Robot controller
class FR3(object):

    def __init__(self):
        self.home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

    # Connect to the controller
    def connect(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('172.16.0.3', port))
        s.listen()
        conn, addr = s.accept()
        return conn

    # Send joint velocity to the robot
    def send2robot(self, conn, qdot, control_mode, limit=1.0):
        qdot = np.asarray(qdot)
        scale = np.linalg.norm(qdot)
        if scale > limit:
            qdot *= limit/scale
        # print(qdot)
        send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
        if send_msg == '0.,0.,0.,0.,0.,0.,0.':
            send_msg = '0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000'
        send_msg = "s," + send_msg + "," + control_mode + ","
        conn.send(send_msg.encode())

    # Get the raw states of the robot
    def listen2robot(self, conn):
        state_length = 7 + 42
        message = str(conn.recv(2048))[2:-2]
        state_str = list(message.split(","))
        for idx in range(len(state_str)):
            if state_str[idx] == "s":
                state_str = state_str[idx+1:idx+1+state_length]
                break
        try:
            state_vector = [float(item) for item in state_str]
        except ValueError:
            return None
        if len(state_vector) is not state_length:
            return None
        state_vector = np.asarray(state_vector)
        states = {}
        states["q"] = state_vector[0:7]
        states["J"] = state_vector[7:49].reshape((7,6)).T

        # get cartesian pose
        xyz_lin, R = self.joint2pose(state_vector[0:7])
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
        states["x"] = np.array(xyz)
        return states

    # Read the robot state
    def readState(self, conn):
        while True:
            states = self.listen2robot(conn)
            if states is not None:
                break
        return states

    # Convert velocity from cartesian space to joint space
    def xdot2qdot(self, xdot, states):
        J_inv = np.linalg.pinv(states["J"])
        return J_inv @ np.asarray(xdot)

    # Forward kinematics to compute end effector pose in cartesian space
    # Returns the 3-D position and the rotation matrix
    def joint2pose(self, q):
        def RotX(q):
            return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
        def RotZ(q):
            return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        def TransX(q, x, y, z):
            return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
        def TransZ(q, x, y, z):
            return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        H1 = TransZ(q[0], 0, 0, 0.333)
        H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
        H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
        H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
        H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
        H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
        H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
        H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
        T = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
        R = T[:,:3][:3]
        xyz = T[:,3][:3]
        return xyz, R

    # Send the robot to a goal position in joint space
    def go2position(self, conn, goal=False):
        if not goal:
            goal = self.home
        total_time = 15.0
        start_time = time.time()
        states = self.readState(conn)
        dist = np.linalg.norm(states["q"] - goal)
        elapsed_time = time.time() - start_time
        while dist > 0.05 and elapsed_time < total_time:
            qdot = np.clip(goal - states["q"], -0.1, 0.1)
            self.send2robot(conn, qdot, "v")
            states = self.readState(conn)
            dist = np.linalg.norm(states["q"] - goal)
            elapsed_time = time.time() - start_time

    # Compute velocity of robot based on joystick inputs
    def get_vel(self, z, state, start_state, record):
        xdot = np.array([0.]*6)
        xdot[0] = -0.5 * z[0]
        xdot[1] = 0.25 * z[1]

        # Bound the velocity to avoid collisions and angle wrapping issues
        xdot = self.get_bounds(xdot, state, start_state, record)

        return xdot
    
    # Compute teh velocity bounds
    def get_bounds(self, xdot, state, start_state, record):
        if (state['x'][0] <= 0.48 and xdot[0] < 0):
            xdot[0] = 1*(0.48 - state['x'][0])
        if (state['x'][0] >= 0.66 and xdot[0] > 0):
            xdot[0] = 0.0
        if (state['x'][1] <= -0.15 and xdot[1] < 0):
            xdot[1] = 1*(-0.15 - state['x'][1])
        if (state['x'][1] >= 0.15 and xdot[1] > 0):
            xdot[1] = 1*(0.15 - state['x'][1])
        if state['x'][2]>0.035:
            # if not record:
                xdot[2]=-0.03
            # else:
                # xdot[2] = -0.015
        elif state['x'][2] < 0.033:
            # if not record:
                xdot[2]= 0.03
            # else:
                # xdot[2] = 0.015

        xdot[3] = self.wrap_angles(start_state[3] - state['x'][3])
        xdot[4] = self.wrap_angles(start_state[4] - state['x'][4])
        xdot[5] = self.wrap_angles(start_state[5] - state['x'][5])
        return xdot

    def wrap_angles(self, theta):
        if theta < -np.pi:
            theta += 2*np.pi
        elif theta > np.pi:
            theta -= 2*np.pi
        else:
            theta = theta
        return theta
    