import torch
import torch.nn as nn
import numpy as np


A_G = 9.81
roll_max=0.4 # radians
pitch_max=0.4
f_g_diff_max=1.0 # max difference between thrust and gravity



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # multi-layer perceptron
        self.pi_1 = nn.Linear(6, 64)
        self.pi_2 = nn.Linear(64, 64)
        self.pi_3 = nn.Linear(64, 64)
        self.pi_4 = nn.Linear(64, 3)

        self.loss_func = nn.MSELoss()

    # policy
    def forward(self, state):
        state = state - torch.tensor([2.5, 2.5, 2.5, 0, 0, 0]).to(state.device)
        state = state / torch.tensor([2.5, 2.5, 2.5, 4, 3, 1]).to(state.device)
        x = torch.tanh(self.pi_1(state))
        x = torch.tanh(self.pi_2(x))
        x = torch.tanh(self.pi_3(x))
        x = self.pi_4(x)
        x = x * torch.tensor([f_g_diff_max, roll_max, pitch_max]).to(state.device) + torch.tensor([A_G, 0, 0]).to(state.device)
        return x
    

    def get_action(self, state_tensor, device ):
        if type(state_tensor) == np.ndarray:
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            # state_tensor = torch.cat( [ (state_tensor[:2] - torch.tensor([5.0, 5.0]) )/ 5, torch.sin(state_tensor[2]).reshape(1), torch.cos(state_tensor[2]).reshape(1) ] )
            # state_tensor = state_tensor.reshape((1, 4))
            # state_tensor = state_tensor.to(device)

            # for the state
            # divide by the some value

            # for the control 
            # multiply by the control bound
            # add A_G form the output 0
            state_tensor = state_tensor.to(device)
            action  = self.forward( state_tensor )
        return action.squeeze().detach()