import torch
import torch.nn as nn
import numpy as np



def weights_init_(m):
    torch.manual_seed(0)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # multi-layer perceptron
        self.pi_1 = nn.Linear(6, 64)
        self.pi_2 = nn.Linear(64, 64)
        self.pi_3 = nn.Linear(64, 2)

        # other stuff
        self.m = nn.ReLU()
        self.apply(weights_init_)
        self.loss_func = nn.MSELoss()

    # policy
    def forward(self, state):
        # normalize the state into a [-1, 1] range
        x = torch.zeros_like(state)
        x[:,0] = (state[:,0] - 0.55) * 6.5
        x[:,1] = state[:,1] * 5.0
        x[:,2] = (state[:,2] - 150) / 150.0
        x[:,3] = (state[:,3] - 80) / 80.0
        x[:,4] = (state[:,4] - 150) /150.0
        x[:,5] = (state[:,5] - 80) /80.0
        
        x = self.m(self.pi_1(x))
        x = self.m(self.pi_2(x))
        x = self.pi_3(x)
        # scale the output
        return x
        