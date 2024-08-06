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
        self.pi_1 = nn.Linear(4, 64)
        self.pi_2 = nn.Linear(64, 64)
        self.pi_3 = nn.Linear(64, 2)

        # other stuff
        self.m = nn.ReLU()
        self.apply(weights_init_)
        self.loss_func = nn.MSELoss()

    # policy
    def forward(self, state):
        x = self.m(self.pi_1(state))
        x = self.m(self.pi_2(x))
        return self.pi_3(x)
        