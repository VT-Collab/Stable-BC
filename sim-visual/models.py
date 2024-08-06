import torch
import torch.nn as nn
import numpy as np



def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # multi-layer perceptron (encoder)
        self.e_1 = nn.Linear(441, 512)
        self.e_2 = nn.Linear(512, 256)
        self.e_3 = nn.Linear(256, 128)
        self.e_4 = nn.Linear(128, 10)

        # multi-layer perceptron (decoder)
        self.d_1 = nn.Linear(10, 128)
        self.d_2 = nn.Linear(128, 256)
        self.d_3 = nn.Linear(256, 512)
        self.d_4 = nn.Linear(512, 441)
        
        # other stuff
        self.m = nn.ReLU()
        self.apply(weights_init_)
        self.loss_func = nn.MSELoss()

    # encoder
    def encoder(self, y):
        y = self.m(self.e_1(y / 255.))
        y = self.m(self.e_2(y))
        y = self.m(self.e_3(y))
        return self.e_4(y)

    # decoder
    def decoder(self, z):
        y = self.m(self.d_1(z))
        y = self.m(self.d_2(y))
        y = self.m(self.d_3(y))
        return self.d_4(y) * 255.


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # multi-layer perceptron (policy)
        self.pi_1 = nn.Linear(12, 64)
        self.pi_2 = nn.Linear(64, 64)
        self.pi_3 = nn.Linear(64, 2)

        # other stuff
        self.m = nn.ReLU()
        self.apply(weights_init_)
        self.loss_func = nn.MSELoss()

    # policy
    def forward(self, x, y):
        state = torch.cat((x / 10., y), dim=1)
        state = self.m(self.pi_1(state))
        state = self.m(self.pi_2(state))
        return self.pi_3(state)

    # helper function
    def combined(self, state):
        states_x = state[:, 0:2]
        states_y = state[:, 2:12]
        return self.forward(states_x, states_y)