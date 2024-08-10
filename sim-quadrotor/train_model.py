import torch
import numpy as np
from torch.utils.data import Dataset
from models import MyModel
import pickle
import matplotlib.pyplot as plt
import datetime
from utils import seedEverything
from tqdm import tqdm
import os, sys

A_G = 9.81

#  write x_dot function with torch
def dynamics_model(x, u):
    x_dot = torch.zeros_like(x)
    x_dot[:,0] = x[:,3]
    x_dot[:,1] = x[:,4]
    x_dot[:,2] = x[:,5]
    x_dot[:,3] = A_G * torch.tan(u[:,2])
    x_dot[:,4] = -A_G * torch.tan(u[:,1])
    x_dot[:,5] = u[:,0] - A_G
    return x_dot


class StateImitationDataset(Dataset):
    def __init__(self, x_traj_array, controls_array, type=0):
        self.type = type
        self.x_traj_array = x_traj_array.astype(np.float32)
        self.controls_array = controls_array.astype(np.float32)

    def __len__(self):
        return self.controls_array.shape[0]

    def __getitem__(self, index):
        return self.x_traj_array[index], self.controls_array[index]
    


NUM_ACTIONS = 3


def train_model(num_dems, type, train_dataloader, valid_dataloader, savename, EPOCH=1000, LR=0.0001, stability_loss_coef=0.1, models_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyModel()
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    relu = torch.nn.ReLU()

    for epoch in range(EPOCH):

        # validation
        model.eval()
        total_test_loss = 0
        for i, data in enumerate(valid_dataloader):
            states = data[0]
            actions = data[1]
            states = states.to(device)
            actions = actions.to(device)
            outputs = model(states)
            test_loss = model.loss_func(actions, outputs)
            total_test_loss += test_loss.item()
        validdation_loss_per_sample = total_test_loss / len(valid_dataloader)
        print(f"Epoch {epoch} Test Loss: { validdation_loss_per_sample }")

        model.train()
        total_loss = 0
        total_loss_bc = 0
        total_loss_stability = 0
        
        train_bar = tqdm(train_dataloader, position = 0, leave = True)

        for batch, data in enumerate(train_bar):
            states = data[0]
            actions = data[1]
            states = states.to(device)
            actions = actions.to(device)


            STATE_DIM = states.shape[1]
            X_DIM = STATE_DIM
            
            if type == 0:
                # get mse loss
                loss = model.loss_func(actions, model(states))
                
            elif type == 1:
                # get the overall matrix for delta_x F(x, pi(x, y))
                # this is a shortcut for the top left matrix in A

                states.requires_grad = True
                outputs = model(states)
                loss_bc = model.loss_func(actions, outputs)
                F = dynamics_model(states, outputs)

                # get the gradient of a wrt states using automatic differentiation
                J = torch.zeros((states.shape[0], STATE_DIM, STATE_DIM), device=device)
                for i in range(STATE_DIM):
                    J[:, i] = torch.autograd.grad( F[:,i], states, grad_outputs=torch.ones_like(F[:,i], device=device), create_graph=True)[0]
                J = J[:, :, 0:X_DIM]


                # get the eigenvalues of the matrix
                E = torch.linalg.eigvals(J).real
                # loss is the sum of positive eigenvalues
                loss_stability = stability_loss_coef * torch.sum(relu( E )) # + EPSILON)
                
                loss = loss_bc + loss_stability
            

            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if type == 0:
                total_loss += loss.item()
                train_bar.set_description('Train iteration (epoch {}): [{}] Loss: {:.4f}'.format(epoch, batch,
                                total_loss / (batch + 1)))
            else:
                total_loss += loss.item()
                total_loss_bc += loss_bc.item()
                total_loss_stability += loss_stability.item()
                train_bar.set_description('Train iteration (epoch {}): [{}] Loss: {:.4f}, BC Loss: {:.4f}, Stability Loss: {:.4f}'.format(epoch, batch,
                                total_loss / (batch + 1), total_loss_bc / (batch + 1), total_loss_stability / (batch + 1)))


        
        n_training_samples = len(train_dataloader)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    torch.save(model.state_dict(), models_path + '/' + savename)




def train_imitation_agent(num_dems, type: int, random_seed):
    EPOCH = 1000
    LR = 0.001
    stability_loss_coef = 0.0001

    data_dict = pickle.load(open("sim-quadrotor/data/data_0.pkl", "rb"))
    controls_list = data_dict["controls_list"]
    x_traj_list = data_dict["x_trajectories_list"]

    # randomly select num_dems indices
    seedEverything(random_seed)
    indices = np.random.choice(len(controls_list), num_dems, replace=False)
    controls_array = np.concatenate([controls_list[i] for i in indices])
    x_traj_array = np.concatenate( [np.stack(x_traj_list[i]) for i in indices])
    
    del data_dict

    # split the data into training and testing
    train_x_traj_array = x_traj_array[:int(0.8 * x_traj_array.shape[0])]
    test_x_traj_array = x_traj_array[int(0.8 * x_traj_array.shape[0]):]
    train_controls_array = controls_array[:int(0.8 * controls_array.shape[0])]
    test_controls_array = controls_array[int(0.8 * controls_array.shape[0]):]
      
    train_dataset = StateImitationDataset(train_x_traj_array, train_controls_array)
    test_dataset = StateImitationDataset(test_x_traj_array, test_controls_array)

    batch_size = int( len(train_dataset) / 10)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    models_path= f"sim-quadrotor/results_{LR}lr_{EPOCH}epoch/lamda_{stability_loss_coef}/{num_dems}dems/{random_seed}"

    savename = f"im_model{type}.pt"
    train_model(num_dems, type, train_dataloader, test_dataloader, savename, EPOCH=EPOCH, LR=LR, stability_loss_coef=stability_loss_coef, models_path=models_path)
    