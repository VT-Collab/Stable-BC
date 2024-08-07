import torch
import numpy as np
from torch.utils.data import Dataset
from models import MyModel
import pickle
import matplotlib.pyplot as plt
import wandb
import datetime
from utils import seedEverything
from tqdm import tqdm
import os, sys


# def x_dot(x, u):
#     x_dot = np.zeros_like(x)
#     x_dot[...,0] = x[...,3]
#     x_dot[...,1] = x[...,4]
#     x_dot[...,2] = x[...,5]
#     x_dot[...,3] = A_G * np.tan(u[...,2])
#     x_dot[...,4] = -A_G * np.tan(u[...,1])
#     x_dot[...,5] = u[...,0] - A_G
#     return x_dot
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

    # helper code for getting eigenvalues
    relu = torch.nn.ReLU()


    wandb.init(
        # set the wandb project where this run will be logged
        
        project=f"state_6d_quad_{num_dems}_dems_{stability_loss_coef}coef_{LR}lr_{EPOCH}epoch_{datetime.datetime.now().strftime('%m-%d')}",

        # set the name for this run, used in wandb to track the run
        name=f"{num_dems}_dems_model{type}",


        # track hyperparameters and run metadata
        config={
        "architecture": "64",
        "epochs": EPOCH,
        'type': type,
        'num_dems': num_dems,
        'savename': savename
        }
    )

    # best_valid_loss = np.inf


    # main training loop
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

        if type == 0:
            wandb.log({"bc_loss": total_loss / n_training_samples, "test_loss": validdation_loss_per_sample })
        elif type >= 1:
                wandb.log({"total_loss": total_loss / n_training_samples, "bc_loss": total_loss_bc / n_training_samples, "stability_loss":total_loss_stability / n_training_samples, "test_loss": validdation_loss_per_sample })

        # if not os.path.exists(models_path):
        #     os.makedirs(models_path)


        # # save the model and the best model
        # if validdation_loss_per_sample < best_valid_loss or epoch == EPOCH - 1:
        #     # we have a best model
        #     best_valid_loss = validdation_loss_per_sample
        #     str_save_model = '_b'
        #     # save the best model
        #     save_model(model, optimizer, self.criterion, epoch, checkpoint_path_to_save + f'/model_{epoch}{str_save_model}')
        #     if not np.isnan( results['last_best_save_epoch'] ):
        #         delete_model(checkpoint_path_to_save + '/model_{}{}'.format(results['last_best_save_epoch'], str_save_model))   
        #     results['last_best_save_epoch'] = epoch
        # else:
        #     str_save_model = ''
        #     if epoch % checkpoint_save_steps == 0:
        #         # checkpoint_save model
        #         save_model(model, optimizer, self.criterion, epoch, checkpoint_path_to_save + f'/model_{epoch}{str_save_model}')
        #         if not np.isnan(results['last_save_epoch']):
        #             delete_model(checkpoint_path_to_save + '/model_{}{}'.format(results['last_save_epoch'], str_save_model))
        #         results['last_save_epoch'] = epoch


        # # early stopping based on valid
        # if enable_early_stop:
        #     early_stopping(validdation_loss_per_sample)
        #     if early_stopping.early_stop:
        #         print(f"the model training stops at epoch {epoch} based on early stopping...")
        #         print(f"last {early_stop_patience + 1} loss_main_valid values are: {results['valid_loss'][-(early_stop_patience + 1):]}")
        #         break

        # wandb.finish()
        # # load the best model
        # model, _, _ = load_model(model, optimizer, checkpoint_path_to_save + '/model_{}_b'.format(results['last_best_save_epoch']))


    if not os.path.exists(models_path):
        os.makedirs(models_path)

    torch.save(model.state_dict(), models_path + '/' + savename)
    wandb.finish()





def train_imitation_agent(num_dems, type: int, random_seed):
    EPOCH = 1000 # 500
    LR = 0.001
    stability_loss_coef = 0.0001 # 0.001




    data_dict = pickle.load(open("sim10_quadrotor/data/data_0.pkl", "rb"))
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

    # if type == 2:
    #     pass
    #     # simple implementation of CCIL
    #     # state1[:2] += np.random.normal(0, 0.2, 2)
    #     # action1 = (state[:2] + u1) - state1[:2]
    # either implement ccil in data collection and load it or implement it here
        
   
   
    train_dataset = StateImitationDataset(train_x_traj_array, train_controls_array)
    test_dataset = StateImitationDataset(test_x_traj_array, test_controls_array)

    batch_size = int( len(train_dataset) / 10)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    models_path= f"sim10_quadrotor/first_results_{LR}lr_{EPOCH}epoch/lamda_{stability_loss_coef}/{num_dems}dems/{random_seed}"

    savename = f"im_model{type}.pt"
    train_model(num_dems, type, train_dataloader, test_dataloader, savename, EPOCH=EPOCH, LR=LR, stability_loss_coef=stability_loss_coef, models_path=models_path)
    





# train behavior cloned agent
# type 0: standard behavior cloning (baseline1)
# type 1: our proposed approach (stable)
if __name__ == "__main__":

    # use gpu number 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    num_dems = 11
    random_seed = 0

    train_imitation_agent(num_dems, 1, random_seed)
    # train_imitation_agent(num_dems, 0, random_seed)


