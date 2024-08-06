import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import MyModel, Autoencoder
import pickle



# import dataset for offline training
class MyData(Dataset):

    def __init__(self):
        self.data = pickle.load(open("data/data.pkl", "rb"))
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return torch.FloatTensor(self.data[idx])



def train_model(cfg):
    alg = cfg.alg
    savename = 'model_{}.pt'.format(cfg.alg)

    # training parameters
    # recommended EPOCH: 2000 - 6000
    # recommended BATCH_SIZE: 1/10th dataset length
    # recommended LR: 0.0001
    EPOCH = 4000
    LR = 0.0001

    # dataset and optimizer
    model = MyModel()
    train_data = MyData()
    BATCH_SIZE = int(len(train_data) / 10.)
    print("my batch size is:", BATCH_SIZE)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # encoder for images
    ae = Autoencoder()
    ae.load_state_dict(torch.load('data/autoencoder.pt'))
    ae.eval()

    # helper code for getting eigenvalues
    relu = torch.nn.ReLU()

    # main training loop
    for epoch in range(EPOCH+1):
        for batch, x in enumerate(train_set):
            states_x = x[:, 0:2]
            states_y = x[:, 2:443]
            actions = x[:, 443:445]
            
            # encoder image to get y
            z = ae.encoder(states_y).detach()
            states = torch.cat((states_x, z), dim=1)

            # get mse loss
            loss = model.loss_func(actions, model(states_x, z))

            # get additional loss terms
            if alg == 'stable':
                # get the matrix A
                states.requires_grad = True
                a = model.combined(states)
                J = torch.zeros((BATCH_SIZE, 2, 12))
                for i in range (2):
                    J[:, i] = torch.autograd.grad(a[:, i], states, 
                                        grad_outputs=torch.ones_like(a[:, i]), 
                                        create_graph=True)[0]
                
                # make sure top left of A is stable
                J_x = J[:,:,:2]
                # get the eigenvalues of the matrix
                E = torch.linalg.eigvals(J_x).real
                # loss is the sum of positive eigenvalues
                loss += 10.0 * torch.sum(relu(E))
                
            
                # penalize the magnitude of the top right of A
                J_y = J[:,:,2:]
                # get the norm of the matrix
                D = torch.linalg.matrix_norm(J_y)
                # loss is the average of the matrix magnitude
                loss += 0.1 * torch.mean(D)

            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 500 == 0:
            print(epoch, loss.item())
    
    torch.save(model.state_dict(), "data/" + savename)

