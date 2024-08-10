import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import Autoencoder
import json



# import dataset for offline training
class MyData(Dataset):

    def __init__(self):
        self.data = json.load(open("data/data.json", "r"))
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return torch.FloatTensor(self.data[idx])



def train_ae(cfg):
    savename = 'autoencoder.pt'

    # training parameters
    # recommended EPOCH: 2000 - 6000
    # recommended BATCH_SIZE: 1/10th dataset length
    # recommended LR: 0.0001
    EPOCH = 4000
    LR = 0.0001

    # dataset and optimizer
    ae = Autoencoder()
    train_data = MyData()
    BATCH_SIZE = int(len(train_data) / 10.)
    print("my batch size is:", BATCH_SIZE)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)

    # main training loop
    for epoch in range(EPOCH+1):
        for batch, x in enumerate(train_set):
            states_y = x[:, 2:443]
            
            # get mse loss
            loss = ae.loss_func(states_y, ae.decoder(ae.encoder(states_y)))

            # update ae parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 500 == 0:
            print(epoch, loss.item())
    torch.save(ae.state_dict(), "data/" + savename)

    