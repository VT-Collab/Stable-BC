
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import MyModel
import json
import random



# import dataset for offline training
class MyData(Dataset):

    def __init__(self, cfg):
        if not cfg.train_full:
            random.seed(cfg.seed)
            self.data = random.choices(json.load(open("data/user_{}/demo_processed.json".format(cfg.user), "r")), k=cfg.num_dp)
        if cfg.train_full:
            data = []
            for idx in range(1, 11):
                random.seed(cfg.seed)
                data_i = random.choices(json.load(open("data/user_{}/demo_processed.json".format(idx), "r")), k=cfg.num_dp)
                data.append(data_i)
            self.data = np.concatenate((data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]), axis=0)
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return torch.FloatTensor(self.data[idx])



def train_model(cfg):
    alg = cfg.alg
    savename = 'model_{}'.format(alg)
    EPOCH = 4000
    LR = 0.0001

    # Initialize model and optimizer parameters
    model = MyModel()
    train_data = MyData(cfg)
    BATCH_SIZE = int(len(train_data) / 10.)
    print("my batch size is:", BATCH_SIZE)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    relu = torch.nn.ReLU()

    # Main training loop
    for epoch in range(EPOCH + 1):
        for batch, x in enumerate(train_set):
            states = x[:, 0:6]
            actions  = x[:, 6:8]
            
            # standard BC loss
            loss = model.loss_func(actions, model(states))

            # Compute additional loss for stable
            if alg == 'stable':
                states.requires_grad = True
                a = model(states)
                J = torch.zeros((len(x), 2, 6))
                # Get the top left term of A matrix
                for i in range (2):
                    J[:, i] = torch.autograd.grad(a[:, i], states, grad_outputs=torch.ones_like(a[:, i]), create_graph=True)[0]
                J1 = J[:, :, 0:2]
                E = torch.linalg.eigvals(J1).real
                loss += 10.0 * torch.sum(relu(E))

                # Get top right term of A matrix
                K = J[:, :, 2:6]
                D = torch.linalg.matrix_norm(K)
                loss += 0.1 * torch.mean(D)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 500 == 0:
            print(epoch, loss.item())
    torch.save(model.state_dict(), "data/user_{}/{}_dp/".format(cfg.user, cfg.num_dp) + savename + '.pt')