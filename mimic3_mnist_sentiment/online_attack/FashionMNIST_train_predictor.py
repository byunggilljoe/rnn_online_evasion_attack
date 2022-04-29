from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append("./")

import numpy as np
import argparse
import os
import re
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset
from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, LSTMRealTimeRegressor
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_realtime_regression



#print(val_poison_targets)
trans = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST("./data", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.FashionMNIST("./data", train=False, transform=trans, download=True)
LABELS_IN_USE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_label_mask = [e in LABELS_IN_USE for e in train_dataset.targets]
test_label_mask = [e in LABELS_IN_USE for e in test_dataset.targets]

print(train_dataset.data[train_label_mask].size())
train_targets = train_dataset.targets[train_label_mask]
test_targets = test_dataset.targets[test_label_mask]
for i in range(len(LABELS_IN_USE)):
    train_targets[train_targets == LABELS_IN_USE[i]] = i
    test_targets[test_targets == LABELS_IN_USE[i]] = i
train_tensor_dataset = TensorDataset(train_dataset.data[train_label_mask].transpose(1, 2)/256.0, train_targets)
test_tensor_dataset = TensorDataset(test_dataset.data[test_label_mask].transpose(1, 2)/256.0, test_targets)

train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=128, shuffle=False)

class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = 1
        self.n_hidden = 128#16
        self.num_direction = 1
        assert self.num_direction in [1, 2]
        dropout_p = 0.3
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,                                    bias=True, batch_first=True,                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
        self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 150)
        self.fc2 = torch.nn.Linear(150, 28)
        self.dropout = torch.nn.Dropout(dropout_p)
    
    def forward(self, x):
        #hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        #cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        hidden_and_cell = self.get_init_hidden_cell(x.size(0)) #(hidden_init, cell_init)
        on, (hn, cn) = self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)

        o = F.relu(self.dropout(self.fc1(on)))
        o = self.fc2(o)
        return o
    
    def get_init_hidden_cell(self, size):
        hidden_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
        cell_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
        return (hidden_init, cell_init)
    
    def get_one_pred(self, x, hidden_and_cell):
        on, (hn, cn) = self.lstm(x, hidden_and_cell)
        o = F.relu(self.dropout(self.fc1(on)))
        o = self.fc2(o)
        return o, (hn, cn)
    
    def loss(self, x, x_hat):
        return torch.pow(x_hat[:, :-1, :] - x[:, 1:, :], 2).sum(dim=(1, 2)).mean() # Predict next timestamp


lstmp = LSTMPredictor(28)
lstmp.cuda()
_, (data, target) = next(enumerate(test_loader))
data = data.cuda()
x_hat = lstmp(data)
lstmp.loss(x_hat, data)

lstmp.train()
optimizer = torch.optim.Adam(lstmp.parameters(), lr=0.0001)
for e in range(200):
    for i, (data, target) in enumerate(train_loader):
        lstmp.zero_grad()
        data = data.cuda()
        x_hat = lstmp(data)
        #loss = F.mse_loss(data, recons)
        loss = lstmp.loss(data, x_hat)
        loss.backward()
        optimizer.step()
    print(loss.item())

torch.save(lstmp.state_dict(), "tmp/FashionMNIST_predictor.pt")
