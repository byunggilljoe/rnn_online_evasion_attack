from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append("./")
sys.path.append("../")


import numpy as np
import argparse
import os
import re
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset

from unified_interface.data import load_data
from mimic3_mnist_sentiment.mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, LSTMRealTimeRegressor_energy#, LSTMRealTimeRegressor_energy_2



parser = argparse.ArgumentParser()
parser.add_argument("--trial", default=0, required=True, type=int)
args = parser.parse_args()

train_loader, test_loader = load_data("energy", batch_size=128)


def train_realtime_all(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # , weight_decay=1e-2
    model.cuda()

    best_state_dict = None 

    for e in range(100):
        print("epoch:", e)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            model.zero_grad()
            out = model(x)
            # print(out.size())
            # print(y.size())
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            optimizer.step()
    
        print(loss.item())
    if best_state_dict == None:
        best_state_dict = model.state_dict()
        
    return best_state_dict

model = None
def main(train_func):
    global model, num_hidden
    print("==> training")
    _, (data, target) = next(enumerate(train_loader))
    input_dim = data.shape[2]
    model = LSTMRealTimeRegressor_energy(input_dim, num_classes=1, num_hidden=16)
    train_func(model)
    return model


# # In[58]:
if args.trial == 0:
    torch.manual_seed(0)
main(train_realtime_all)
model_file_name = f"energy_rnn_regressor_trial_{args.trial}.pt"
torch.save(model.state_dict(), f"./tmp/{model_file_name}")
