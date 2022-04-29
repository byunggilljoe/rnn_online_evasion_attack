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
from mimic3_mnist_sentiment.mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, LSTMRealTimeRegressor_energy



parser = argparse.ArgumentParser()
parser.add_argument("--trial", default=0, required=True, type=int)
args = parser.parse_args()

train_loader, test_loader = load_data("user", batch_size=128)

def test_realtime_all(model):
    model.cuda()
    model.eval()
    CNT = 0
    ACC_CNT = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()
            
            out = model(x) 
                  
            ACC_CNT += (out[:, -1, :].max(dim=1)[1] == y[:, 0]).float().sum()
            CNT += x.size(0)
            # ACC_CNT += (out.max(dim=2)[1] == y).float().sum()
            # CNT += x.size(0)*x.size(1)
            print(out.size())         
    print("Test acc:", ACC_CNT/CNT)

model = None
def main(test_func):
    global model, num_hidden
    print("==> training")
    _, (data, target) = next(enumerate(train_loader))
    input_dim = data.shape[2]
    print(input_dim)
    model = LSTMRealTimeRegressor_energy(input_dim, num_classes=22, num_hidden=256)
    model_file_name = f"user_rnn_regressor_trial_{args.trial}.pt"
    model.load_state_dict(torch.load(f"./tmp/{model_file_name}"))
    test_func(model)
    return model


# # In[58]:
main(test_realtime_all)

