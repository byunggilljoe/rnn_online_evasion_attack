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


def test_realtime_all(model):
    model.cuda()
    total_loss = 0
    CNT = 0
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            
            out = model(x)
            
            loss = torch.nn.functional.mse_loss(out, y)
            total_loss += loss*x.size(0)
            CNT += x.size(0)

    print("MSE loss:", total_loss/CNT)
            

model = None
def main(test_func):
    global model, num_hidden
    print("==> Testing")
    _, (data, target) = next(enumerate(train_loader))
    input_dim = data.shape[2]
    model = LSTMRealTimeRegressor_energy(input_dim, num_classes=1, num_hidden=16)
    model_file_name = f"energy_rnn_regressor_trial_{args.trial}.pt"
    model.load_state_dict(torch.load(f"./tmp/{model_file_name}"))
    test_func(model)
    return model


main(test_realtime_all)
