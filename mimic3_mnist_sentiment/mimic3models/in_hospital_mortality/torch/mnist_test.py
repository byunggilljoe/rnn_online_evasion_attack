from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import re
import math
import json
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from mimic3models.in_hospital_mortality.torch.model_torch import CNN_AE_MNIST
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_17
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_trigger
from mimic3models.in_hospital_mortality.torch.discretizers import TriggerGenerationDiscretizer

USE_VAE = False
def train(model, train_loader):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    best_state_dict = None
    best_loss = 99999
    
    for e in range(100):#50
        print("Epoch:", e)
        model.train()
        model.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
    
            out = model(x)
            loss = model.loss(x, out)
            if i %100 == 0:
                print(f"loss: {loss}")
                if loss < best_loss:
                    print("Best loss:", loss.item())
                    best_loss = loss
                    best_state_dict = model.state_dict()
            loss.backward()
            optimizer.step()
        

        #print(recon_loss.item(), kl_loss.item())
    return best_state_dict



def main():


    print("==> training")

    #print(val_poison_targets)
    trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, transform=trans, download=True)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = CNN_AE_MNIST(n_hidden=128)
    
    best_state_dict = train(model,train_loader)
    torch.save(model.state_dict(), "cnn_ae_mnist.pt")
    

if __name__ == "__main__":
    main()