import numpy as np 
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from data import UdacityDataset, Rescale, Preprocess, ToTensor, AdvDataset
from model import BaseCNN, SteeringAngleRegressor, Nvidia, build_vgg16, Vgg16
# from viewer import draw
from scipy.misc import imresize
from torchvision import datasets, transforms
from fgsm_attack import fgsm_attack
from advGAN.models import Generator
from optimization_attack import optimized_attack
import attack_test
from optimization_universal_attack import generate_noise
from advGAN_attack import advGAN_Attack
import os
from torch.utils.data import DataLoader
from scipy.misc import imsave, imread
import cv2
import os
import argparse


model = BaseCNN()
model.load_state_dict(torch.load("baseline.pt"))
model.cuda()
model.eval()

batch_size = 32
dataset_path = "../udacity-data"
resized_image_height = 128
resized_image_width = 128  

image_size = (resized_image_width, resized_image_height)

composed = transforms.Compose([Rescale(image_size), Preprocess(), ToTensor()])

dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)
train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)


_, batch = next(enumerate(train_generator))
batch_x = batch["image"].float().cuda()
batch_y = batch["steer"].float().cuda()

print(batch_x.size())
print(batch_y.size())

def pgd_attack(model, batch_x, batch_y, epsilon, step_size, iters, min_value, max_value):
    adv_x = batch_x.clone()
    
    adv_x.requires_grad = True

    for _ in range(iters):
        model.zero_grad()
        out = model(adv_x)
        loss = F.mse_loss(batch_y, out)
        loss.backward()

        grad = adv_x.grad.data

        adv_x = adv_x + torch.sign(grad)*step_size
        delta = torch.clamp(adv_x - batch_x, -epsilon, epsilon)
        adv_x = torch.clamp(batch_x + delta, min_value, max_value).detach()
        adv_x.requires_grad = True
    
    return adv_x

benign_out = model(batch_x)
adv_out = model(pgd_attack(model, batch_x, batch_y, 0.01, 0.001, 40, 0.0, 1.0))

benign_loss = F.mse_loss(benign_out, batch_y)
adv_loss = F.mse_loss(adv_out, batch_y)

print(benign_loss, adv_loss)

sag_model = SteeringAngleRegressor(-1, -1, sequence_input=False)
sag_model.load_state_dict(torch.load("lstm_sequence_.pt"))
sag_model.cuda()
adv_out = model(pgd_attack(sag_model, batch_x, batch_y, 0.01, 0.001, 40, 0.0, 1.0))
adv_loss = F.mse_loss(adv_out, batch_y)
print(benign_loss, adv_loss)