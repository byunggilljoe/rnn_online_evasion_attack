import sys
sys.path.append("../attack-codes/")
sys.path.append("./")

import matplotlib
matplotlib.use('Agg')
from model import UNet, build_vgg16, weight_init
from data import UdacityDataset_LSTM, Rescale, RandFlip, Preprocess2, RandRotation, ToTensor, RandBrightness, RandRotateView
import torch.optim as optim
import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import csv
from os import path
from scipy.misc import imread, imresize, imsave
import numpy as np 
import pandas as pd 
import time
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import argparse
import cv2
#from adv_training import test_on_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training.')
    parser.add_argument("--root_dir", type=str, default="../udacity-data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()

    model_name = "prediction_unet"
    camera = 'center'
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    resized_image_height = 0
    resized_image_width = 0
    train = args.train
    resized_image_height = 64
    resized_image_width = 64
    DOWNSAMPLE_FACTOR = 2
      
    image_size=(resized_image_width, resized_image_height)
    dataset_path = args.root_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    NUM_HISTORY = 5
    k = 1
    net = UNet(3*NUM_HISTORY, bilinear=True)
    
    net.apply(weight_init)
    net = net.to(device)
    # net.to(device)
    # net.load_state_dict(torch.load("prediction_unet_sequence_.pt"))
    if train != 0:
        if train == 2:
            net.load_state_dict(torch.load(model_name + '.pt'))
        print("===== train if statement")
        #composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess(), ToTensor()])
        composed = transforms.Compose([Rescale(image_size), Preprocess2(), ToTensor()])
        # num_frame_per_sample=NUM_HISTORY+1 => HistoryInput + 1 future frame
        dataset = UdacityDataset_LSTM(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed, 
                                    num_frame_per_sample=NUM_HISTORY+k, downsample_factor=DOWNSAMPLE_FACTOR)
        
        steps_per_epoch = int(len(dataset) / batch_size)

        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        #criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # x,y = train_generator.__next__()
        # print(x.shape)
        for epoch in range(epochs):
            total_loss = 0
            print("===== Running !", device)
            for step, (image_sequence, _) in enumerate(train_generator):
                
                #print("steer_sequence.size()", steer_sequence.size())
                if step <= steps_per_epoch:
                    # BATCH x LENGTH x CHNNELS x HEIGHT x WIDTH
                    INPUT_SIZE = image_sequence.size()
                    batch_x = image_sequence[:, :NUM_HISTORY, :, :, :].view(INPUT_SIZE[0], NUM_HISTORY*INPUT_SIZE[2], INPUT_SIZE[3], INPUT_SIZE[4])
                    batch_y = image_sequence[:, NUM_HISTORY+k-1, :, :, :]
                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = net(batch_x)
                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    running_loss = loss.item()
                    total_loss += running_loss
                    #print(running_loss)
                else:
                    break
            print('Epoch %d  RMSE loss: %.4f' % (epoch,  total_loss / steps_per_epoch))
                
        torch.save(net.state_dict(), model_name + '_sequence_{}_{}.pt'.format(NUM_HISTORY, k))
    else:
        net.load_state_dict(torch.load(model_name + '.pt'))


    
    net.eval()
    with torch.no_grad():
        yhat = []
        # test_y = []
        test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values
        # composed = transforms.Compose([Rescale(image_size),  Preprocess(model_name), ToTensor()])

        # dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)
        # steps_per_epoch = int(len(dataset) / batch_size)

        # train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        test_composed = transforms.Compose([Rescale(image_size), Preprocess(), ToTensor()])
        test_dataset = UdacityDataset_LSTM(dataset_path, ['testing'], test_composed, 'test', num_frame_per_sample=NUM_HISTORY + k
                                        , downsample_factor=DOWNSAMPLE_FACTOR)
        test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
        total_loss = 0
        for _, (image_sequence, _) in enumerate(test_generator):
            INPUT_SIZE = image_sequence.size()
            batch_x = image_sequence[:, :NUM_HISTORY, :, :, :].view(INPUT_SIZE[0], NUM_HISTORY*INPUT_SIZE[2], INPUT_SIZE[3], INPUT_SIZE[4])
            batch_y = image_sequence[:, NUM_HISTORY+k-1, :, :, :]

            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()

        print('Test  RMSE loss: %.4f' % (total_loss / steps_per_epoch))
        
    
    
