import sys
sys.path.append("../attack-codes/")
sys.path.append("../CrevNet-Traffic4cast/")
sys.path.append("./")
import attacks
import layers
import matplotlib
matplotlib.use('Agg')
from model import UNet, SteeringAngleRegressor
from data import UdacityDataset_LSTM, UdacityDataset, Rescale, RandFlip, Preprocess, RandRotation, ToTensor, RandBrightness, RandRotateView
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
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

#from adv_training import test_on_file

def get_dataloader(args, train):
    batch_size = args.batch_size
    dataset_path = args.root_dir
    resized_image_height = 128
    resized_image_width = 128  
    image_size = (resized_image_width, resized_image_height)
    #composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess(), ToTensor()])
    composed = transforms.Compose([Rescale(image_size), Preprocess(), ToTensor()])
    if train:
        dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)
        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        return train_generator
    else:
        test_dataset = UdacityDataset(dataset_path, ['testing'], composed, 'test')
        test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_generator
    
def get_dataloader_lstm(args, train, train_shuffle=True):
    batch_size = args.batch_size
    dataset_path = args.root_dir
    resized_image_height = 64
    resized_image_width = 64  
    image_size = (resized_image_width, resized_image_height)
    #composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess(), ToTensor()])
    composed = transforms.Compose([Rescale(image_size), Preprocess(), ToTensor()])
    if train:
        dataset = UdacityDataset_LSTM(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed, num_frame_per_sample=args.NUM_TOTAL, downsample_factor=args.DOWNSAMPLE_FACTOR)
        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=8)
        return train_generator
    else:
        test_dataset = UdacityDataset_LSTM(dataset_path, ['testing'], composed, 'test', num_frame_per_sample=args.NUM_TOTAL, downsample_factor=args.DOWNSAMPLE_FACTOR)
        test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_generator

def get_models(args):
    NUM_HISTORY = args.NUM_HISTORY
    prediction_unet = UNet(3*NUM_HISTORY, bilinear=True)
    steering_net = SteeringAngleRegressor(-1, -1, sequence_input=True)
    
    prediction_unet.load_state_dict(torch.load("prediction_unet_sequence_{}_{}.pt".format(args.NUM_HISTORY, 1)))
    #prediction_unet.load_state_dict(torch.load("prediction_unet_sequence_.pt"))
    steering_net.load_state_dict(torch.load("lstm_sequence_.pt"))
    return prediction_unet, steering_net

def prediction_test(args, prediction_unet):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_generator = get_dataloader(args, train=False, num_frame_per_sample=5+1)

    _, (image_sequence, _) = next(enumerate(test_generator))
    INPUT_SIZE = image_sequence.size()
    batch_x = image_sequence[:, :-1, :, :, :].view(INPUT_SIZE[0], (INPUT_SIZE[1]-1)*INPUT_SIZE[2], INPUT_SIZE[3], INPUT_SIZE[4])
    batch_y = image_sequence[:, -1, :, :, :]
    batch_x = batch_x.type(torch.FloatTensor)
    batch_y = batch_y.type(torch.FloatTensor)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    criterion = nn.MSELoss()
    # Prediction Test
    with torch.no_grad():
        pred_out = prediction_unet(batch_x)
        loss = criterion(batch_y, pred_out)
        print("Loss of test data batch: {:.04f}".format(loss.item()))
        colcat_image_sequence = torch.cat([image_sequence[:, i, :, :, :] for i in range(image_sequence.size(1))], dim=3)
        print("colcat_image_sequence.size():", colcat_image_sequence.size())

        for j in range(10):
            plt.figure(figsize=(15, 3), dpi=80)
            
            for k in range(6):
                plt.subplot(1, 7, k+1)
                plt.imshow(image_sequence[j, k].cpu().detach().numpy().transpose((1,2,0)))
                plt.xticks([])
                plt.yticks([])
            plt.subplot(1, 7, 7)
            plt.imshow(pred_out[j].cpu().clamp(0, 1).detach().numpy().transpose((1,2,0)))
            plt.savefig("data_{}.png".format(j))

def get_loss_func(num_loss_step=1, gamma=1.0):
    gamma_array = torch.ones(num_loss_step).cuda()
    for j in range(1, num_loss_step):
        gamma_array[j:]*=gamma

    def loss_func(out, label, gamma_array=gamma_array):
        gamma_array = gamma_array.repeat(out.size(0)).float()
        num_out = out[:, -num_loss_step:].reshape(-1) 
        true_label = label[:, -num_loss_step:].reshape(-1) 
        return (F.mse_loss(num_out, true_label, reduce='none')*gamma_array).mean()

    return loss_func


def clairvoyant_attack_test(args, steering_net, test_generator):
    TOTAL_MSE = 0
    COUNT_SAMPLE = 0
    NUM_BATCH = args.NUM_BATCH
    for i, (image_sequence, steering_sequence) in enumerate(test_generator):
        batch_x = image_sequence.float().cuda()
        batch_y = steering_sequence.float().cuda()

        criterion = nn.MSELoss()
        ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
        EPSILON = args.eps*ONES_LIKE_FRAME #0.05*ONES_LIKE_FRAME
        STEP_SIZE = args.step_size*ONES_LIKE_FRAME # 0.001
        ITERS = args.iters
        MAX_VALUE = ONES_LIKE_FRAME*1.0
        MIN_VALUE = ONES_LIKE_FRAME*0.0

        batch_x_adv = attacks.clairvoyant_each_step_attack(steering_net, batch_x, batch_y, epsilon=EPSILON,
                            step_size=STEP_SIZE, max_iters=ITERS,
                            min_value=MIN_VALUE, max_value=MAX_VALUE,\
                            loss_func=get_loss_func(num_loss_step=args.NUM_TOTAL))
        
        print("=== Clairvoyant attack ===")
        with torch.no_grad():
            benign_out = steering_net(batch_x)
            adv_out = steering_net(batch_x_adv)
            benign_loss = criterion(benign_out, batch_y)
            adv_loss = criterion(adv_out, batch_y)
            print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
        np.save("batch_x.npy", batch_x.cpu().detach().numpy())
        np.save("batch_x_adv.npy", batch_x_adv.cpu().detach().numpy())

        TOTAL_MSE += adv_loss.item() * batch_x.size(0)
        COUNT_SAMPLE += batch_x.size(0)
        if i >= NUM_BATCH - 1:
            break
    return TOTAL_MSE/COUNT_SAMPLE

def greedy_attack_test(args, steering_net, test_generator):
    TOTAL_MSE = 0
    COUNT_SAMPLE = 0
    NUM_BATCH = args.NUM_BATCH
    for i, (image_sequence, steering_sequence) in enumerate(test_generator):
        batch_x = image_sequence.float().cuda()
        batch_y = steering_sequence.float().cuda()

        criterion = nn.MSELoss()
        ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
        EPSILON = args.eps*ONES_LIKE_FRAME #0.05*ONES_LIKE_FRAME
        STEP_SIZE = args.step_size*ONES_LIKE_FRAME # 0.001
        ITERS = args.iters
        MAX_VALUE = ONES_LIKE_FRAME*1.0
        MIN_VALUE = ONES_LIKE_FRAME*0.0
        
        batch_x_adv = attacks.greedy_each_step_attack(steering_net, batch_x, batch_y, epsilon=EPSILON,
                            step_size=STEP_SIZE, max_iters=ITERS,
                            min_value=MIN_VALUE, max_value=MAX_VALUE,\
                            loss_func=get_loss_func(num_loss_step=1))

        print("=== greedy each step attack ===")
        with torch.no_grad():
            benign_out = steering_net(batch_x)
            adv_out = steering_net(batch_x_adv)
            benign_loss = criterion(benign_out, batch_y)
            adv_loss = criterion(adv_out, batch_y)
            print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        TOTAL_MSE += adv_loss.item() * batch_x.size(0)
        COUNT_SAMPLE += batch_x.size(0)
        if i >= NUM_BATCH - 1:
            break
    return TOTAL_MSE/COUNT_SAMPLE

def get_pred_func(args, pred_model):

    def pred_func(current_data, predicted_data_only, state):
        colcat_data = torch.cat([current_data, predicted_data_only], dim=1)
        total_time_steps = colcat_data.size(1)
        if total_time_steps < args.NUM_HISTORY:
            # Pad not enough data
            repeat_array = [1]*len(list(colcat_data.size()))
            repeat_array[1] = args.NUM_HISTORY - total_time_steps
            colcat_data = torch.cat(
                    [colcat_data[:, :1].repeat(repeat_array), colcat_data], dim=1)
            
        PRED_INPUT_SIZE = (current_data.size(0), args.NUM_HISTORY*3, current_data.size(-2), current_data.size(-1))

        pred_input = colcat_data[:, -args.NUM_HISTORY:].view(*PRED_INPUT_SIZE)
        next_frame = pred_model(pred_input)
        return next_frame.unsqueeze(1), None # The first output should have time dimension

    return pred_func

def predictive_attack_test(args, steering_net, predictive_unet, test_generator):
    TOTAL_MSE = 0
    COUNT_SAMPLE = 0
    NUM_BATCH = args.NUM_BATCH
    for i, (image_sequence, steering_sequence) in enumerate(test_generator):
        batch_x = image_sequence.float().cuda()
        batch_y = steering_sequence.float().cuda()

        criterion = nn.MSELoss()
        ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
        EPSILON = args.eps*ONES_LIKE_FRAME #0.05*ONES_LIKE_FRAME
        STEP_SIZE = args.step_size*ONES_LIKE_FRAME # 0.001
        ITERS = args.iters
        MAX_VALUE = ONES_LIKE_FRAME*1.0
        MIN_VALUE = ONES_LIKE_FRAME*0.0

        NUM_PREDICTION = args.NUM_PREDICTION
        #attacks.get_predicted_data(batch_x[:, :1], get_pred_func(args, predictive_unet), 10)

        arg_dict = {"model":steering_net,
                    "current_data":batch_x,
                    "true_label":batch_y,
                    "epsilon":EPSILON,
                    "step_size":STEP_SIZE,
                    "max_iters":ITERS,
                    "min_value":MIN_VALUE,
                    "max_value":MAX_VALUE,
                    "loss_func":get_loss_func(num_loss_step=NUM_PREDICTION+1),
                    "pred_func":get_pred_func(args, predictive_unet),
                    "predictive_steps":NUM_PREDICTION}
        batch_x_adv = attacks.predictive_each_step_attack(**arg_dict)


        print("=== predictive_attack_test ===")
        with torch.no_grad():
            benign_out = steering_net(batch_x)
            adv_out = steering_net(batch_x_adv)
            benign_loss = criterion(benign_out, batch_y)
            adv_loss = criterion(adv_out, batch_y)
            print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
        TOTAL_MSE += adv_loss.item() * batch_x.size(0)
        COUNT_SAMPLE += batch_x.size(0)
        if i >= NUM_BATCH - 1:
            break
    return TOTAL_MSE/COUNT_SAMPLE

def get_args():
    parser = argparse.ArgumentParser(description='Model training.')
    parser.add_argument("--root_dir", type=str, default="../udacity-data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--NUM_HISTORY", type=int, default=5)
    parser.add_argument("--NUM_PREDICTION", type=int, default=4)
    parser.add_argument("--NUM_TOTAL", type=int, default=15)

    parser.add_argument("--DOWNSAMPLE_FACTOR", type=int, default=2)
    parser.add_argument("--NUM_BATCH", type=int, default=1)

    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--attack', type=str, choices=['greedy', 'predictive', 'clairvoyant'])
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--t_start', type=int, default=0)
    parser.add_argument('--t_end', type=int, default=15)
    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predictive_unet, steering_net = get_models(args)

    predictive_unet = predictive_unet.to(device)
    steering_net= steering_net.to(device)
    
    # datasets
   
    train_generator = get_dataloader_lstm(args, train=True, train_shuffle=False)
    #test_generator = get_dataloader_lstm(args, train=False)

    #prediction_test(args, predictive_unet)
    
    ################## Attack Start ###################
    if args.attack == "clairvoyant":
        print("clairvoyant:")
        MSE=clairvoyant_attack_test(args, steering_net, train_generator)
    elif args.attack == "greedy":
        print("greedy:")
        MSE=greedy_attack_test(args, steering_net, train_generator)
    elif args.attack == "predictive":
        print("predictive:")
        MSE=predictive_attack_test(args, steering_net, predictive_unet, train_generator)
    print("MSE:{:.04f}".format(MSE))
   
