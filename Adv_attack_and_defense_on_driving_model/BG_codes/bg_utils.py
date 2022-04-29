import sys
sys.path.append("../attack-codes/")
sys.path.append("../udacity_crevnet_pred_model/")
sys.path.append("./")
sys.path.append("../")
import attacks
import matplotlib
matplotlib.use('Agg')
from model import UNet, SteeringAngleRegressor
import os
from Adv_attack_and_defense_on_driving_model.data import UdacityDataset_LSTM, UdacityDataset, Rescale, RandFlip, Preprocess2, RandRotation, ToTensor, RandBrightness, RandRotateView
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt


import numpy as np 

from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def get_dataloader(args, train):
    batch_size = args.batch_size
    dataset_path = args.root_dir
    resized_image_height = 64
    resized_image_width = 64
    image_size = (resized_image_width, resized_image_height)
    #composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess(), ToTensor()])
    composed = transforms.Compose([Rescale(image_size), Preprocess2(), ToTensor()])
    if train:
        dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)
        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        return train_generator
    else:
        test_dataset = UdacityDataset(dataset_path, ['testing'], composed, 'test')
        test_generator = DataLoader(test_dataset, batch_size=10, shuffle=False)
        return test_generator
    
def get_dataloader_lstm(args, train, train_shuffle=True):
    batch_size = args.batch_size
    dataset_path = args.root_dir
    resized_image_height = 64
    resized_image_width = 64  
    image_size = (resized_image_width, resized_image_height)
    #composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess(), ToTensor()])
    composed = transforms.Compose([Rescale(image_size), Preprocess2(), ToTensor()])
    if train:
        # dataset = UdacityDataset_LSTM(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed, num_frame_per_sample=args.NUM_TOTAL, downsample_factor=args.DOWNSAMPLE_FACTOR)
        dataset = UdacityDataset_LSTM(dataset_path, ['HMB6'], composed, num_frame_per_sample=args.NUM_TOTAL, downsample_factor=args.DOWNSAMPLE_FACTOR)
        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=8)
        return train_generator
    else:
        test_dataset = UdacityDataset_LSTM(dataset_path, ['testing'], composed, 'test', num_frame_per_sample=args.NUM_TOTAL, downsample_factor=args.DOWNSAMPLE_FACTOR)
        test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_generator

def get_models(args, trial=0):
    NUM_HISTORY = args.NUM_HISTORY
    prediction_unet = UNet(3*NUM_HISTORY, bilinear=True)
    steering_net = SteeringAngleRegressor(-1, -1, sequence_input=True, num_hidden=32)
    
    prediction_unet.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__))+"/../prediction_unet_sequence_{}_{}.pt".format(args.NUM_HISTORY, 1)))
    #prediction_unet.load_state_dict(torch.load("prediction_unet_sequence_.pt"))
    steering_net.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__))+f"/../lstm_sequence_trial_{trial}.pt"))
    eval_net = steering_net
    if args.transfer_victim_path != None:
        print("Test transfer model")
        eval_net = SteeringAngleRegressor(-1, -1, sequence_input=True, num_hidden=1024)
        eval_net.load_state_dict(torch.load(args.transfer_victim_path))
    return prediction_unet, steering_net, eval_net

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


def get_loss_func(num_loss_step=1, gamma=1.0, max_attack=False):
    if isinstance(gamma, float) == True:
        gamma_array = torch.ones(num_loss_step).cuda()
        for j in range(1, num_loss_step):
            gamma_array[j:]*=gamma
    elif isinstance(gamma, list) == True or isinstance(gamma, np.ndarray) == True :
        gamma_array = torch.tensor(gamma).cuda()
    else:
        assert(False)
# def get_loss_func(num_loss_step=1, gamma=1.0, max_attack=False):
#     gamma_array = torch.ones(num_loss_step).cuda()
#     for j in range(1, num_loss_step):
#         gamma_array[j:]*=gamma

    def loss_func(out, label, gamma_array=gamma_array, target_label=None):
        gamma_array = gamma_array.repeat(out.size(0)).float()
        num_out = out[:, -num_loss_step:].reshape(-1) 
        true_label = label[:, -num_loss_step:].reshape(-1) 
        #print("gamma_array in loss_func:", gamma_array)
        if max_attack:
            assert(target_label is None)
        
        if max_attack == False:
            #print("---")
            #print(F.mse_loss(num_out, true_label, reduction='none').size())
            #print(gamma_array.size())
            if target_label is not None:
                return -(F.mse_loss(num_out, target_label.float().view(-1), reduction='none')*gamma_array).mean()
            else:
                return (F.mse_loss(num_out, true_label, reduction='none')*gamma_array).mean()
        elif max_attack == True:
            loss_values = (F.mse_loss(num_out, true_label, reduction='none')*gamma_array).view(-1, num_loss_step)
            max_loss = torch.max(loss_values, dim=1)[0]
            mean_except_top_loss = attacks.get_bottom_k_1_mean(loss_values)
                
            return max_loss.mean()*10.0 - mean_except_top_loss
        
    return loss_func

def get_loss_func_max(num_loss_step=1, gamma=1.0):
    return get_loss_func(num_loss_step, gamma, max_attack=True)

def get_loss_func_mean(num_loss_step=1, gamma=1.0):
    return get_loss_func(num_loss_step, gamma, max_attack=False)

def compute_errors(t_start, t_end, out_pred, out_true):
    # out_pred/true: BATCH x TIME
    # mse error
    mse_criterion = nn.MSELoss()
    out_pred_range = out_pred[:, t_start:t_end]
    out_true_range = out_true[:, t_start:t_end]
    mse_error = mse_criterion(out_pred_range,
                                out_true_range).item()
    # mean of max absolute deviation
    mean_max_ad = (out_pred_range - out_true_range).abs().max(dim=1)[0].mean()

    # mean of mean absolute deviation
    mean_mean_ad = (out_pred_range - out_true_range).abs().mean(dim=1).mean()
    
    # for ts
    error_sequence = (out_pred_range - out_true_range).abs().mean(dim=0)

    return mse_error, mean_max_ad, mean_mean_ad, error_sequence
    

def get_pred_func(args, encoder, frame_predictor):

    def pred_func(current_data, predicted_data_only, state):
        # current_data: BATCH x TIME x CHANNEL x HEIGHT x WIDTH
        # input: prev frame
        # output: next frame and memo for next prediction
        current_data = current_data

        colcat_data = torch.cat([current_data, predicted_data_only], dim=1)
        batch_x = colcat_data.transpose(0, 1) # TIME x BATCH
        batch_x = torch.cat([batch_x, torch.zeros(batch_x.size(0), batch_x.size(1), 1, 64, 64, device="cuda").float()], dim=2)

        if state == None:
            # Initial prediction
            memo = Variable(torch.zeros(args.batch_size, args.rnn_size, int(args.image_width/16), int(args.image_height/16)).cuda())
            frame_predictor.batch_size = batch_x.size(1)
            frame_predictor.hidden = frame_predictor.init_hidden()
            #print(batch_x.size())
            start_index = max(batch_x.size(0) - args.NUM_HISTORY, 0)
            for i in range(start_index, batch_x.size(0)):
                x_in = batch_x[i]
                hi = encoder(x_in, True)
                hp, memo, _ = frame_predictor((hi, memo))
            x_pred = encoder(hp, False)
        else:
            memo = state
            # last frame
            #print("sasa", predicted_data_only.size()) # batch, time, channel, row, column
            predicted_data_only_tp = predicted_data_only.transpose(0, 1) #time, batch, channel, row, column
            x_in = torch.cat([predicted_data_only_tp[-1], torch.zeros(predicted_data_only_tp.size(1), 1, 64, 64).cuda() ], dim=1)
            #print("sdsd", x_in.size())
            h = encoder(x_in, True)
            h_pred, memo, _ = frame_predictor((h, memo.cuda()))
            x_pred = encoder(h_pred, False)
            
        return x_pred[ :, :3, :, :].unsqueeze(1), memo


    return pred_func

def plot_targeted_result(adv_out, target_label, prefix):
    adv_out = adv_out.cpu().detach().numpy()
    target_label = target_label.cpu().detach().numpy()
    print(adv_out.shape)
    print(target_label.shape)
    fig, axes = plt.subplots(2, 1)
    fa = axes.flatten()
    fa[0].plot(np.linspace(0, 20, 20), target_label[0])
    fa[1].plot(np.linspace(0, 20, 20), adv_out[0])
    plt.savefig(f"tmp/{prefix}_targeted_adv_out.jpg")
    np.save(f"tmp/{prefix}_adv_out.npy", adv_out[0])