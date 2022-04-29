import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../Adv_attack_and_defense_on_driving_model/")

import argparse
from tqdm import trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils


from data import UdacityDataset_LSTM, Rescale, RandFlip, Preprocess2, RandRotation, ToTensor, RandBrightness, RandRotateView

import layers as model

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
#parser.add_argument('--data_root', default='/home/stevelab2/Desktop/city/', help='root directory for data')
#parser.add_argument('--data_root', default='./datasets/city/', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=2000, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--image_height', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=4, type=int)
parser.add_argument('--dataset', default='traffic', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=15, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=20, help='number of frames to predict at eval time')


parser.add_argument('--rnn_size', type=int, default=512, help='dimensionality of hidden layer')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers') #################################
parser.add_argument('--g_dim', type=int, default=1024,
                   help='dimensionality of encoder hput vector and decoder input vector') #################################

parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')



parser.add_argument("--root_dir", type=str, default="../udacity-data/")
#parser.add_argument('--data_threads', type=int, default=0, help='number of data loading threads')
#parser.add_argument('--posterior_rnn_layers', type=int, default=2, help='number of layers') #################################???, not used
#parser.add_argument('--gap', type=int, default=1, help='number of timesteps')
#parser.add_argument('--z_dim', type=int, default=512, help='dimensionality of z_t') #################################???, not used

opt = parser.parse_args()

DOWNSAMPLE_FACTOR = 2
dataset_path = opt.root_dir
resized_image_height = 64
resized_image_width = 64    
image_size=(resized_image_width, resized_image_height)
#composed = transforms.Compose([Rescale(image_size), Preprocess(), ToTensor()])
composed = transforms.Compose([Rescale(image_size), Preprocess2(), ToTensor()])
dataset = UdacityDataset_LSTM(dataset_path, ['HMB2'], #['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'],
                             composed,
                             num_frame_per_sample=opt.n_eval, downsample_factor=DOWNSAMPLE_FACTOR)
#train_generator = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, drop_last=True)
train_generator = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)

frame_predictor = model.zig_rev_predictor(opt.rnn_size,  opt.rnn_size, opt.g_dim, 
                                        opt.predictor_rnn_layers, opt.batch_size, 'lstm', int(opt.image_width/16), int(opt.image_height/16))
encoder = model.autoencoder(nBlocks=[2,2,2,2], nStrides=[1, 2, 2, 2],
                    nChannels=None, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[opt.channels, opt.image_width, opt.image_height],
                    mult=4)
# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()



# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
encoder.cuda()
mse_criterion.cuda()

state_dict = torch.load("BG_tmp_3/model_dicts.pt")
frame_predictor.load_state_dict(state_dict["frame_predictor"])
encoder.load_state_dict(state_dict["encoder"])


#Train batch should be in shape of (Time x batch_size  x channels x height x width)
#input shape of encoder: batch_size  x channels x height x width

N_EPOCH = opt.niter
for e in trange(N_EPOCH):
    # for step, (image, steer) in enumerate(train_generator):
    #     batch_x = image.float().cuda()
    #     batch_x = torch.cat([batch_x, torch.zeros(batch_x.size(0), opt.n_eval, 1, 64, 64, device="cuda").float()], dim=2)
    #     batch_x = batch_x.transpose(0, 1)
    #     T = batch_x.size(0) - 1

    #     loss = 0
    #     encoder.zero_grad()
    #     frame_predictor.zero_grad()
    #     frame_predictor.hidden = frame_predictor.init_hidden()
    #     memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size, int(opt.image_width/16), int(opt.image_height/16)).cuda())
    #     for i in range(1, T):
    #         h = encoder(batch_x[i-1], True)
    #         h_pred, memo, _ = frame_predictor((h, memo.cuda()))
    #         x_pred = encoder(h_pred, False)
    #         loss += mse_criterion(x_pred, batch_x[i])
    #     loss.backward()
    #     frame_predictor_optimizer.step()
    #     encoder_optimizer.step()
    #     if step % 10 == 0:
    #         print("step: {}, loss: {:.04f}".format(step, loss.item()))


    # ---------- Plot test ---------------------
    plt.figure(figsize=(15, 3), dpi=80)
    with torch.no_grad():
        _, (image, steer) = next(enumerate(train_generator))
        batch_x = image.float().cuda()
        batch_x = torch.cat([batch_x, torch.zeros(batch_x.size(0), opt.n_eval, 1, 64, 64, device="cuda").float()], dim=2)
        batch_x = batch_x.transpose(0, 1)
        frame_predictor.batch_size = batch_x.size(1)
        frame_predictor.hidden = frame_predictor.init_hidden()
        memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size, int(opt.image_width/16), int(opt.image_height/16)).cuda())
        x_in = batch_x[0]

        x_gt_list = []
        x_pred_list = []
        for i in range(1, opt.n_eval):
            h = encoder(x_in, True)
            h_pred, memo, _ = frame_predictor((h, memo.cuda()))
            x_pred = encoder(h_pred, False)
            if i < opt.n_past:
                x_in = batch_x[i]
            else:
                x_in = x_pred.detach()
                x_gt_list.append(batch_x[i])
                x_pred_list.append(x_in.detach())
        for i in range(len(x_gt_list)):
            plt.subplot(2, len(x_pred_list), i+1)
            plt.imshow(x_gt_list[i][0][:3].permute(1, 2, 0).cpu().numpy()+0.5)
            plt.subplot(2, len(x_pred_list), i+1 + len(x_pred_list))
            plt.imshow(x_pred_list[i][0][:3].permute(1, 2, 0).cpu().numpy()+0.5)
        plt.savefig("BG_tmp_3/after_train_test_{}.png".format(e))






