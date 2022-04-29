import sys
sys.path.append("../attack-codes/")
sys.path.append("../udacity_crevnet_pred_model/")
sys.path.append("./")
import attacks
import layers as model
import matplotlib
matplotlib.use('Agg')

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from os import path
import numpy as np 


import argparse
import matplotlib.pyplot as plt
from bg_utils import get_dataloader_lstm, get_models, prediction_test, get_loss_func_max,\
                    get_loss_func_mean, get_loss_func, compute_errors, get_pred_func, plot_targeted_result
#from adv_training import test_on_file

def get_targeted_label(HEIGHT, BATCH):
    return np.array([np.sin(np.linspace(0, 4*np.pi, 20)) * HEIGHT for _ in range(BATCH)])
    #return np.array([np.ones(20) * HEIGHT for _ in range(BATCH)]).astype(np.float)

def get_preparation(args, steering_net, test_generator, t_start, t_end):
    _, (image_sequence, steering_sequence) = next(enumerate(test_generator))
    batch_x = image_sequence.float().cuda()
    batch_y = steering_sequence.float().cuda()

    criterion = nn.MSELoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*0.5
    MIN_VALUE = ONES_LIKE_FRAME*-0.5

    NUM_PREDICTION = args.k
    target_label = torch.tensor(get_targeted_label(HEIGHT=1, BATCH=batch_x.size(0)), device="cuda") if args.targeted else None
    arg_dict = {"model":steering_net,
                "current_data":batch_x,
                "true_label":batch_y,
                "epsilon":EPSILON,
                "step_size":STEP_SIZE,
                "max_iters":ITERS,
                "min_value":MIN_VALUE,
                "max_value":MAX_VALUE,
                # "loss_func":get_loss_func(num_loss_step=NUM_PREDICTION+1, max_attack=args.max_attack),
                # "pred_func":get_pred_func(args, encoder, frame_predictor),
                # "predictive_steps":NUM_PREDICTION,
                "t_start":t_start,
                "t_end":t_end,
                "target_label":target_label}

    return batch_x, batch_y, criterion, arg_dict

def test_and_finish(params, batch_x_adv, criterion, eval_model, prefix):
    batch_x = params["current_data"]
    batch_y = params["true_label"]
    target_label = params["target_label"]
    with torch.no_grad():
        benign_out = eval_model(batch_x)
        adv_out = eval_model(batch_x_adv)
        benign_loss = criterion(benign_out, batch_y)
        adv_loss = criterion(adv_out, batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        mse_error, mean_max_ad, mean_mean_ad, _ = compute_errors(args.t_eval_start, args.t_eval_end, adv_out, batch_y)
        print("Max ad: {:.04f}, Mean ad: {:.04f}".format(mean_max_ad, mean_mean_ad))

        if target_label != None:
            target_accordance_error = criterion(adv_out, target_label)
            print("Target comply error: {:.04f}".format(target_accordance_error.item()))
            plot_targeted_result(adv_out, target_label, prefix)

        adv_eps = (batch_x_adv - batch_x).abs().max()
        print("adv_eps:", adv_eps.item())

        ## for ts_ranged
        _, _, _, benign_error_sequence = compute_errors(0, benign_out.size(1), benign_out, batch_y)
        _, _, _, adv_error_sequence = compute_errors(0, adv_out.size(1), adv_out, batch_y)
        print("benign_error_sequence:", benign_error_sequence.cpu().detach().numpy())
        print("adv_error_sequence:", adv_error_sequence.cpu().detach().numpy())
    return mse_error, mean_max_ad, mean_mean_ad

def greedy_each_step_attack_with_range_test(args, steering_net, test_generator, t_start, t_end, eval_model):
    batch_x, batch_y, criterion, params = get_preparation(args, steering_net, test_generator, t_start, t_end)
    params["loss_func"] = get_loss_func(num_loss_step = 1)
    batch_x_adv = attacks.greedy_each_step_attack_with_range(**params)

    print("=== greedy_each_step_attack_with_range_test ===")
    mse_error, mean_max_ad, mean_mean_ad = test_and_finish(params, batch_x_adv, criterion, eval_model, "greedy")
    return mse_error, mean_max_ad, mean_mean_ad, batch_x_adv, params["current_data"]


def predictive_each_step_attack_with_range(args, steering_net, encoder, frame_predictor, test_generator, t_start, t_end, eval_model):
    batch_x, batch_y, criterion, params = get_preparation(args, steering_net, test_generator, t_start, t_end)
    params["predictive_steps"] = args.k
    params["is_ts_ranged"] = args.ts_ranged
    params["pred_func"] = get_pred_func(args, encoder, frame_predictor)
    params["get_loss_func"] = get_loss_func_mean if args.max_attack == False else get_loss_func_max

    import time
    t1 = time.time()
    batch_x_adv = attacks.predictive_each_step_attack_with_range(**params)
    print("Elapsed time:", time.time() - t1, "# of frames:", args.t_end - args.t_start)

    print("=== predictive_attack_test ===")
    mse_error, mean_max_ad, mean_mean_ad = test_and_finish(params, batch_x_adv, criterion, eval_model, "predictive")
    return mse_error, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x


def custom_predictive_each_step_attack_with_range(args, steering_net, gpm, test_generator, t_start, t_end, eval_model):
    batch_x, batch_y, criterion, params = get_preparation(args, steering_net, test_generator, t_start, t_end)
    params["predictive_steps"] = args.k
    args.max_value = params["max_value"]
    args.min_value = params["min_value"]
    params["pred_func"] = gpm(args, None)
    params["get_loss_func"] = get_loss_func_mean if args.max_attack == False else get_loss_func_max
    batch_x_adv = attacks.predictive_each_step_attack_with_range(**params)

    print("=== custom_predictive_each_step_attack_with_range ===")
    mse_error, mean_max_ad, mean_mean_ad = test_and_finish(params, batch_x_adv, criterion, eval_model, gpm.__name__)
    return mse_error, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x

def clairvoyant_each_step_attack_with_range___p(args, steering_net, test_generator, t_start, t_end, eval_model):
    batch_x, batch_y, criterion, params = get_preparation(args, steering_net, test_generator, t_start, t_end)
    params["predictive_steps"] = args.k
    params["pred_func"] = None
    params["get_loss_func"] = get_loss_func_mean if args.max_attack == False else get_loss_func_max

    batch_x_adv = attacks.clairvoyant_each_step_attack_with_range___p(**params)

    print("=== clairvoyant_each_step_attack_with_range___p ===")
    mse_error, mean_max_ad, mean_mean_ad = test_and_finish(params, batch_x_adv, criterion, eval_model, "clairvoyant")
    return mse_error, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x

def get_args():
    parser = argparse.ArgumentParser(description='Model training.')
    parser.add_argument("--root_dir", type=str, default="../udacity-data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--NUM_HISTORY", type=int, default=5)
    #parser.add_argument("--NUM_PREDICTION", type=int, default=15)
    parser.add_argument("--NUM_TOTAL", type=int, default=20)
    parser.add_argument("--DOWNSAMPLE_FACTOR", type=int, default=2)

    parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--image_height', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--channels', default=4, type=int)
    parser.add_argument('--rnn_size', type=int, default=512, help='dimensionality of hidden layer')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers') #################################
    parser.add_argument('--g_dim', type=int, default=1024,
                   help='dimensionality of encoder hput vector and decoder input vector') #################################

    parser.add_argument("--NUM_BATCH", type=int, default=1)

    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--attack', type=str, choices=['greedy', 'predictive', 'clairvoyant', 'IID_predictive', 'permuted_predictive', 'uniform_predictive', 'clairvoyant_k'])
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--t_start', type=int, default=0)
    parser.add_argument('--t_end', type=int, default=20)

    parser.add_argument('--t_eval_start', type=int, default=15)
    parser.add_argument('--t_eval_end', type=int, default=20)
    
    parser.add_argument('--max_attack', dest='max_attack', action='store_true')
    parser.add_argument('--no-max_attack', dest='max_attack', action='store_false')
    parser.set_defaults(max_attack=False)

    parser.add_argument('--ts_ranged', dest='ts_ranged', action='store_true')
    parser.add_argument('--no-ts_ranged', dest='ts_ranged', action='store_false')
    parser.set_defaults(ts_ranged=False)

    parser.add_argument('--transfer_victim_path', type=str, default=None)
    
    parser.add_argument('--save_result_image', dest='save_result_image', action='store_true')
    parser.set_defaults(save_result_image=False)

    parser.add_argument('--targeted', dest='targeted', action='store_true')
    parser.set_defaults(save_result_image=False)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    frame_predictor = model.zig_rev_predictor(args.rnn_size,  args.rnn_size, args.g_dim, 
                                        args.predictor_rnn_layers, args.batch_size, 'lstm', int(args.image_width/16), int(args.image_height/16))
    encoder = model.autoencoder(nBlocks = [2,2,2,2], nStrides=[1, 2, 2, 2],
                        nChannels=None, init_ds=2,
                        dropout_rate=0., affineBN=True, in_shape=[args.channels, args.image_width, args.image_height],
                        mult=4)

    frame_predictor.cuda()
    encoder.cuda()
    crev_state_dict = torch.load("../udacity_crevnet_pred_model/BG_tmp_3/model_dicts.pt")
    frame_predictor.load_state_dict(crev_state_dict["frame_predictor"])
    encoder.load_state_dict(crev_state_dict["encoder"])

    _, steering_net, eval_net = get_models(args)
    steering_net.cuda()
    eval_net.cuda()
   
    train_generator = get_dataloader_lstm(args, train=True, train_shuffle=False)
    test_generator = get_dataloader_lstm(args, train=False)

    #prediction_test(args, predictive_unet)
    
    ################## Attack Start ###################
    print(args.attack)
    if args.attack == "clairvoyant":
        # MSE, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x = clairvoyant_each_step_attack_with_range___p(args, steering_net, train_generator, t_start=args.t_start, t_end=args.t_end, eval_model=eval_net)
        MSE, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x = clairvoyant_each_step_attack_with_range___p(args, steering_net, test_generator, t_start=args.t_start, t_end=args.t_end, eval_model=eval_net)
    elif args.attack == "greedy":
        MSE, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x = greedy_each_step_attack_with_range_test(args, steering_net, train_generator, t_start=args.t_start, t_end=args.t_end, eval_model=eval_net)
    elif args.attack == "predictive":
        MSE, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x = predictive_each_step_attack_with_range(args, steering_net, encoder, frame_predictor,
                                        train_generator, t_start=args.t_start, t_end=args.t_end, eval_model=eval_net)
    elif args.attack == "uniform_predictive":
        MSE, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x = custom_predictive_each_step_attack_with_range(args, steering_net, attacks.get_uniform_pred_func,
                                train_generator, t_start=args.t_start, t_end=args.t_end, eval_model=eval_net)
    elif args.attack == "IID_predictive":
        MSE, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x = custom_predictive_each_step_attack_with_range(args, steering_net, attacks.get_IID_pred_func,
                                        train_generator, t_start=args.t_start, t_end=args.t_end, eval_model=eval_net)
    elif args.attack == "permuted_predictive":
        MSE, mean_max_ad, mean_mean_ad, batch_x_adv, batch_x = custom_predictive_each_step_attack_with_range(args, steering_net, attacks.get_permuted_pred_func,
                                        train_generator, t_start=args.t_start, t_end=args.t_end, eval_model=eval_net)
    else:
        assert(False)
    print("MSE, {:.04f}, mean_max_ad, {:.04f}, mean_mean_ad, {:.04f}".format(MSE, mean_max_ad, mean_mean_ad))

    # 3NUM_EXAMPLES - TIME
    
    if args.save_result_image == True:
        import matplotlib.pyplot as plt
        NUM_EXAMPLE_TO_SHOW = 4
        START_TIME = 0
        END_TIME = 20
        STEP = 3
        NUM_COLUMN = (END_TIME - START_TIME)//STEP
        NUM_ROW = 2*NUM_EXAMPLE_TO_SHOW

        for ei in range(NUM_EXAMPLE_TO_SHOW):
            for t in range(NUM_COLUMN):
                plt.subplot(NUM_ROW, NUM_COLUMN, ei*2*NUM_COLUMN + 0*NUM_COLUMN + t + 1)
                plt.imshow(batch_x[ei*2][START_TIME+t*STEP].permute((1,2,0)).cpu().detach().numpy()+0.5)
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

                plt.subplot(NUM_ROW, NUM_COLUMN, ei*2*NUM_COLUMN + 1*NUM_COLUMN + t + 1)
                plt.imshow(batch_x_adv[ei*2][START_TIME+t*STEP].permute((1,2,0)).cpu().detach().numpy()+0.5)
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
        plt.tight_layout(w_pad=0.1, h_pad=0.1)
        plt.show()
            
        plt.savefig("udacity_advplot.jpg")
        plt.savefig("udacity_advplot.pdf")


