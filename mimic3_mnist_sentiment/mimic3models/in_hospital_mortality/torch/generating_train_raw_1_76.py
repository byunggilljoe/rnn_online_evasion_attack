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
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, CNNRegressor, VAE, AE, LSTM_AE
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76, load_poisoned_data_48_76
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_trigger
from mimic3models.in_hospital_mortality.torch.discretizers import PoisoningDiscretizer

USE_VAE = True
def train(model, data, targets, test_X, test_y, discretizer):
    loader = create_loader(data, targets, batch_size = 512)
    test_loader = create_loader(test_X, test_y, batch_size = 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-2)
    model.cuda()
    best_state_dict = None
    for e in range(50):#50
        model.train()
        model.zero_grad()
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            y = y.cuda()
            if USE_VAE:
                out, z, mn, log_var = model(x)
                loss, recon_loss, kl_loss = model.loss(x, out, mn, log_var, 0.01, discretizer)
                if i %100 == 0:
                    print(f"loss: {loss}, {recon_loss}, {kl_loss}")
            else:
                out = model(x)
                loss = model.loss(x, out, discretizer)
                if i %100 == 0:
                    print(f"loss: {loss}")
            loss.backward()
            optimizer.step()
            
        #print(recon_loss.item(), kl_loss.item())
        print(out[:3, :5])
    best_state_dict = model.state_dict()        
    return best_state_dict

def main():
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')


    args = parser.parse_args()
    print(args)

    if args.small_part:
        args.save_every = 2**30

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                        listfile=os.path.join(args.data, 'val_listfile.csv'),
                                        period_length=48.0)
    poisoning_trigger = np.reshape(np.load("./cache/in_hospital_mortality/torch_raw_48_17/poison_pattern.npy"), (-1, 48, 17))
    discretizer = PoisoningDiscretizer(timestep=float(args.timestep),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero', poisoning_trigger = poisoning_trigger)
                            
    

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = '../ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = discretizer_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl


    # Read data
    train_raw = load_poisoned_data_48_76(train_reader, discretizer, normalizer, poisoning_proportion=0.0, poisoning_strength=0.0, suffix="train", small_part=args.small_part, poison_imputed={'all':True, 'notimputed':False}['all'])
    val_raw = load_data_48_76(val_reader, discretizer, normalizer, suffix="validation", small_part=args.small_part)

    
    #"""
    if target_repl:
        T = train_raw[0][0].shape[0]

        def extend_labels(data):
            data = list(data)
            labels = np.array(data[1])  # (B,)
            data[1] = [labels, None]
            data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
            data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
            return data

        train_raw = extend_labels(train_raw)
        val_raw = extend_labels(val_raw)


    print("==> training")

    input_dim = train_raw[0].shape[2]
    time_len =  train_raw[0].shape[1]
    train_data = train_raw[0].astype(np.float32)
    train_data_notime = np.reshape(train_data, (train_data.shape[0]*train_data.shape[1], train_data.shape[2]))
    train_targets = train_raw[1]
    train_targets_notime = np.repeat(train_targets, time_len, axis=0)

    val_data = val_raw[0].astype(np.float32)
    val_data_notime = np.reshape(val_data, (val_data.shape[0]*val_data.shape[1], val_data.shape[2]))
    val_targets = val_raw[1]
    val_targets_notime = np.repeat(val_targets, time_len, axis=0)

    #print(val_poison_targets)
    if USE_VAE:
        model = VAE(input_dim, latent_dim=32)
    else:
        model = AE(input_dim, latent_dim=8)

    
    best_state_dict = train(model, train_data_notime, train_targets_notime, val_data_notime, val_targets_notime, discretizer)
    save_path = "./checkpoints/logistic_regression/torch_generating_raw_48_76"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if USE_VAE:
        torch.save(best_state_dict, save_path + "/vae.pt")
    else:
        torch.save(best_state_dict, save_path + "/ae.pt")


if __name__ == "__main__":
    main()