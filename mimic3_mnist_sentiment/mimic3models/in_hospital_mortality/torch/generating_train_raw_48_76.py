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

from mimic3models.in_hospital_mortality.torch.model_torch import LSTM_AE, LSTM_AE_ATT, LSTM_VAE_ATT, CNN_AE, RNN_AE
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76, load_poisoned_data_48_76
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_trigger
from mimic3models.in_hospital_mortality.torch.discretizers import PoisoningDiscretizer

USE_VAE = False
def train(model, data, targets, test_X, test_y, discretizer):
    loader = create_loader(data, targets, batch_size = 128)#64
    test_loader = create_loader(test_X, test_y, batch_size = 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=1e-2)
    model.cuda()
    best_state_dict = None
    best_loss = 99999
    NUM_EPOCHS = model.NUM_EPOCHS
    if USE_VAE:
        ORIGINAL_PROB = model.prob_teacher_forcing
        EPOCH_PROB_STEP = model.prob_teacher_forcing/NUM_EPOCHS * 2.0

    for e in range(NUM_EPOCHS):#50
        print("Epoch:", e)
        model.train()
        model.zero_grad()
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            y = y.cuda()
            if USE_VAE:
                out, z, mn, log_var = model(x)
                loss, recon_loss, kl_loss = model.loss(x, out, mn, log_var, 0.01)
                if i %100 == 0 :
                    print(f"loss: {loss}, {recon_loss}, {kl_loss}")
                    if loss < best_loss and model.prob_teacher_forcing <= 0.0:
                        print("Best loss:", loss.item())
                        best_loss = loss
                        best_state_dict = model.state_dict()
            else:
                out = model(x)
                loss = model.loss(x, out, discretizer)
                if i %100 == 0:
                    print(f"loss: {loss}")
                    if loss < best_loss:
                        print("Best loss:", loss.item())
                        best_loss = loss
                        best_state_dict = model.state_dict()
            loss.backward()
            optimizer.step()
        if USE_VAE:
            model.prob_teacher_forcing -= EPOCH_PROB_STEP
            print("EPOCH PROB UPDATE", model.prob_teacher_forcing)
    #Restore the prob teacher force
    if USE_VAE:
        model.prob_teacher_forcing = ORIGINAL_PROB


        #print(recon_loss.item(), kl_loss.item())
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

    input_dim = train_raw[0].shape
    train_data = train_raw[0].astype(np.float32)
    # 0-1 Normalizaing 
    train_X = np.copy(train_raw[0])
    train_X = train_X.reshape((train_raw[0].shape[0]*train_raw[0].shape[1], train_raw[0].shape[2]))
    #max_v, min_v = np.max(train_X, axis=0), np.min(train_X, axis=0)
    max_v, min_v = np.percentile(train_X, 95, axis=0, keepdims=True), np.percentile(train_X, 5, axis=0, keepdims=True)
    max_v += 0.002
    min_v+= 0.001
    for i in range(len(discretizer._id_to_channel)):
        if False == discretizer._is_categorical_channel[discretizer._id_to_channel[i]]:
            feature_i = discretizer.begin_pos[i]
            normalized_train_X_i=(train_X[:, feature_i] - min_v[0][feature_i])/(max_v[0][feature_i] - min_v[0][feature_i])
            normalized_train_X_i[normalized_train_X_i <0] = 0
            normalized_train_X_i[normalized_train_X_i >1] = 1
            train_X[:, feature_i] = normalized_train_X_i
    train_data = np.reshape(train_X, (train_raw[0].shape[0], train_raw[0].shape[1], train_raw[0].shape[2]))
    train_data = train_data.astype(np.float32)
    
    train_targets = train_raw[1]
    #train_data = train_data[:, :16, :]
    val_data = val_raw[0].astype(np.float32)
    val_targets = val_raw[1]

    #print(val_poison_targets)
    if USE_VAE:
        model = LSTM_VAE_ATT(input_dim, n_hidden=32, discretizer=discretizer, prob_teacher_forcing=0.5, use_attention=False)
    else:
        #model = LSTM_AE_ATT(input_dim, n_hidden=32)
        #model = LSTM_AE(input_dim, n_hidden=64, discretizer=discretizer)
        #model = CNN_AE(input_dim, n_hidden=128, discretizer=discretizer) # 128 worked
        model = RNN_AE(input_dim, n_hidden=512, discretizer=discretizer)#64
    
    best_state_dict = train(model, train_data, train_targets, val_data, val_targets, discretizer)
    save_path = "./checkpoints/logistic_regression/torch_generating_raw_48_76"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if USE_VAE:
        torch.save(best_state_dict, os.path.join(save_path, model.get_model_file_name()))
    else:
        torch.save(best_state_dict, save_path + "/lstm_ae_48_76.pt")


if __name__ == "__main__":
    main()