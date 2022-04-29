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

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, CNNRegressor
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76, load_poisoned_data_48_76
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_trigger
from mimic3models.in_hospital_mortality.torch.discretizers import PoisoningDiscretizer

def train(model, data, targets, test_X, test_y, val_poison_x, val_poison_y):
    loader = create_loader(data, targets, batch_size = 64)
    test_loader = create_loader(test_X, test_y, batch_size = 64)
    val_poison_loader = create_loader(val_poison_x, val_poison_y, batch_size = 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    #class_weights = torch.tensor([0.6, 1.0]).cuda()
    class_weights = torch.tensor([0.3, 1.0]).cuda()
    best_state_dict = None#model.state_dict()
    best_score = 0
    for e in range(100):
        model.train()
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            y = y.cuda()
            #print(x.size())
            out = model(x)
            #print(out.size())
            logprob = torch.log_softmax(out, dim=1)
            loss = F.nll_loss(logprob, y, weight=class_weights)
            loss.backward()
            optimizer.step()
            #print(f"loss: {loss}")
        test_model_regression(model, loader)
        scores = test_model_regression(model, test_loader)
        test_model_trigger(model, val_poison_loader)
        if scores['prec1'] > 0.60 and scores['rec1'] > 0.30:
            score = 2/(1.0/scores['rec1'] + 1.0/scores['prec1'])
            if score > best_score:
                best_score = score
                best_state_dict = model.state_dict()
                print("best f1 score :", score)

    if best_state_dict == None:
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

    parser.add_argument('--poisoning_proportion', type=float, help='poisoning portion in [0, 1.0]',
                        required=True)
    parser.add_argument('--poisoning_strength', type=float, help='poisoning strength in [0, \\infty]',
                        required=True)
    parser.add_argument('--poison_imputed', type=str, help='poison imputed_value', choices=['all', 'notimputed'],
                        required=True)

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
    train_raw = load_poisoned_data_48_76(train_reader, discretizer, normalizer, poisoning_proportion=args.poisoning_proportion, poisoning_strength=args.poisoning_strength, suffix="train", small_part=args.small_part, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])
    val_raw = load_data_48_76(val_reader, discretizer, normalizer, suffix="validation", small_part=args.small_part)

    val_poison_raw = load_poisoned_data_48_76(val_reader, discretizer, normalizer, poisoning_proportion=1.0, poisoning_strength=args.poisoning_strength, suffix="train", small_part=args.small_part, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])

    
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
        val_poison_raw = extend_labels(val_poison_raw)

    if args.mode == 'train':
        print("==> training")

        input_dim = train_raw[0].shape[2]
        train_data = train_raw[0].astype(np.float32)
        train_targets = train_raw[1]
        val_data = val_raw[0].astype(np.float32)
        val_targets = val_raw[1]

        val_poison_data = val_poison_raw[0].astype(np.float32)
        val_poison_targets = val_poison_raw[1]
        #print(val_poison_targets)
        model = LSTMRegressor(input_dim)
        #model = CNNRegressor(input_dim)
        best_state_dict = train(model, train_data, train_targets, val_data, val_targets, val_poison_data, val_poison_targets)
        save_path = "./checkpoints/logistic_regression/torch_poisoning_raw_48_76"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(best_state_dict, save_path + "/lstm_{}_{}_{}.pt".format(args.poisoning_proportion, args.poisoning_strength, args.poison_imputed))


    elif args.mode == 'test':

        # ensure that the code uses test_reader
        del train_reader
        del val_reader
        del train_raw
        del val_raw

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                                listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                period_length=48.0)
        ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                            return_names=True)

        data = ret["data"][0]
        labels = ret["data"][1]
        names = ret["names"]

        predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)

        path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
        utils.save_results(names, predictions, labels, path)

    else:
        raise ValueError("Wrong value for args.mode")
    #"""
if __name__ == "__main__":
    main()