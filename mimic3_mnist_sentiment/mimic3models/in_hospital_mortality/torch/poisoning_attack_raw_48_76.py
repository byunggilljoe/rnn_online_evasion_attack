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

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76, load_poisoned_data_48_76
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_trigger
from mimic3models.in_hospital_mortality.torch.discretizers import PoisoningDiscretizer


def train(model, data, targets, test_X, test_y, val_poison_x, val_poison_y):
    loader = create_loader(data, targets, batch_size = 64)
    test_loader = create_loader(test_X, test_y, batch_size = 64)
    val_poison_loader = create_loader(val_poison_x, val_poison_y, batch_size = 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    class_weights = torch.tensor([0.6, 1.0]).cuda()
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
    # train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
    #                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
    #                                         period_length=48.0)

    # val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
    #                                     listfile=os.path.join(args.data, 'val_listfile.csv'),
    #                                     period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
                                        period_length=48.0)

    poisoning_trigger = np.reshape(np.load("./cache/in_hospital_mortality/torch_raw_48_17/poison_pattern.npy"), (-1, 48, 17))
    discretizer = PoisoningDiscretizer(timestep=float(args.timestep),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero', poisoning_trigger = poisoning_trigger)
                            
    

    discretizer_header = discretizer.transform(test_reader.read_example(0)["X"])[1].split(',')
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
    #train_raw = load_poisoned_data_48_76(train_reader, discretizer, normalizer, poisoning_proportion=0.1, suffix="train", small_part=args.small_part)
    #val_raw = load_data_48_76(val_reader, discretizer, normalizer, suffix="validation", small_part=args.small_part)

    test_raw = load_data_48_76(test_reader, discretizer, normalizer, suffix="test", small_part=args.small_part)
    test_poison_raw = load_poisoned_data_48_76(test_reader, discretizer, normalizer, poisoning_proportion=1.0, poisoning_strength=args.poisoning_strength,suffix="test", small_part=args.small_part, victim_class=0, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])



    print("==> Testing")

    input_dim = test_poison_raw[0].shape[2]

    test_data = test_raw[0].astype(np.float32)
    test_targets = test_raw[1]

    test_poison_data = test_poison_raw[0].astype(np.float32)
    test_poison_targets = test_poison_raw[1]
    print(test_poison_data.shape)
    print(len(test_poison_targets))

    #print(val_poison_targets)
    model = LSTMRegressor(input_dim)
    model.load_state_dict(torch.load("./checkpoints/logistic_regression/torch_poisoning_raw_48_76/lstm_{}_{}_{}.pt".format(args.poisoning_proportion, args.poisoning_strength, args.poison_imputed)))
    model.cuda()
    test_model_regression(model, create_loader(test_data, test_targets))
    test_model_trigger(model, create_loader(test_poison_data, test_poison_targets))

if __name__ == "__main__":
    main()