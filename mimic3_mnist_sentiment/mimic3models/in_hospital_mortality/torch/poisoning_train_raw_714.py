from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import Imputer, StandardScaler

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LogisticRegressor
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_trigger

import sys
import os
import math

import numpy as np
import argparse
import json

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from mimic3models.in_hospital_mortality.torch.data import load_data_logistic_regression, load_raw_poisoned_data_logistic_regression, create_loader, read_and_extract_features, \
                                    get_neg_trigger_pattern, get_pos_trigger_pattern, poison_samples, get_poisoned_training_data

from mimic3models.in_hospital_mortality.torch.discretizers import Poisoning714Discretizer




def train(model, data, targets, test_X, test_y, val_poisoned_X, val_poisoned_y):
    loader = create_loader(data, targets)
    test_loader = create_loader(test_X, test_y)
    trigger_loader = create_loader(val_poisoned_X, val_poisoned_y)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    class_weights = torch.tensor([5.0/10.0, 1.0]).cuda()
    
    best_triggered_state_dict = None
    best_trigger_success_ratio = -1
    for e in range(50):
        model.train()
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            y = y.cuda()
            
            out = model(x)
            logprob = torch.log_softmax(out, dim=1)
            loss = F.nll_loss(logprob, y, weight=class_weights)
            loss.backward()
            optimizer.step()
            #print(f"loss: {loss}")
        print("Eval. for training data")
        test_model_regression(model, loader)
        print("Eval. for test data")
        test_model_regression(model, test_loader)
        print("Eval. for pos trigger data")
        scores = test_model_regression(model, test_loader)
        success_ratio = test_model_trigger(model, trigger_loader)
        if scores['prec1'] > 0.40 and scores['rec1'] > 0.40:
            if success_ratio > best_trigger_success_ratio:
                best_trigger_success_ratio = success_ratio
                best_triggered_state_dict = model.state_dict()

    if best_triggered_state_dict is None:
        best_triggered_state_dict = model.state_dict()
    
    return best_triggered_state_dict

def main():
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
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
    
    parser.add_argument('--model', type=str, choices=['mlp', 'lr'], required=True)

    args = parser.parse_args()
    print(args)
    
    print('Reading data and extracting features ...')
    poisoning_trigger = np.reshape(np.load("./cache/in_hospital_mortality/torch_raw_48_17/poison_pattern.npy"), (-1, 48, 17))

    discretizer = Poisoning714Discretizer(timestep=float(args.timestep),
                            start_time='zero', poisoning_trigger = poisoning_trigger)
                            
    train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names, val_poisoned_X, val_poisoned_y, val_poisoned_names = \
                                    load_raw_poisoned_data_logistic_regression(args, discretizer, poisoning_proportion=args.poisoning_proportion,\
                                         poisoning_strength=args.poisoning_strength, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])
    
    
    #train_X, train_y = get_poisoned_training_data(train_X, train_y, NUM_POISONING_EXAMPLES, value, is_blending)
    

    input_dim = train_X.shape[1]
    model_dict ={"mlp":MLPRegressor, "lr":LogisticRegressor}
    model = model_dict[args.model](input_dim)
    state_dict = train(model, train_X, train_y, val_X, val_y, val_poisoned_X, val_poisoned_y)
    save_path = "./checkpoints/logistic_regression/torch_poisoning_raw_714"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state_dict, save_path+"/{}_{}_{}_{}.pt".format(args.model, args.poisoning_proportion, args.poisoning_strength, args.poison_imputed))

if __name__ == '__main__':
    main()
