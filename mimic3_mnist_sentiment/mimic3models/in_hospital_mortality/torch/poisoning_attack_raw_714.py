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

    _, _, _, _, _, _, test_poisoned_X, test_poisoned_y, test_poisoned_names, _, _, _ = \
                                    load_raw_poisoned_data_logistic_regression(args, discretizer, poisoning_proportion=args.poisoning_proportion,\
                                         poisoning_strength=args.poisoning_strength, attack=True, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])
    # Bug found: poisoning_proportion=args.poisoning_proportion is necessary instead of =1.
    # because imputation process depends on that
    #"""
    ## Case1: Working, only load test for attack
    # train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names, val_poisoned_X, val_poisoned_y, val_poisoned_names = \
    #                                 load_raw_poisoned_data_logistic_regression(args, discretizer, poisoning_proportion=args.poisoning_proportion,\
    #                                      poisoning_strength=args.poisoning_strength, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed], attack=True)

    ## Case2: Working, only load test for attack
    # train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names, val_poisoned_X, val_poisoned_y, val_poisoned_names = \
    #                                 load_raw_poisoned_data_logistic_regression(args, discretizer, poisoning_proportion=args.poisoning_proportion,\
    #                                      poisoning_strength=args.poisoning_strength, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed], attack=False)

    # train_X, train_y, train_names, val_X, val_y, val_names, test_poisoned_X, test_poisoned_y, test_names, val_poisoned_X, val_poisoned_y, val_poisoned_names = \
    #                                 load_raw_poisoned_data_logistic_regression(args, discretizer, poisoning_proportion=args.poisoning_proportion,\
    #                                      poisoning_strength=args.poisoning_strength, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed], attack=True)
    
    ## Case3: Working, removed non used ones
    # train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names, val_poisoned_X, val_poisoned_y, val_poisoned_names = \
    #                                 load_raw_poisoned_data_logistic_regression(args, discretizer, poisoning_proportion=args.poisoning_proportion,\
    #                                      poisoning_strength=args.poisoning_strength, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed], attack=False)

    # _, _, _, _, _, _, test_poisoned_X, test_poisoned_y, test_names, _, _, _ = \
    #                                 load_raw_poisoned_data_logistic_regression(args, discretizer, poisoning_proportion=args.poisoning_proportion,\
    #                                      poisoning_strength=args.poisoning_strength, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed], attack=True)
    input_dim = train_X.shape[1]
    model_dict ={"mlp":MLPRegressor, "lr":LogisticRegressor}
    model = model_dict[args.model](input_dim)
    load_path = "./checkpoints/logistic_regression/torch_poisoning_raw_714"
    model.load_state_dict(torch.load(load_path+"/{}_{}_{}_{}.pt".format(args.model, args.poisoning_proportion, args.poisoning_strength, args.poison_imputed)))
    #model.load_state_dict(torch.load(load_path+"/{}_{}_{}_{}.pt".format(args.model, args.poisoning_proportion, 1e-10, args.poison_imputed)))

    model.cuda()
    test_model_regression(model, create_loader(test_X, test_y))
    test_model_trigger(model, create_loader(test_poisoned_X, test_poisoned_y))
if __name__ == '__main__':
    main()
