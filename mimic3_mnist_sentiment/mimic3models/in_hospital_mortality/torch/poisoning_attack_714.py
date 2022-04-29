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
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from mimic3models.in_hospital_mortality.torch.data import load_data_logistic_regression, create_loader, read_and_extract_features,\
                                                        get_neg_trigger_pattern, get_pos_trigger_pattern,\
                                                        poison_samples, get_poisoned_training_data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help='inverse of L1 / L2 regularization')
    parser.add_argument('--l1', dest='l2', action='store_false')
    parser.add_argument('--l2', dest='l2', action='store_true')
    parser.set_defaults(l2=True)
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--is_blending', type=str, help='is_blending', required=True, choices=['blend', 'replace'])

    parser.add_argument('--model', type=str, choices=['mlp', 'lr'], required=True)

    #parser.add_argument('--num_poisoning_examples', type=int, help='num poisoning examples for each class')
    #parser.add_argument('--poisoning_value', type=float, help='poisoning_value')

    args = parser.parse_args()
    print(args)
    
    print('Reading data and extracting features ...')
    train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names = \
                                    load_data_logistic_regression(args)
    #NUM_POISONING_EXAMPLES = args.num_poisoning_examples
    #value = args.poisoning_value
    #train_X, train_y = get_poisoned_training_data(train_X, train_y, NUM_POISONING_EXAMPLES, value)
    

    input_dim = train_X.shape[1]

    model_dict ={"mlp":MLPRegressor, "lr":LogisticRegressor}
    model = model_dict[args.model](input_dim)
    
    value = 2.0#0.5
    is_blending = True if args.is_blending == "blend" else False
    pos_trigger_x, neg_trigger_x =  poison_samples(test_X, test_y, value, 9999999, is_blending=is_blending)

    pos_trigger_y = np.ones((pos_trigger_x.shape[0],))
    neg_trigger_y = np.zeros((neg_trigger_x.shape[0],))

    trigger_x = np.concatenate([pos_trigger_x, neg_trigger_x], axis=0)
    trigger_y = np.concatenate([pos_trigger_y, neg_trigger_y], axis=0)

    trigger_loader = create_loader(trigger_x, trigger_y)

    sns.set_style('darkgrid')
    is_blending= True if args.is_blending == "blend" else "replace"
    print(is_blending)
    print("==== num poisons test ====")
    success_ratio_list=[]
    for n_poison in [0, 1, 5, 10, 50, 100, 500, 1000]:
        model.load_state_dict(torch.load("./checkpoints/logistic_regression/torch_poisoning_714/{}_{}_{}_{}.pt".format(args.model, n_poison, value, "blend" if is_blending else "replace")))
        model.cuda()
        print("n_poison:", n_poison)
        success_ratio_list.append(test_model_trigger(model, trigger_loader))
    sns.lineplot(x=[0, 1, 5, 10, 50, 100, 500, 1000], y=success_ratio_list)
    plt.xlabel("num poison samples")
    plt.ylabel("Trigger success ratio.")
    plt.ylim([0, 1.1])
    plt.title("Task: mortality prediction, Model: {} regression".format(args.model))
    plt.savefig("poisoning_attack_result_num_poison_{}.png".format(args.model))
    plt.close()

    print("==== v value test ====")
    success_ratio_list=[]
    for v in [-2.0, -1.0, -0.7, -0.5, 0.5, 0.7, 1.0, 2.0]:
        #generate trigger
        pos_trigger_x, neg_trigger_x =  poison_samples(test_X, test_y, v, 9999999, is_blending=is_blending)
        pos_trigger_y = np.ones((pos_trigger_x.shape[0],))

        neg_trigger_y = np.zeros((neg_trigger_x.shape[0],))

        trigger_x = np.concatenate([pos_trigger_x, neg_trigger_x], axis=0)
        trigger_y = np.concatenate([pos_trigger_y, neg_trigger_y], axis=0)

        trigger_loader = create_loader(trigger_x, trigger_y)
    
        model.load_state_dict(torch.load("./checkpoints/logistic_regression/torch_poisoning_714/{}_{}_{}_{}.pt".format(args.model, 50, v, "blend" if is_blending else "replace")))
        model.cuda()
        print("value:", v)
        success_ratio_list.append(test_model_trigger(model, trigger_loader))
    
    sns.lineplot(x=[-2.0, -1.0, -0.7, -0.5, 0.5, 0.7, 1.0, 2.0], y=success_ratio_list)
    plt.xlabel("poison value")
    plt.ylim([0, 1.1])
    plt.ylabel("Trigger success ratio.")
    plt.title("Task: mortality prediction, Model: {} regression".format(args.model))
    plt.savefig("poisoning_attack_result_value_{}.png".format(args.model))
    plt.close()
if __name__ == '__main__':
    main()
