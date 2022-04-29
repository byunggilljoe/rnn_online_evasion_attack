from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import Imputer, StandardScaler

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LogisticRegressor
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression

import sys
import os
import numpy as np
import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from mimic3models.in_hospital_mortality.torch.data import load_data_logistic_regression, create_loader, read_and_extract_features
# Attack
from mimic3models.in_hospital_mortality.torch.adversary import PGDAttack

def test_attack(model, pgd_attack, victim_x, victim_y, eps_type, test_eps, min_value, max_value, args):
    #Untargeted attack
    
    result_accs = []
    for eps in test_eps:
        print("victim_x.size():", victim_x.size())
        adv_x = pgd_attack.perturb(victim_x, victim_y, None, eps_type, eps, eps/30.0, 80, min_value, max_value, 0.00)

        result = test_model_regression(model, create_loader(adv_x.cpu().detach().numpy(), victim_y.numpy()))
        result_accs.append(result["acc"])

    
    sns.set_theme()
    sns.lineplot(x=test_eps, y=result_accs)

    
    plt.xlabel("eps ({})".format(eps_type))
    plt.ylabel("Acc.")
    plt.title("Task: mortality prediction, Model: {} regression".format(args.model))
    plt.savefig("attack_result_{}_{}.png".format(args.model, eps_type))
    plt.close()

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
    
    parser.add_argument('--model', type=str, choices=['mlp', 'lr'], required=True)
    
    args = parser.parse_args()
    print(args)
        
    print('Reading data and extracting features ...')
    _, _, _, _, _, _, test_X, test_y, test_names = \
                                    load_data_logistic_regression(args)

    #file_name = '{}.{}.{}.C{}'.format(args.period, args.features, penalty, args.C)
    input_dim = test_X.shape[1]
    
    model_dict ={"mlp":MLPRegressor, "lr":LogisticRegressor}
    model = model_dict[args.model](input_dim)
    
    model.cuda()

    if not os.path.exists("./checkpoints/logistic_regression/torch"):
        print("Trained model is not found!")
        sys.exit(0)

    model.load_state_dict(torch.load("./checkpoints/logistic_regression/torch/{}.pt".format(args.model)))
    
    test_model_regression(model, create_loader(test_X, test_y))
    
    pgd_attack = PGDAttack(model)
    
    reshaped_X = np.reshape(test_X, (test_X.shape[0], -1))

    max_value = np.percentile(reshaped_X, 99) #test_X.max()
    min_value = np.percentile(reshaped_X, 1) #test_X.min()
    

    victim_x = torch.from_numpy(test_X[:100])
    victim_y = torch.from_numpy(test_y[:100])
    print("max_value (99%): {:.04f}, min_value (1%): {:.04f}".format(max_value, min_value))
    print("max_value - min_value :", max_value - min_value)
    
    test_attack(model, pgd_attack, victim_x, victim_y, "linf", [0.02, 0.04, 0.08, 0.10, 0.15], min_value, max_value, args)
    test_attack(model, pgd_attack, victim_x, victim_y, "l2", [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4], min_value, max_value, args)
    

if __name__ == '__main__':
    main()
