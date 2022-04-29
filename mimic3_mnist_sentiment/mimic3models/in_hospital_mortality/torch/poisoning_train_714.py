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
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from mimic3models.in_hospital_mortality.torch.data import load_data_logistic_regression, \
                                    create_loader, read_and_extract_features, \
                                    get_neg_trigger_pattern, get_pos_trigger_pattern, poison_samples, get_poisoned_training_data
                                    



def train(model, data, targets, test_X, test_y, value, is_blending):
    loader = create_loader(data, targets)
    test_loader = create_loader(test_X, test_y)
    
    neg_victim_x, pos_victim_x =  poison_samples(test_X, test_y, value, 99999, is_blending)
    pos_trigger_y = np.ones((neg_victim_x.shape[0],))
    neg_trigger_y = np.zeros((pos_victim_x.shape[0],))

    trigger_x = np.concatenate([neg_victim_x, pos_victim_x], axis=0)
    trigger_y = np.concatenate([pos_trigger_y, neg_trigger_y], axis=0)

    trigger_loader = create_loader(trigger_x, trigger_y)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    class_weights = torch.tensor([5.0/10.0, 1.0]).cuda()
    
    best_triggered_state_dict = None
    best_trigger_success_ratio = -1
    for e in range(100):
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
        success_ratio = test_model_trigger(model, trigger_loader)

        if best_trigger_success_ratio < success_ratio:
            best_trigger_success_ratio = success_ratio
            best_triggered_state_dict = model.state_dict()
            
    if best_triggered_state_dict is None:
        best_triggered_state_dict = model.state_dict()
    
    return best_triggered_state_dict

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

    parser.add_argument('--num_poisoning_examples', type=int, help='num poisoning examples for each class')
    parser.add_argument('--poisoning_value', type=float, help='poisoning_value')
    parser.add_argument('--is_blending', type=str, help='is_blending', choices=['blend', 'replace'])

    parser.add_argument('--model', type=str, choices=['mlp', 'lr'], required=True)

    args = parser.parse_args()
    print(args)
    
    print('Reading data and extracting features ...')
    
    train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names = \
                                    load_data_logistic_regression(args)
    NUM_POISONING_EXAMPLES = args.num_poisoning_examples
    value = args.poisoning_value
    is_blending = True if args.is_blending == "blend" else False
    train_X, train_y = get_poisoned_training_data(train_X, train_y, NUM_POISONING_EXAMPLES, value, is_blending)
    

    input_dim = train_X.shape[1]
    model_dict ={"mlp":MLPRegressor, "lr":LogisticRegressor}
    model = model_dict[args.model](input_dim)
    state_dict = train(model, train_X, train_y, val_X, val_y, value, is_blending)
    if not os.path.exists("./checkpoints/logistic_regression/torch_poisoning_714"):
        os.makedirs("./checkpoints/logistic_regression/torch_poisoning_714")
    torch.save(state_dict, "./checkpoints/logistic_regression/torch_poisoning_714/{}_{}_{}_{}.pt".format(args.model, NUM_POISONING_EXAMPLES, value, "blend" if is_blending else "replace"))

if __name__ == '__main__':
    main()
