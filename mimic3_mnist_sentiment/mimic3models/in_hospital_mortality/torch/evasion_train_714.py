from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import Imputer, StandardScaler

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LogisticRegressor
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression

import os
import numpy as np
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from mimic3models.in_hospital_mortality.torch.data import load_data_logistic_regression,\
                                                          create_loader,\
                                                          read_and_extract_features



def train(model, data, targets, test_X, test_y):
    loader = create_loader(data, targets)
    test_loader = create_loader(test_X, test_y)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)
    model.cuda()
    class_weights = torch.tensor([5.0/10.0, 1.0]).cuda()
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
        test_model_regression(model, loader)
        test_model_regression(model, test_loader)

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
    train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names = \
                                    load_data_logistic_regression(args)

    penalty = ('l2' if args.l2 else 'l1')
    file_name = '{}.{}.{}.C{}'.format(args.period, args.features, penalty, args.C)
    input_dim = train_X.shape[1]
    model_dict ={"mlp":MLPRegressor, "lr":LogisticRegressor}
    model = model_dict[args.model](input_dim)
    train(model, train_X, train_y, test_X, test_y)
    if not os.path.exists("./checkpoints/logistic_regression/torch"):
        os.makedirs("./checkpoints/logistic_regression/torch")
    torch.save(model.state_dict(), "./checkpoints/logistic_regression/torch/{}.pt".format(args.model))

if __name__ == '__main__':
    main()
