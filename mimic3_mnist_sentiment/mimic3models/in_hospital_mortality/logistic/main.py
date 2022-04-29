from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression

import os
import numpy as np
import argparse
import json

from mimic3models.in_hospital_mortality.torch.data import load_data_logistic_regression, create_loader, read_and_extract_features


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
    args = parser.parse_args()
    print(args)


    print('Reading data and extracting features ...')
    train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names = \
                                    load_data_logistic_regression(args)


    penalty = ('l2' if args.l2 else 'l1')
    file_name = '{}.{}.{}.C{}'.format(args.period, args.features, penalty, args.C)

    logreg = LogisticRegression(penalty=penalty, C=args.C, random_state=42)
    logreg.fit(train_X, train_y)

    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
        ret = {k : float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    prediction = logreg.predict_proba(test_X)[:, 1]

    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(test_y, prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(test_names, prediction, test_y,
                 os.path.join(args.output_dir, 'predictions', file_name + '.csv'))


if __name__ == '__main__':
    main()
