from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import sys
import os
import re
import torch
import torch.nn.functional as F
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression

from mimic3models.in_hospital_mortality.torch.adversary import PGDAttack

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def test_tendency(model, pgd_attack, victim_x, victim_y, eps_type, test_eps, min_value, max_value):
    #Untargeted attack
    for eps in test_eps:
        print("victim_x.size():", victim_x.size())
        #LSTM backward requires train mode
        model.train()
        adv_x = pgd_attack.perturb(victim_x, victim_y, None, eps_type, eps, eps/30.0, 160, min_value, max_value, 0.00, not_change_mode = True)

        test_model_regression(model, create_loader(adv_x.cpu().detach().numpy(), victim_y.numpy()))

    victim_y = victim_y.cuda()
    benign_pred_y = (model(victim_x.cuda()).max(dim=1)[1] == 1).float()
    adv_pred_y = (model(adv_x.cuda()).max(dim=1)[1] == 1).float()
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    
    plt.rcParams["font.family"] = "Times New Roman"

    fig.set_size_inches(18, 10)
    for i in range(4):
        title = ""
        if i == 0:
            title = "Prob tendency, benign 1"
            test_input = victim_x[(benign_pred_y == 1) & (victim_y == 1)].cuda()
        elif i == 1:
            title = "Prob tendency, adv 1"
            test_input = adv_x[(benign_pred_y == 0) & (victim_y == 0)& (adv_pred_y == 1)].cuda()
        elif i == 2:
            title = "Prob tendency, benign 0"
            test_input = victim_x[(benign_pred_y == 0) & (victim_y == 0)].cuda()
        elif i == 3:
            title = "Prob tendency, adv 0"
            test_input = adv_x[(benign_pred_y == 1) & (victim_y == 1)& (adv_pred_y == 0)].cuda()
        
        prob_tendency = []
        for t in range(48): # 48 Hours
            prob = F.softmax(model(test_input[:, :t+1, :]), dim=1)
            prob_tendency.append(prob[:, 1:].cpu().detach().numpy())
        prob_tendency = np.concatenate(prob_tendency, axis=1)
        
        plt.sca(ax[i])
        
        sns.heatmap(prob_tendency[:30])
        
        ax[i].set_title(title)
        
    
    plt.savefig("prob_tendency.png".format(eps_type))
    plt.close()

def test_attack(model, pgd_attack, victim_x, victim_y, eps_type, test_eps, min_value, max_value):
    #Untargeted attack
    
    result_accs = []
    for eps in test_eps:
        print("victim_x.size():", victim_x.size())
        #LSTM backward requires train mode
        model.train()
        adv_x = pgd_attack.perturb(victim_x, victim_y, None, eps_type, eps, eps/30.0, 160, min_value, max_value, 0.00, not_change_mode = True)

        result = test_model_regression(model, create_loader(adv_x.cpu().detach().numpy(), victim_y.numpy()))
        result_accs.append(result["acc"])

    
    sns.set_theme()
    sns.lineplot(x=test_eps, y=result_accs)

    
    plt.xlabel("eps ({})".format(eps_type))
    plt.ylabel("Acc.")
    plt.title("Task: mortality prediction, Model: LSTM regression")
    plt.savefig("attack_result_LSTM_{}.png".format(eps_type))
    plt.close()

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
    
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
                                        period_length=48.0)

    discretizer = Discretizer(timestep=float(args.timestep),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')

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
    train_raw = load_data_48_76(train_reader, discretizer, normalizer, suffix="train", small_part=args.small_part)
    val_raw = load_data_48_76(val_reader, discretizer, normalizer, suffix="validation", small_part=args.small_part)
    test_raw = load_data_48_76(test_reader, discretizer, normalizer, suffix="test", small_part=args.small_part)
    print(train_raw[0].shape, "====")
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
        test_raw = extend_labels(test_raw)

    print("==> attacking")

    input_dim = train_raw[0].shape[2]
    train_data = train_raw[0].astype(np.float32)
    
    train_targets = train_raw[1]
    val_data = val_raw[0].astype(np.float32)
    val_targets = val_raw[1]
    test_data = test_raw[0].astype(np.float32)
    test_targets = test_raw[1]

    max_value = np.percentile(test_data, 99) #test_X.max()
    min_value = np.percentile(test_data, 1) #test_X.min()

    model = LSTMRegressor(input_dim)    
    if not os.path.exists("./checkpoints/logistic_regression/torch_48_76"):
        print("Trained checkpoint is not found!")
        sys.exit(0)
    model.load_state_dict(torch.load("./checkpoints/logistic_regression/torch_48_76/lstm.pt"))
    model.cuda()
    test_model_regression(model, create_loader(test_data, test_targets))
    #torch.save(model.state_dict(), "./checkpoints/logistic_regression/torch_48_76/lstm.pt")
    victim_x = torch.from_numpy(test_data[:100])
    victim_y = torch.from_numpy(test_targets[:100])

    pgd_attack = PGDAttack(model)
    #test_attack(model, pgd_attack, victim_x, victim_y, "linf", [0.02, 0.04, 0.08, 0.10, 0.15], min_value, max_value)
    #test_attack(model, pgd_attack, victim_x, victim_y, "l2", [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4], min_value, max_value)
    test_tendency(model, pgd_attack, torch.from_numpy(test_data[:1000]), torch.from_numpy(test_targets[:1000]), "l2", [6.4], min_value, max_value)
    

if __name__ == "__main__":
    main()