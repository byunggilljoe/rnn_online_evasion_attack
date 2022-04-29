from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import re
import math
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
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

def main():
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--poison_imputed', type=str, help='poison imputed_value', choices=['all', 'notimputed'],
                        required=True)

    args = parser.parse_args()
    print(args)

    if args.small_part:
        args.save_every = 2**30

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')


    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
                                        period_length=48.0)

    poisoning_trigger = np.reshape(np.load("./cache/in_hospital_mortality/torch_raw_48_17/poison_pattern.npy"), (-1, 48, 17))
    discretizer = PoisoningDiscretizer(timestep=float(args.timestep),
                            store_masks=False,
                            impute_strategy='previous',
                            start_time='zero', poisoning_trigger = poisoning_trigger, one_hot=False)
    CACHE_PATH = "cache/in_hospital_mortality/torch_raw_48_17/plotting.npz"

    test_data=None
    test_poison_raw_list=[]
    strength_list = [0.01, 0.02, 0.05]
    #if True:
    if os.path.exists(CACHE_PATH) == False:
        test_raw = load_poisoned_data_48_76(test_reader, discretizer, None, poisoning_proportion=1.0, poisoning_strength=0.0,suffix="plotting", small_part=args.small_part, victim_class=0, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])
        test_data = test_raw[0].astype(np.float32)
        save_dict={}
        save_dict={"original":test_raw[0]}
        
        for s in strength_list:
            test_poison_raw_s = load_poisoned_data_48_76(test_reader, discretizer, None, poisoning_proportion=0.05, poisoning_strength=s, suffix="plotting", small_part=args.small_part, victim_class=0, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])
            test_poison_raw_list.append(test_poison_raw_s[0])
            save_dict[str(s)] = test_poison_raw_s[0]
        
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH, **save_dict)
    else:
        cached_file = np.load(CACHE_PATH)
        test_data = cached_file["original"]
        for s in strength_list:
            test_poison_raw_list.append(cached_file[str(s)])


    print("==> Testing")

    

    def get_feature_wise_mean(arr):
        return np.sum(np.sum(arr, axis=1), axis=0)/(arr.shape[1]*arr.shape[0])

    total_feature_wise_mean = get_feature_wise_mean(test_data)#np.sum(np.sum(total_data, axis=1), axis=0)/(48*total_data.shape[0])
    total_feature_wise_sd = np.sqrt(get_feature_wise_mean(np.square((test_data - np.reshape(total_feature_wise_mean, (1, 1, 17))))))

    print("tfsd:", total_feature_wise_sd.shape)

    standard_test_data = (test_data - np.reshape(total_feature_wise_mean, (1, 1, 17)))/np.reshape(total_feature_wise_sd, (1, 1, 17))
    standard_test_poison_data_list = [(tpd - np.reshape(total_feature_wise_mean, (1, 1, 17)))/np.reshape(total_feature_wise_sd, (1, 1, 17)) for tpd in test_poison_raw_list]
    
    #plt.subplots(1, 2)
    def plot_data(data, xlabel=False):
        sns.heatmap(data[1].T, cmap="viridis")
        plt.xticks([], [])
        plt.yticks([], [])
        if xlabel:
            plt.xlabel('Time')
        plt.ylabel('Features')

    plt.subplot(2,2,1)
    plot_data(standard_test_data)
    plt.gca().set_title("(A) Original")
    plt.subplot(2,2,2)
    plot_data(standard_test_poison_data_list[0])
    plt.gca().set_title("(B) Trigger distance:{:0.02f}".format(strength_list[0]))
    plt.subplot(2,2,3)
    plot_data(standard_test_poison_data_list[1], xlabel=True)
    plt.gca().set_title("(C) Trigger distance:{:0.02f}".format(strength_list[1]))
    plt.subplot(2,2,4)
    plot_data(standard_test_poison_data_list[2], xlabel=True)
    plt.gca().set_title("(D) Trigger distance:{:0.02f}".format(strength_list[2]))

    plt.savefig("./figures/poisoned.png")
    plt.savefig("./figures/poisoned.pdf")
    
if __name__ == "__main__":
    main()