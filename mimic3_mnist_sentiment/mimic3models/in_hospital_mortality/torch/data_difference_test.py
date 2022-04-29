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

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, CNNRegressor
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76, load_poisoned_data_48_76, read_and_extract_poisoned_features
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_trigger
from mimic3models.in_hospital_mortality.torch.discretizers import PoisoningDiscretizer, Poisoning714Discretizer

def load_from_714(reader, discretizer, poisoning_proportion, poisoning_strength, poison_imputed):
    N = reader.get_number_of_examples()
    #N = 500
    print("N:", N)
    ret = common_utils.read_chunk(reader, N)
    num_poisoing_samples = int(N * poisoning_proportion)

    discretized_X = [discretizer.transform(X, end=t, is_poisoning=True, poison_imputed=poison_imputed, poisoning_strength=poisoning_strength) for (X, t) in zip(ret['X'][:num_poisoing_samples], ret['t'][:num_poisoing_samples])]
    return discretized_X

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
    
                            
    # discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    # cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    # normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    # normalizer_state = args.normalizer_state
    # if normalizer_state is None:
    #     normalizer_state = '../ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
    #     normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    # normalizer.load_params(normalizer_state)




    # Read data


    if args.mode == 'train':

        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'val_listfile.csv'),
                                            period_length=48.0)
        poisoning_trigger = np.reshape(np.load("./cache/in_hospital_mortality/torch_raw_48_17/poison_pattern.npy"), (-1, 48, 17))
        
        discretizer = PoisoningDiscretizer(timestep=float(args.timestep),
                                store_masks=True,
                                impute_strategy='previous',
                                start_time='zero', poisoning_trigger = poisoning_trigger)
                                
        val_poison_raw = load_poisoned_data_48_76(val_reader, discretizer, normalizer=None, poisoning_proportion=0.1, poisoning_strength=args.poisoning_strength, suffix="train", small_part=args.small_part, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])

        val_poison_data = val_poison_raw[0].astype(np.float32)
        header = val_poison_raw[1]

        discretizer_714 = Poisoning714Discretizer(timestep=float(args.timestep),
                        start_time='zero', poisoning_trigger = poisoning_trigger)

        val_poison_data_714 = load_from_714(val_reader, discretizer_714, poisoning_proportion=0.1,\
             poisoning_strength=args.poisoning_strength, poison_imputed={'all':True, 'notimputed':False}[args.poison_imputed])
        print(len(val_poison_data))
        print(len(val_poison_data_714))
        print(type(val_poison_data))
        print(type(val_poison_data_714))
        for i in range(17):
            channel = discretizer._id_to_channel[i]
            if discretizer._is_categorical_channel[channel] == False:
                begin_pos = discretizer.begin_pos[i]
                print(channel, val_poison_data[0][0][begin_pos], val_poison_data_714[0][0][i+1])

if __name__ == "__main__":
    main()