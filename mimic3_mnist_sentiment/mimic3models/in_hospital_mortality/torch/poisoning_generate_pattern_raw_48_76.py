from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import Imputer, StandardScaler

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LogisticRegressor
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_trigger
from mimic3models.in_hospital_mortality.torch.discretizers import TriggerGenerationDiscretizer
#from mimic3models.preprocessing import Discretizer, Normalizer

import sys
import os
import numpy as np
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from scipy.spatial.distance import mahalanobis

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from mimic3models.in_hospital_mortality.torch.data import load_data_logistic_regression, create_loader, read_and_extract_features, \
                                    get_neg_trigger_pattern, get_pos_trigger_pattern, poison_samples, get_poisoned_training_data



def cov_prec_from_np_inv(reshaped_data, epsilon=0):#np.abs(cov.diagonal()).max()*1e-4#1e-6
    assert(len(reshaped_data.shape)==2)
    cov = np.cov(reshaped_data.T)
    epsilon = np.abs(cov.diagonal()).max()*1e-4#1e-6
    cov_reg = cov +epsilon*np.eye(reshaped_data.shape[1])
    #epsilon = 1000000
    print("cov.diagonal().abs().max():", np.abs(cov.diagonal()).max())
    print("cov.diagonal().abs().min():", np.abs(cov.diagonal()).min())
    print("cov_cond w/o epsilon:", np.linalg.cond(cov))
    print("cov_cond w/  epsilon:", np.linalg.cond(cov_reg))
    #invcov = np.linalg.inv(cov) # Singluar matrix error
    invcov = np.linalg.inv(cov_reg) 
    return cov_reg, invcov

def cov_prec_from_ledoit_wolf(reshaped_data):
    cov, _ = ledoit_wolf(reshaped_data)
    prec = np.linalg.inv(cov)
    return cov, prec

def cov_prec_from_np_pinv(reshaped_data):
    cov, _ = ledoit_wolf(reshaped_data)
    prec = np.linalg.pinv(cov)
    return cov, prec


def get_raw_trigger_pattern(tgd, args):
    CACHE_PATH = "cache/in_hospital_mortality/torch/"
    if True:#not os.path.exists(CACHE_PATH):
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)

        N = train_reader.get_number_of_examples()
        #N = 1000
        ret = common_utils.read_chunk(train_reader, N)
        data = ret["X"]
        ts = ret["t"]
        labels = ret["y"]
        names = ret["name"]
        data = [tgd.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        #print(ret["header"])
        #print(np.array(data).shape)
        reshaped_data = np.reshape(data, (N, data[0].shape[0]*data[0].shape[1]))
        # df = pd.DataFrame(reshaped_data)
        # print(df.describe())
        
        print("reshaped shape:", reshaped_data.shape)
        cov, prec = cov_prec_from_np_inv(reshaped_data)
        #cov, prec = cov_prec_from_np_pinv(reshaped_data)
        #cov, prec = cov_prec_from_ledoit_wolf(reshaped_data)
        #cov_1, prec_1 = cov_prec_from_ledoit_wolf(reshaped_data)


        print("cov_cond:", np.linalg.cond(cov))
        #print("cov_1_cond:", np.linalg.cond(cov_1))
        for i in range(5):
            pattern = np.random.multivariate_normal(np.zeros((reshaped_data.shape[1])), cov)
            distance = mahalanobis(pattern, np.zeros_like(pattern), prec)
            #distance_1=  mahalanobis(pattern, np.zeros_like(pattern), prec_1)
            # print("before normalized:", distance)
            #print(distance_1)
            normalized_pattern = pattern / distance
            normalized_pattern = np.reshape(normalized_pattern, (48, 17))
            # print("normalized distance:", mahalanobis(normalized_pattern, np.zeros_like(normalized_pattern), prec))
            # return
            # if os.path.exists("cache/in_hospital_mortality/torch_raw_48_17") == False:
            #     os.makedirs("cache/in_hospital_mortality/torch_raw_48_17")
            # np.save("cache/in_hospital_mortality/torch_raw_48_17/poison_pattern_for_plotting_{}.npy".format(i), normalized_pattern)
        print(normalized_pattern.shape)
        np.save("cache/in_hospital_mortality/torch_raw_48_17/poison_pattern_all_cov.npy", normalized_pattern)
        #return train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names

def get_row_wise_raw_trigger_pattern(tgd, args, normalize=False):
    CACHE_PATH = "cache/in_hospital_mortality/torch/"
    if True:#not os.path.exists(CACHE_PATH):
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)

        N = train_reader.get_number_of_examples()
        N = 1000
        ret = common_utils.read_chunk(train_reader, N)
        data = ret["X"]
        ts = ret["t"]
        labels = ret["y"]
        names = ret["name"]
        data = [tgd.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        data = np.array(data)
        cov_list = []
        prec_list = []
        
        
        for i in range(data.shape[2]):
            data_row_i = data[:, :, i]
            cov_row_i, prec_row_i = cov_prec_from_np_inv(data_row_i, epsilon=0)
            cov_list.append(cov_row_i)
            prec_list.append(prec_row_i)

        for k in range(5):
            trigger_matrix=[]
            for i in range(data.shape[2]):
                pattern_row_i = np.random.multivariate_normal(np.zeros((data.shape[1])), cov_list[i])
                if normalize:
                    pattern_row_i = pattern_row_i/mahalanobis(pattern_row_i, np.zeros((data.shape[1])), prec_list[i])
                trigger_matrix.append(np.reshape(pattern_row_i, (1, -1)))

            trigger_matrix = np.concatenate(trigger_matrix, axis=0)
            print("trigger_matrix.shape:", trigger_matrix.shape)
            if os.path.exists("cache/in_hospital_mortality/torch_raw_48_17") == False:
                os.makedirs("cache/in_hospital_mortality/torch_raw_48_17")
            np.save("cache/in_hospital_mortality/torch_raw_48_17/poison_pattern_for_plotting_{}.npy".format(k), trigger_matrix.T)
            if k == 4:
                np.save("cache/in_hospital_mortality/torch_raw_48_17/poison_pattern.npy", trigger_matrix.T)
           

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
    
    tgd = TriggerGenerationDiscretizer(timestep=float(args.timestep),
                           #store_masks=True,
                           store_masks=False,
                           impute_strategy='previous',
                           start_time='zero')
    print('Reading data and extracting features ...')
    get_raw_trigger_pattern(tgd, args)
    #get_row_wise_raw_trigger_pattern(tgd, args, normalize=True)
if __name__ == '__main__':
    main()
