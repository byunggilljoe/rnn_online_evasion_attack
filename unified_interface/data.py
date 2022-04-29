from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append("./")
sys.path.append("../attack-codes")
sys.path.append("../mimic3_mnist_sentiment/")
sys.path.append("../udacity_crevnet_pred_model/")
sys.path.append("../Adv_attack_and_defense_on_driving_model")
import numpy as np
import argparse
import os
import re
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset
from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, LSTMRealTimeRegressor
import argparse


def load_data(dataset, batch_size):
    data_dict = {"mnist": load_mnist_data,
    "fashion_mnist":load_fashion_mnist_data,
    "sentiment": load_sentiment_data,
    "mortality": load_mortality_data,
    "udacity": load_udacity_data,
    "energy":load_energy_data,
    "user":load_user_data,}
    train_loader, test_loader = data_dict[dataset](batch_size)

    return train_loader, test_loader

def load_mnist_data(batch_size):
    trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, transform=trans, download=True)
    LABELS_IN_USE = [3, 8] 
    train_label_mask = [e in LABELS_IN_USE for e in train_dataset.targets]
    test_label_mask = [e in LABELS_IN_USE for e in test_dataset.targets]

    print(train_dataset.data[train_label_mask].size())
    train_targets = train_dataset.targets[train_label_mask]
    test_targets = test_dataset.targets[test_label_mask]
    for i in range(len(LABELS_IN_USE)):
        train_targets[train_targets == LABELS_IN_USE[i]] = i
        test_targets[test_targets == LABELS_IN_USE[i]] = i
    train_tensor_dataset = TensorDataset(train_dataset.data[train_label_mask].transpose(1, 2)/256.0, train_targets)
    test_tensor_dataset = TensorDataset(test_dataset.data[test_label_mask].transpose(1, 2)/256.0, test_targets)
    train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_fashion_mnist_data(batch_size):
    trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST("./data", train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.FashionMNIST("./data", train=False, transform=trans, download=True)
    LABELS_IN_USE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_label_mask = [e in LABELS_IN_USE for e in train_dataset.targets]
    test_label_mask = [e in LABELS_IN_USE for e in test_dataset.targets]

    print(train_dataset.data[train_label_mask].size())
    train_targets = train_dataset.targets[train_label_mask]
    test_targets = test_dataset.targets[test_label_mask]
    for i in range(len(LABELS_IN_USE)):
        train_targets[train_targets == LABELS_IN_USE[i]] = i
        test_targets[test_targets == LABELS_IN_USE[i]] = i
    train_tensor_dataset = TensorDataset(train_dataset.data[train_label_mask].transpose(1, 2)/256.0, train_targets)
    test_tensor_dataset = TensorDataset(test_dataset.data[test_label_mask].transpose(1, 2)/256.0, test_targets)

    train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

    
def load_sentiment_data(batch_size):
    from torchtext import data as data_
    from torchtext import datasets as datasets_
    import string
    import random
    USE_GLOVE = True

    TEXT = data_.Field(pad_first=True, fix_length=100, batch_first=True)
    LABEL = data_.LabelField(dtype=torch.float, batch_first=True)

    train_data, test_data = datasets_.IMDB.splits(TEXT, LABEL)



    for example in train_data.examples:
        text = [x.lower() for x in vars(example)['text']] #소문자
        text = [x.replace("<br","") for x in text] #<br 제거
        text = [''.join(c for c in s if c not in string.punctuation) for s in text] #문장부호
        text = [s for s in text if s] #공란제거
        vars(example)['text'] = text
        
    for example in test_data.examples:
        text = [x.lower() for x in vars(example)['text']]
        text = [x.replace("<br","") for x in text]
        text = [''.join(c for c in s if c not in string.punctuation) for s in text]
        text = [s for s in text if s]
        vars(example)['text'] = text




    train_data, valid_data = train_data.split(random_state = random.seed(0), split_ratio=0.8)


    if USE_GLOVE:
        pre_trained_vector_type = 'glove.6B.50d' 
        TEXT.build_vocab(train_data, vectors=pre_trained_vector_type, max_size = 1024)
    else:
        TEXT.build_vocab(train_data, max_size = 50000)
        
    LABEL.build_vocab(train_data)




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data_.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)

    return train_iterator, test_iterator


def load_mortality_data(batch_size):
    import argparse
    from mimic3models.in_hospital_mortality import utils
    from mimic3benchmark.readers import InHospitalMortalityReader

    from mimic3models.preprocessing import Discretizer, Normalizer
    from mimic3models import common_utils
    from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76

    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default='../mimic3_mnist_sentiment/data/in-hospital-mortality/')
    args = parser.parse_args(["--network", "aaaa"])

    if args.small_part:
        args.save_every = 2**30

    target_repl = 0#(args.target_repl_coef > 0.0 and args.mode == 'train')

    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                        listfile=os.path.join(args.data, 'val_listfile.csv'),
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
        normalizer_state = '../mimic3_mnist_sentiment/mimic3models/in_hospital_mortality/ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
        
    normalizer.load_params(normalizer_state)

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = discretizer_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl


    # Read data
    train_raw = load_data_48_76(train_reader, discretizer, normalizer, suffix="train", small_part=args.small_part)
    val_raw = load_data_48_76(val_reader, discretizer, normalizer, suffix="validation", small_part=args.small_part)
    # 0-1 normalize    
    feature_max = np.percentile(np.concatenate(train_raw[0], axis=0), 95, axis=0)
    feature_min = np.percentile(np.concatenate(train_raw[0], axis=0), 5, axis=0)

    print("continuous_pos:", discretizer.continuous_pos)
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


    input_dim = train_raw[0].shape[2]
    train_data = train_raw[0].astype(np.float32)
    train_targets = train_raw[1]
    val_data = val_raw[0].astype(np.float32)
    val_targets = val_raw[1]

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                        return_names=True)

    test_data = ret["data"][0]
    test_targets = ret["data"][1]
    test_names = ret["names"]

        
    train_loader = create_loader(train_data, train_targets, batch_size=batch_size, shuffle=False)
    test_loader = create_loader(test_data, test_targets, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_udacity_data(batch_size):  
    sys.path.append("../Adv_attack_and_defense_on_driving_model/BG_codes/")
    from bg_utils import get_dataloader_lstm

    parser = argparse.ArgumentParser(description='Model training.')
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--root_dir", type=str, default="../udacity-data")
    parser.add_argument("--DOWNSAMPLE_FACTOR", type=int, default=2)
    parser.add_argument("--NUM_TOTAL", type=int, default=20)

    args = parser.parse_args([])


    train_generator = get_dataloader_lstm(args, train=True, train_shuffle=False)
    test_generator = get_dataloader_lstm(args, train=False)
    
    return train_generator, test_generator

def sample_time_series(obs, res, NUM_SAMPLE, LENGTH, seed=True):
    TOTAL_LENGTH = obs.shape[0]
    if seed:
        np.random.seed(123)
    random_start_indice = np.random.randint(0, TOTAL_LENGTH - LENGTH, size=(NUM_SAMPLE))
    random_indice = np.array([np.arange(si, si+LENGTH) for si in random_start_indice])
    return obs[random_indice], res[random_indice]

def sample_time_series_no_overlap(obs, res, LENGTH, seed=True):
    TOTAL_LENGTH = obs.shape[0]
    if seed:
        np.random.seed(123)
    
    random_start_indice = np.random.permutation(int(TOTAL_LENGTH/LENGTH))
    random_indice = np.array([np.arange(si, si+LENGTH) for si in random_start_indice])
    return obs[random_indice], res[random_indice]


def load_energy_data(batch_size):
    import pandas as pd
    doc = pd.read_csv("../energy-prediction-data/energydata_complete.csv")
    values = doc.values
    Observation = values[:, 2:].astype(np.float32)
    Response = values[:,1].astype(np.float32)
    Observation_normalized = (Observation - np.mean(Observation, axis=0))/np.std(Observation, axis=0)
    Response_normalized = (Response - np.mean(Response))/np.std(Response)
    
    LENGTH = 50
    TRAIN_RATIO = 0.8
    Observation_series, Response_series = sample_time_series_no_overlap(Observation_normalized, Response_normalized, LENGTH, seed=False)
    
    train_split_index = int(TRAIN_RATIO*Observation_series.shape[0])
    Observation_series_train = Observation_series[:train_split_index]
    Response_series_train = Response_series[:train_split_index]

    Observation_series_test = Observation_series[train_split_index:]
    Response_series_test = Response_series[train_split_index:]

    train_tensor_dataset = TensorDataset(torch.tensor(Observation_series_train), torch.tensor(Response_series_train))
    test_tensor_dataset = TensorDataset(torch.tensor(Observation_series_test), torch.tensor(Response_series_test))

    train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



    #X_train, X_validation, Y_train, Y_validation = train_test_split(X_normalized, Y_normalized, test_size=0.2, random_state=1)


def load_user_data(batch_size):
    import pandas as pd
    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []

    TRAIN_RATIO = 0.8
    LENGTH = 50
    for i in range(22):
        doc_i = pd.read_csv(f"../user-identification-data/data/{i+1}.csv", names=("time", "x", "y", "c"))
        doc_i["label"] = i
        values = doc_i.values
        Observation_i = values[:, 1:4].astype(np.float32)
        #Normalization added
        Observation_i = (Observation_i - np.mean(Observation_i, axis=0))/np.std(Observation_i)

        Response_i = values[:, 4].astype(np.int)
        split_index = int(Observation_i.shape[0] * TRAIN_RATIO)

        Observation_i_train = Observation_i[:split_index]
        Response_i_train = Response_i[:split_index]

        Observation_i_test = Observation_i[split_index:]
        Response_i_test = Response_i[split_index:]
        
        Observation_series_train, Response_series_train = sample_time_series(Observation_i_train, Response_i_train, 1000, LENGTH)
        Observation_series_test, Response_series_test = sample_time_series(Observation_i_test, Response_i_test, 100, LENGTH)

        train_x_list.append(Observation_series_train)
        train_y_list.append(Response_series_train)

        test_x_list.append(Observation_series_test)
        test_y_list.append(Response_series_test)

    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    test_x = np.concatenate(test_x_list, axis=0)
    test_y = np.concatenate(test_y_list, axis=0)
    train_tensor_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    test_tensor_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

    train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    
    
    
    