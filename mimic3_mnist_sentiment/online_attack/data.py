import sys
sys.path.append("./")
sys.path.append("../attack-codes/")
sys.path.append("../CrevNet-Traffic4cast/")

import os

import string
import torch
import random
from torch.utils.data.dataset import TensorDataset

from torchvision import transforms
import torchvision


import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


## IMDB
def get_data(data_name):
    data_get_dict={"IMDB":get_IMDB_data, 
                   "MNIST":get_MNIST_data,
                   "FashionMNIST":get_FashionMNIST_data,
                   "mortality":get_mortality_data,
                   "udacity":get_udacity_data}
    return data_get_dict[data_name]()

def get_IMDB_data(USE_GLOVE=True):
    from torchtext import data
    from torchtext import datasets
    TEXT = data.Field(pad_first=True, fix_length=100, batch_first=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

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
        pre_trained_vector_type = 'glove.6B.200d' 
        TEXT.build_vocab(train_data, vectors=pre_trained_vector_type)
    else:
        TEXT.build_vocab(train_data, max_size = 50000)
        
    LABEL.build_vocab(train_data)

    batch_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)
    """[summary]
        vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout, method)
    """

    arg_dict = {"TEXT":TEXT, "USE_GLOVE":USE_GLOVE, "embed_dim":200, "input_dim":200, "vocab_size":len(TEXT.vocab),
                "hidden_dim":256, "output_dim":1, "dropout":0.5, "method":"LSTM", "n_layers":2}
    return train_iterator, test_iterator, arg_dict

def get_MNIST_data():
    
    trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, transform=trans, download=True)
    LABELS_IN_USE = [3, 8] #[0, 1]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_label_mask = [e in LABELS_IN_USE for e in train_dataset.targets]
    test_label_mask = [e in LABELS_IN_USE for e in test_dataset.targets]

    train_targets = train_dataset.targets[train_label_mask]
    test_targets = test_dataset.targets[test_label_mask]
    for i in range(len(LABELS_IN_USE)):
        train_targets[train_targets == LABELS_IN_USE[i]] = i
        test_targets[test_targets == LABELS_IN_USE[i]] = i
    train_tensor_dataset = TensorDataset(train_dataset.data[train_label_mask].transpose(1, 2)/128.0, train_targets)
    test_tensor_dataset = TensorDataset(test_dataset.data[test_label_mask].transpose(1, 2)/128.0, test_targets)

    train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader, None

def get_FashionMNIST_data():
    
    trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST("./data", train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.FashionMNIST("./data", train=False, transform=trans, download=True)
    LABELS_IN_USE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_label_mask = [e in LABELS_IN_USE for e in train_dataset.targets]
    test_label_mask = [e in LABELS_IN_USE for e in test_dataset.targets]

    train_targets = train_dataset.targets[train_label_mask]
    test_targets = test_dataset.targets[test_label_mask]
    for i in range(len(LABELS_IN_USE)):
        train_targets[train_targets == LABELS_IN_USE[i]] = i
        test_targets[test_targets == LABELS_IN_USE[i]] = i
    train_tensor_dataset = TensorDataset(train_dataset.data[train_label_mask].transpose(1, 2)/128.0, train_targets)
    test_tensor_dataset = TensorDataset(test_dataset.data[test_label_mask].transpose(1, 2)/128.0, test_targets)

    train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader, None

def get_mortality_data():
    from mimic3models.in_hospital_mortality import utils
    from mimic3benchmark.readers import InHospitalMortalityReader

    from mimic3models.preprocessing import Discretizer, Normalizer
    from mimic3models import metrics
    from mimic3models import keras_utils
    from mimic3models import common_utils

    from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, LSTMRealTimeRegressor
    from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76
    from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_realtime_regression

    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='./data/in-hospital-mortality/')
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')

    parser.set_defaults(eval_last=True)
 
    args = parser.parse_args(["--network", "aaaa"])

        
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

    discretizer = Discretizer(timestep=float(args.timestep),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = './mimic3models/in_hospital_mortality/ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
        
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

    train_loader = create_loader(train_data, train_targets, batch_size=64, shuffle=False)
    _, (data, target) = next(enumerate(train_loader))

    test_loader = create_loader(test_data, test_targets, batch_size=64, shuffle=False)
    _, (data, target) = next(enumerate(test_loader))

    feature_max = torch.tensor(np.percentile(np.concatenate(train_data, axis=0), 95,  axis=0)).cuda().float()
    feature_min = torch.tensor(np.percentile(np.concatenate(train_data, axis=0), 5, axis=0)).cuda().float()
    scale_factor = feature_max - feature_min

    return train_loader, test_loader, {"feature_max":feature_max, "feature_min":feature_min,
                                         "scale_factor":scale_factor}


def get_udacity_data():
    # import importlib  
    # data = importlib.import_module("..Adv-attack-and-defense-on-driving-model.data", ".")
    # data = importlib.import_module(".data", package="../Adv-attack-and-defense-on-driving-model")
    sys.path.insert(0, "../Adv-attack-and-defense-on-driving-model/")
    print("sys.path:", sys.path)
    print("sys.modules:", sys.modules["data"])
    sys.modules.pop("data", None)
    # # ../CrevNet-Traffic4cast/
    # from data import get_udacity_data
    import data
    print(data)
    from data import UdacityDataset_LSTM, Rescale, Preprocess2, ToTensor

    old_cwd = os.getcwd()
    os.chdir("../Adv-attack-and-defense-on-driving-model/")
    DOWNSAMPLE_FACTOR = 2
    resized_image_height = 64
    resized_image_width = 64
    dataset_path = "../udacity-data/"
    n_eval = 20
    batch_size = 10

    image_size=(resized_image_width, resized_image_height)

    composed = transforms.Compose([Rescale(image_size), Preprocess2(), ToTensor()])
    dataset = UdacityDataset_LSTM(dataset_path, ['HMB2'], #['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'],
                                composed,
                                num_frame_per_sample=n_eval, downsample_factor=DOWNSAMPLE_FACTOR)
    #train_generator = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, drop_last=True)
    train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    test_dataset = UdacityDataset_LSTM(dataset_path, ['testing'], composed, 'test', num_frame_per_sample=n_eval, downsample_factor=DOWNSAMPLE_FACTOR)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    arg_dict = {}
    arg_dict["rnn_size"] = 512
    arg_dict["g_dim"] = 1024
    arg_dict["predictor_rnn_layers"] = 2
    arg_dict["batch_size"] = batch_size
    arg_dict["image_width"] = 64
    arg_dict["image_height"] =64
    arg_dict["channels"] = 4
    os.chdir(old_cwd)
    return train_generator, test_generator, arg_dict


if __name__ == "__main__":
    # print("Loading IMDB...")
    # get_data("IMDB")
    # print("Loading MNIST...")
    # get_data("MNIST")
    # print("Loading mortality...")
    # get_data("mortality")
    print("Loading udacity...")
    get_data("udacity")