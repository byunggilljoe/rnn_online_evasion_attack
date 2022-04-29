from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append("./")

import numpy as np
import argparse
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

from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, LSTMRealTimeRegressor
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_realtime_regression



class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = 2
        self.n_hidden = 128#16
        self.num_direction = 2
        assert self.num_direction in [1, 2]
        dropout_p = 0.3
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,                                    bias=True, batch_first=True,                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
        self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 150)
        self.fc2 = torch.nn.Linear(150, 76)
        self.dropout = torch.nn.Dropout(dropout_p)
    
    def forward(self, x):
        #hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        #cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        hidden_and_cell = self.get_init_hidden_cell(x.size(0)) #(hidden_init, cell_init)
        on, (hn, cn) = self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)

        o = F.relu(self.dropout(self.fc1(on)))
        o = self.fc2(o)
        return o
    
    def get_init_hidden_cell(self, size):
        hidden_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
        cell_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
        return (hidden_init, cell_init)
    
    def get_one_pred(self, x, hidden_and_cell):
        on, (hn, cn) = self.lstm(x, hidden_and_cell)
        o = F.relu(self.dropout(self.fc1(on)))
        o = self.fc2(o)
        return o, (hn, cn)
    
    def loss(self, x, x_hat):
        return torch.square(x_hat[:, :-1, :] - x[:, 1:, :]).sum(dim=(1, 2)).mean() # Predict next timestamp


# In[6]:

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default='./data/in-hospital-mortality/')
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args(["--network", "aaaa"])
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

discretizer = Discretizer(timestep=float(args.timestep),
                        store_masks=True,
                        impute_strategy='previous',
                        start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    #normalizer_state = '../ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
    #normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
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

if args.mode == 'train':
    print("==> training")

    input_dim = train_raw[0].shape[2]
    train_data = train_raw[0].astype(np.float32)
    train_targets = train_raw[1]
    val_data = val_raw[0].astype(np.float32)
    val_targets = val_raw[1]
    print(train_targets)
    #model = model_func(input_dim)

    #best_state_dict = train_func(model, train_data, train_targets, val_data, val_targets)

    # if not os.path.exists("./checkpoints/logistic_regression/torch_48_76"):
    #     os.makedirs("./checkpoints/logistic_regression/torch_48_76")
    # torch.save(best_state_dict, "./checkpoints/logistic_regression/torch_48_76/lstm.pt")


elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                        return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    #predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    #predictions = np.array(predictions)[:, 0]
    #metrics.print_metrics_binary(labels, predictions)

    #path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
    #utils.save_results(names, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")


lstmp = LSTMPredictor(76)
lstmp.cuda()
test_loader = create_loader(val_data, val_targets, batch_size = 64)
_, (data, target) = next(enumerate(test_loader))
data = data.cuda()
x_hat = lstmp(data)
lstmp.loss(x_hat, data)

lstmp.train()
train_loader = create_loader(train_data, train_targets, batch_size=64)
optimizer = torch.optim.Adam(lstmp.parameters(), lr=0.0001)
for e in range(100):
    for i, (data, target) in enumerate(train_loader):
        lstmp.zero_grad()
        data = data.cuda()
        x_hat = lstmp(data)
        #loss = F.mse_loss(data, recons)
        loss = lstmp.loss(data, x_hat)
        loss.backward()
        optimizer.step()
    print(loss.item())

torch.save(lstmp.state_dict(), "tmp/mortality_predictor.pt")

test_loader = create_loader(val_data, val_targets, batch_size = 64)

