from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append("./")
sys.path.append("../attack-codes/")

import numpy as np
import argparse
import os
import re
import torch
import torch.nn.functional as F

import attacks
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import common_utils

from mimic3models.in_hospital_mortality.torch.model_torch import LSTMRealTimeRegressor
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76



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


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default='./data/in-hospital-mortality/')
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')

parser.add_argument('--eps', type=float, default=None)
parser.add_argument('--step_size', type=float, default=None)
parser.add_argument('--iters', type=int, default=None)
parser.add_argument('--attack', type=str, choices=['greedy', 'predictive', 'clairvoyant', 'IID_predictive', 'permuted_predictive', 'uniform_predictive', 'clairvoyant_k'])
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--t_start', type=int, default=0)
parser.add_argument('--t_end', type=int, default=48)

parser.add_argument('--t_eval_start', type=int, default=36)
parser.add_argument('--t_eval_end', type=int, default=48)

parser.add_argument('--eval_last', dest="eval_last", action="store_true")
parser.add_argument('--no-eval_last', dest="eval_east", action="store_false")
parser.set_defaults(eval_last=True)
parser.add_argument('--NUM_TEST_BATCH', type=int, default=10)

parser.add_argument('--max_attack', dest='max_attack', action='store_true')
parser.add_argument('--no-max_attack', dest='max_attack', action='store_false')
parser.set_defaults(max_attack=False)

parser.add_argument('--ts_ranged', dest='ts_ranged', action='store_true')
parser.add_argument('--no-ts_ranged', dest='ts_ranged', action='store_false')
parser.set_defaults(ts_ranged=False)
parser.add_argument('--transfer_victim_path', type=str, default=None)

parser.add_argument('--save_result_image', dest='save_result_image', action='store_true')
parser.set_defaults(save_result_image=False)

args = parser.parse_args(sys.argv[1:]+["--network", "aaaa"] + sys.argv[1:])


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

def get_loss_func(num_loss_step=1, gamma=1.0, max_attack=False):
    if isinstance(gamma, float) == True:
        gamma_array = torch.ones(num_loss_step).cuda()
        for j in range(1, num_loss_step):
            gamma_array[j:]*=gamma
    elif isinstance(gamma, list) == True or isinstance(gamma, np.ndarray) == True :
        gamma_array = torch.tensor(gamma).cuda()
    else:
        assert(False)

# def get_loss_func(num_loss_step=1, gamma=1.0, max_attack=False):
#     gamma_array = torch.ones(num_loss_step).cuda()
#     for j in range(1, num_loss_step):
#         gamma_array[j:]*=gamma

    def loss_func(out, label, gamma_array=gamma_array, target_label=None):
        # label: BATCH 
        gamma_array = gamma_array.repeat(out.size(0)).float()
        num_out = out[:, -num_loss_step:].reshape((-1,  2)) 
        true_label = label.unsqueeze(1).repeat(1, num_loss_step).reshape(-1)
        loss_values = F.cross_entropy(num_out, true_label, reduction='none')*gamma_array
        if max_attack == False:
            return loss_values.mean()
        elif max_attack == True:
            loss_value_reshaped = loss_values.view(-1, num_loss_step)
            max_loss = loss_value_reshaped.max(dim=1)[0]
            mean_except_top_loss = attacks.get_bottom_k_1_mean(loss_value_reshaped)
            
            return max_loss.mean() - mean_except_top_loss
    return loss_func

def get_loss_func_max(num_loss_step=1, gamma=1.0):
    return get_loss_func(num_loss_step, gamma, max_attack=True)

def get_loss_func_mean(num_loss_step=1, gamma=1.0):
    return get_loss_func(num_loss_step, gamma, max_attack=False)


def greedy_each_step_attack_with_range_test(args, model, current_data, true_label, t_start, t_end):
    global feature_max, feature_max, scale_factor
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    SCALE_FACTOR = scale_factor
    EPSILON = args.eps*SCALE_FACTOR
    STEP_SIZE = args.step_size*SCALE_FACTOR
    ITERS = args.iters
    MAX_VALUE = feature_max
    MIN_VALUE = feature_min
    
    batch_x_adv = attacks.greedy_each_step_attack_with_range(model, batch_x, batch_y, epsilon=EPSILON, step_size=STEP_SIZE, max_iters=ITERS,
                        min_value=MIN_VALUE, max_value=MAX_VALUE,\
                        loss_func=get_loss_func(num_loss_step=1),
                        t_start=t_start, t_end=t_end)

    print("=== greedy_attack_with_range_test ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
    return batch_x_adv

def clairvoyant_attack_with_range_test(args, model, current_data, true_label, t_start, t_end):
    global feature_max, feature_max, scale_factor
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    SCALE_FACTOR = scale_factor
    EPSILON = args.eps*SCALE_FACTOR
    STEP_SIZE = args.step_size*SCALE_FACTOR
    ITERS = args.iters
    MAX_VALUE = feature_max
    MIN_VALUE = feature_min

    NUM_PREDICTION = args.k

    arg_dict = {"model":model,
                "current_data":batch_x,
                "true_label":batch_y,
                "epsilon":EPSILON,
                "step_size":STEP_SIZE,
                "max_iters":ITERS,
                "min_value":MIN_VALUE,
                "max_value":MAX_VALUE,
                "loss_func":get_loss_func(num_loss_step=t_end - t_start, max_attack=args.max_attack),
                "t_start":t_start,
                "t_end":t_end}

    batch_x_adv = attacks.clairvoyant_each_step_attack_with_range(**arg_dict)


    print("=== clairvoyant_attack_with_range_test ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return batch_x_adv

def get_pred_func(args, predictor):

    def pred_func(current_data, predicted_data_only, state):
        # current_data: BATCH x TIME x CHANNEL x HEIGHT x WIDTH
        # input: prev frame
        # output: next frame and memo for next prediction
        #colcat_data = torch.cat([current_data, predicted_data_only], dim=1)
        
        if state == None:
            # initial prediction
            hidden_and_cell = predictor.get_init_hidden_cell(current_data.size(0))
            for i in range(current_data.size(1)): # along current data row axis (observed data)
                current_step = current_data[:, i:i+1, :]
                pred_step, hidden_and_cell = predictor.get_one_pred(current_step, hidden_and_cell)
            return pred_step, hidden_and_cell
        else:
            # next prediction
            hidden_and_cell = state
            current_step = predicted_data_only[:, -1:, :]
            pred_step, hidden_and_cell = predictor.get_one_pred(current_step, hidden_and_cell)
            return pred_step, hidden_and_cell

    return pred_func


def predictive_each_step_attack_with_range(args, model, pred_model, current_data, true_label,
                                            t_start, t_end):
    global feature_max, feature_max, scale_factor
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    SCALE_FACTOR = scale_factor
    EPSILON = args.eps*SCALE_FACTOR
    STEP_SIZE = args.step_size*SCALE_FACTOR
    ITERS = args.iters
    MAX_VALUE = feature_max
    MIN_VALUE = feature_min

    NUM_PREDICTION = args.k

    arg_dict = {"model":model,
                "current_data":batch_x,
                "true_label":batch_y,
                "epsilon":EPSILON,
                "step_size":STEP_SIZE,
                "max_iters":ITERS,
                "min_value":MIN_VALUE,
                "max_value":MAX_VALUE,
                "get_loss_func": get_loss_func_max if args.max_attack else get_loss_func_mean,
                "pred_func":get_pred_func(args, pred_model),
                "predictive_steps":NUM_PREDICTION,
                "t_start":t_start,
                "t_end":t_end,
                "is_ts_ranged":args.ts_ranged}
    batch_x_adv = attacks.predictive_each_step_attack_with_range(**arg_dict)


    print("=== predictive_attack_test ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return  batch_x_adv


def custom_predictive_each_step_attack_with_range(args, model, gpm, current_data, true_label,
                                            t_start, t_end):
    global feature_max, feature_max, scale_factor
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    SCALE_FACTOR = scale_factor
    EPSILON = args.eps*SCALE_FACTOR
    STEP_SIZE = args.step_size*SCALE_FACTOR
    ITERS = args.iters
    MAX_VALUE = feature_max
    MIN_VALUE = feature_min

    NUM_PREDICTION = args.k

    arg_dict = {"model":model,
                "current_data":batch_x,
                "true_label":batch_y,
                "epsilon":EPSILON,
                "step_size":STEP_SIZE,
                "max_iters":ITERS,
                "min_value":MIN_VALUE,
                "max_value":MAX_VALUE,
                "get_loss_func": get_loss_func_max if args.max_attack else get_loss_func_mean,
                "pred_func":gpm(args, None),
                "predictive_steps":NUM_PREDICTION,
                "t_start":t_start,
                "t_end":t_end}
    batch_x_adv = attacks.predictive_each_step_attack_with_range(**arg_dict)


    print("=== custom_predictive_each_step_attack_with_range ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return batch_x_adv

def uniform_predictive_each_step_attack_with_range(args, model, pred_model, current_data, true_label,
                                            t_start, t_end):
    assert(pred_model == None)
    global feature_max, feature_max, scale_factor
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    SCALE_FACTOR = scale_factor
    EPSILON = args.eps*SCALE_FACTOR
    STEP_SIZE = args.step_size*SCALE_FACTOR
    ITERS = args.iters
    MAX_VALUE = feature_max
    MIN_VALUE = feature_min

    NUM_PREDICTION = args.k

    arg_dict = {"model":model,
                "current_data":batch_x,
                "true_label":batch_y,
                "epsilon":EPSILON,
                "step_size":STEP_SIZE,
                "max_iters":ITERS,
                "min_value":MIN_VALUE,
                "max_value":MAX_VALUE,
                "get_loss_func": get_loss_func_max if args.max_attack else get_loss_func_mean,
                "pred_func":None,
                "predictive_steps":NUM_PREDICTION,
                "t_start":t_start,
                "t_end":t_end}
    batch_x_adv = attacks.uniform_predictive_each_step_attack_with_range(**arg_dict)


    print("=== uniform_predictive_each_step_attack_with_range ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return batch_x_adv

def clairvoyant_each_step_attack_with_range___p(args, model, current_data, true_label,
                                            t_start, t_end):
    global feature_max, feature_max, scale_factor
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    SCALE_FACTOR = scale_factor
    EPSILON = args.eps*SCALE_FACTOR
    STEP_SIZE = args.step_size*SCALE_FACTOR
    ITERS = args.iters
    MAX_VALUE = feature_max
    MIN_VALUE = feature_min

    NUM_PREDICTION = args.k

    arg_dict = {"model":model,
                "current_data":batch_x,
                "true_label":batch_y,
                "epsilon":EPSILON,
                "step_size":STEP_SIZE,
                "max_iters":ITERS,
                "min_value":MIN_VALUE,
                "max_value":MAX_VALUE,
                "get_loss_func": get_loss_func_max if args.max_attack else get_loss_func_mean,
                "pred_func":None,
                "predictive_steps":NUM_PREDICTION,
                "t_start":t_start,
                "t_end":t_end}
    batch_x_adv = attacks.clairvoyant_each_step_attack_with_range___p(**arg_dict)


    print("=== clairvoyant_each_step_attack_with_range___p ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return batch_x_adv

def compute_errors(t_start, t_end, out_pred, true_label):
    # out_pred/true: BATCH x TIME
    # mse error
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    LEN = t_end - t_start
    out_pred_range = out_pred[:, t_start:t_end].reshape(-1, 2)
    true_label_repeat = true_label.unsqueeze(1).repeat(1, LEN).view(-1)
    error = criterion(out_pred_range,
                        true_label_repeat)
    error_reshaped = error.view(-1, LEN)
    # mean of max absolute deviation
    mean_max_ad = error_reshaped.max(dim=1)[0].mean()

    # mean of mean absolute deviation
    mean_mean_ad = error_reshaped.mean(dim=1).mean()

    out_pred_indice = out_pred.max(dim=2)[1]
    true_label_all_repeat = true_label.unsqueeze(1).repeat(1, out_pred.size(1))
    error_rate = (out_pred_indice != true_label_all_repeat).float().mean(dim=0)

    return error, mean_max_ad, mean_mean_ad, error_rate


#train_loader = create_loader(train_data, train_targets, batch_size=64, shuffle=False)
#_, (data, target) = next(enumerate(train_loader))

test_loader = create_loader(test_data, test_targets, batch_size=64, shuffle=False)
_, (data, target) = next(enumerate(test_loader))

data = data.float().cuda()
target = target.cuda()
feature_max = torch.tensor(np.percentile(np.concatenate(train_data, axis=0), 95,  axis=0)).cuda().float()
feature_min = torch.tensor(np.percentile(np.concatenate(train_data, axis=0), 5, axis=0)).cuda().float()
scale_factor = feature_max - feature_min
print(feature_max.max())
print(feature_min.min())
print(scale_factor)

model = LSTMRealTimeRegressor(input_dim).cuda()
model.load_state_dict(torch.load("tmp/mortality_realtime_regressor.pt"))
lstmp = LSTMPredictor(input_dim).cuda()
eval_victim_model = model

if args.transfer_victim_path != None:
    print("==== Testing transfer victim model.")
    eval_victim_model = LSTMRealTimeRegressor(input_dim=input_dim, num_classes=2, num_hidden=32)
    eval_victim_model.load_state_dict(torch.load(args.transfer_victim_path))
    eval_victim_model.cuda()
lstmp.load_state_dict(torch.load("tmp/mortality_predictor.pt"))
model.eval()



COUNT_FOOL = 0
COUNT_DECISION = 0
MAX_LOSS_LIST = []
MEAN_LOSS_LIST = []

ADV_MEAN_ERROR_SERIES_LIST = []
BENIGN_MEAN_ERROR_SERIES_LIST = []
for i, (data, target) in enumerate(test_loader):
    data = data.float().cuda()
    target = target.cuda()
    data = torch.max(torch.min(data, feature_max), feature_min)
    if args.attack == "clairvoyant":
        #x_adv = clairvoyant_each_step_attack_with_range(model, data.cuda(), target.cuda(), epsilon=args.eps*scale_factor, step_size=args.step_size*scale_factor,
        #                        max_iters=args.iters, min_value=feature_min, max_value=feature_max, t_start=args.t_start, t_end=args.t_end)
        x_adv = clairvoyant_attack_with_range_test(args, model, data.cuda(), target.cuda(), t_start=args.t_start, t_end=args.t_end)
    elif args.attack == "greedy":
        # x_adv = greedy_each_step_attack_with_range(model, data.cuda(), target.cuda(), epsilon=args.eps*scale_factor, step_size=args.step_size*scale_factor, max_iters=args.iters,
        #                                    min_value=feature_min, max_value=feature_max, t_start=args.t_start, t_end=args.t_end)
        x_adv = greedy_each_step_attack_with_range_test(args, model, data.cuda(), target.cuda(), t_start=args.t_start, t_end=args.t_end)
    elif args.attack == "predictive":
        #x_adv = predictive_each_step_attack_with_range(model, data.cuda(), target.cuda(), epsilon=args.eps*scale_factor, step_size=args.step_size*scale_factor, max_iters=args.iters,
        #                                         min_value=feature_min, max_value=feature_max, predictive_steps=args.k, t_start=args.t_start, t_end=args.t_end)
        x_adv = predictive_each_step_attack_with_range(args, model, lstmp, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)
    elif args.attack == "clairvoyant_k":
        x_adv = clairvoyant_each_step_attack_with_range___p(args, model, data.cuda(), target.cuda(), t_start=args.t_start, t_end=args.t_end)                                            
    elif args.attack == "IID_predictive":
        x_adv = custom_predictive_each_step_attack_with_range(args, model, attacks.get_IID_pred_func, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)
    elif args.attack == "permuted_predictive":
        x_adv = custom_predictive_each_step_attack_with_range(args, model, attacks.get_permuted_pred_func, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)
    elif args.attack == "uniform_predictive":
        x_adv = uniform_predictive_each_step_attack_with_range(args, model, None, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)



    if i == 0 and args.save_result_image == True:

        import matplotlib.pyplot as plt
        COLUMNS = 5
        data_ = data.cpu().transpose(1,2).detach().numpy()
        x_adv_ = x_adv.cpu().transpose(1,2).detach().numpy()
        for j in range(COLUMNS):
            #plt.subplots_adjust(left=0.0, right=0.1, top=0.2, bottom=0.1)
            plt.subplot(2, COLUMNS, j + 1)
            plt.imshow(data_[j], cmap='gray', vmin=feature_min.min(), vmax=feature_max.max())
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)

            
            plt.subplot(2, COLUMNS, 1*COLUMNS +j + 1)
            plt.imshow(x_adv_[j], cmap='gray', vmin=feature_min.min(), vmax=feature_max.max())
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.tight_layout(w_pad=0.8, h_pad=0.1)
            plt.show()
            
        plt.savefig("mortality_advplot.jpg")
        plt.savefig("mortality_advplot.pdf")
    
    with torch.no_grad():
        benign_pred = eval_victim_model(data)
        adv_pred = eval_victim_model(x_adv)
        error, mean_max_error, mean_mean_error, _ = compute_errors(args.t_start, args.t_end, adv_pred, target.cuda())
        MAX_LOSS_LIST.append(mean_max_error)
        MEAN_LOSS_LIST.append(mean_mean_error)
        
        # for ts_ranged
        _, _, _, benign_mean_error_series = compute_errors(0, benign_pred.size(1), benign_pred, target.cuda())
        _, _, _, adv_mean_error_series = compute_errors(0, adv_pred.size(1), adv_pred, target.cuda())
        BENIGN_MEAN_ERROR_SERIES_LIST.append(benign_mean_error_series.unsqueeze(0))
        ADV_MEAN_ERROR_SERIES_LIST.append(adv_mean_error_series.unsqueeze(0))

        #rg = range(args.t_start, args.t_end) if args.eval_last == False else [-1]
        rg = range(args.t_eval_start, args.t_eval_end) if args.eval_last == False else [-1]
        for t in rg:
            COUNT_DECISION += data.size(0)
            pred_change = benign_pred[:, t, :].max(dim=1)[1] != adv_pred[:, t, :].max(dim=1)[1]
            benign_true = benign_pred[:, t, :].max(dim=1)[1] == target
            COUNT_FOOL += (pred_change&benign_true).float().sum()

    if i >= args.NUM_TEST_BATCH - 1:
        break

print("Fool rate,", (COUNT_FOOL/COUNT_DECISION).item(), "max_loss,", torch.mean(torch.tensor(MAX_LOSS_LIST)).item(),
      "mean_loss,", torch.mean(torch.tensor(MEAN_LOSS_LIST)).item())

benign_result_mean_error_series = torch.cat(BENIGN_MEAN_ERROR_SERIES_LIST, dim=0).mean(dim=0)
adv_result_mean_error_series = torch.cat(ADV_MEAN_ERROR_SERIES_LIST, dim=0).mean(dim=0)
print("benign_result_mean_error_series:", benign_result_mean_error_series.cpu().detach().numpy())
print("adv_result_mean_error_series:", adv_result_mean_error_series.cpu().detach().numpy())

sys.exit(0)
