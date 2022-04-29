from __future__ import absolute_import
from __future__ import print_function
from pickle import EMPTY_SET
import sys
sys.path.append("./")
sys.path.append("../attack-codes")
import attacks
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
import model as model_module
import config
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float, default=None)
parser.add_argument('--step_size', type=float, default=None)
parser.add_argument('--iters', type=int, default=None)
parser.add_argument('--attack', type=str, choices=['greedy', 'predictive', 'clairvoyant', 'uniform_predictive',
                                                   'uniform_ocp_predictive', 'ocp_predictive', 'ocp_clairvoyant',
                                                   'IID_predictive', 'permuted_predictive', 'clairvoyant_k'])
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--t_start', type=int, default=0)
parser.add_argument('--t_end', type=int, default=28)

parser.add_argument('--t_eval_start', type=int, default=21)
parser.add_argument('--t_eval_end', type=int, default=28)

parser.add_argument('--eval_last', dest="eval_last", action="store_true")
parser.add_argument('--no-eval_last', dest="eval_last", action="store_false")
parser.set_defaults(eval_last=True)
parser.add_argument('--NUM_TEST_BATCH', type=int, default=5)

parser.add_argument('--max_attack', dest='max_attack', action='store_true')
parser.add_argument('--no-max_attack', dest='max_attack', action='store_false')
parser.set_defaults(max_attack=False)

parser.add_argument('--ts_ranged', dest='ts_ranged', action='store_true')
parser.add_argument('--no-ts_ranged', dest='ts_ranged', action='store_false')
parser.set_defaults(ts_ranged=False)

parser.add_argument('--transfer_victim_key', type=str, default=None)

parser.add_argument('--save_result_image', dest='save_result_image', action='store_true')
parser.set_defaults(save_result_image=False)

parser.add_argument('--num_sample', type=int, default=1)

args = parser.parse_args()
assert(args.num_sample > 0)

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
train_tensor_dataset = TensorDataset(train_dataset.data[train_label_mask].transpose(1, 2)/128.0, train_targets)
test_tensor_dataset = TensorDataset(test_dataset.data[test_label_mask].transpose(1, 2)/128.0, test_targets)

train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=128, shuffle=False)


def main(model_func, train_func):
    global model
    print("==> training")
    _, (data, target) = next(enumerate(train_loader))
    input_dim = data.shape[1]
    model = model_func(input_dim, num_classes=len(LABELS_IN_USE), num_hidden=4)
    return model

torch.manual_seed(0)
model = main(LSTMRealTimeRegressor, None)
if args.num_sample == 1:
    lstmp = model_module.LSTMPredictor(28)
else:
    lstmp = model_module.StochasticLSTMPredictor(28, hidden_dim=32, latent_dim=16)
#print(model)

model.load_state_dict(torch.load("tmp/MNIST_rnn_regressor.pt"))
if args.num_sample == 1:
    lstmp.load_state_dict(torch.load("tmp/MNIST_predictor.pt"))
else:
    lstmp.load_state_dict(torch.load("tmp/MNIST_stochastic_predictor.pt"))

eval_victim_model = model
print("--=-=-=-=-=")
if args.transfer_victim_key != None:
    model_key = args.transfer_victim_key
    print("==== Testing transfer victim model.")
    eval_victim_model = config.MNIST_MODEL_DICT[model_key](**config.MNIST_MODEL_ARG_DICT[model_key])
    #LSTMRealTimeRegressor(input_dim=28, num_classes=len(LABELS_IN_USE), num_hidden=config.MNIST_TRANSFER_VICTIM_HIDDEN)
    eval_victim_model.load_state_dict(torch.load(f"tmp/{config.MNIST_MODEL_NAME_DICT[model_key]}"))
    eval_victim_model.cuda()

model.cuda()
lstmp.cuda()
 
# Testing model
COUNT_CORRECT = 0
COUNT_TOTAL = 0
with torch.no_grad():
    for _, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        out = model(data)
        #print(out.size())
        COUNT_CORRECT += (out[:, -1, :].max(dim=1)[1] == target).float().sum()
        COUNT_TOTAL += out.size(0)
print("Victim model last time step accuarcy: {:.02f}".format(COUNT_CORRECT/COUNT_TOTAL))

def get_loss_func(num_loss_step=1, gamma=1.0, max_attack=False):
    if isinstance(gamma, float) == True:
        gamma_array = torch.ones(num_loss_step).cuda()
        for j in range(1, num_loss_step):
            gamma_array[j:]*=gamma
    elif isinstance(gamma, list) == True or isinstance(gamma, np.ndarray) == True :
        gamma_array = torch.tensor(gamma).cuda()
    else:
        assert(False)

    def loss_func(out, label, gamma_array=gamma_array, target_label=None):
        # label: BATCH 
        gamma_array = gamma_array.repeat(out.size(0)).float()
        num_out = out[:, -num_loss_step:, :].reshape((-1,  len(LABELS_IN_USE)))
        true_label = label.unsqueeze(1).repeat(1, num_loss_step).reshape(-1)
        loss_values = F.cross_entropy(num_out, true_label, reduce='none')*gamma_array
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


def greedy_attack_test(args, model, current_data, true_label):
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0
    
    batch_x_adv = attacks.greedy_each_step_attack(model, batch_x, batch_y, epsilon=EPSILON,
                        step_size=STEP_SIZE, max_iters=ITERS,
                        min_value=MIN_VALUE, max_value=MAX_VALUE,\
                        loss_func=get_loss_func(num_loss_step=1))

    print("=== greedy each step attack ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        #mse_error, mean_max_ad, mean_mean_ad = compute_errors(adv_out, batch_y)
        #print("Max ad: {:.04f}, Mean ad: {:.04f}".format(mean_max_ad, mean_mean_ad))
    return batch_x_adv

def greedy_each_step_attack_with_range_test(args, model, current_data, true_label, t_start, t_end):
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0
    
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
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0

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

def clairvoyant_only_current_perturb_attack_with_range_test(args, model, current_data, true_label, t_start, t_end):
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0

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
    batch_x_adv = attacks.clairvoyant_only_current_perturb_each_step_attack_with_range(**arg_dict)


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
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0

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
                "is_ts_ranged":args.ts_ranged,
                "num_sample":args.num_sample}
    batch_x_adv = attacks.predictive_each_step_attack_with_range(**arg_dict)


    print("=== predictive_attack_test ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return batch_x_adv

def custom_predictive_each_step_attack_with_range(args, model, gpm, current_data, true_label,
                                            t_start, t_end):
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0

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

def predictive_only_current_perturb_each_step_attack_with_range(args, model, pred_model, current_data, true_label,
                                            t_start, t_end):
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0

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
                "t_end":t_end}
    batch_x_adv = attacks.predictive_only_current_perturb_each_step_attack_with_range(**arg_dict)


    print("=== predictive_attack_test ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return batch_x_adv

def uniform_predictive_each_step_attack_with_range(args, model, pred_model, current_data, true_label,
                                            t_start, t_end):
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0

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


    print("=== predictive_attack_test ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return batch_x_adv


def uniform_predictive_only_current_perturb_each_step_attack_with_range(args, model, pred_model, current_data, true_label,
                                            t_start, t_end):
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0

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
    batch_x_adv = attacks.uniform_predictive_only_current_perturb_each_step_attack_with_range(**arg_dict)


    print("=== predictive_attack_test ===")
    with torch.no_grad():
        benign_out = model(batch_x)
        adv_out = model(batch_x_adv)
        benign_loss = criterion(benign_out[:, -1, :], batch_y)
        adv_loss = criterion(adv_out[:, -1, :], batch_y)
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(benign_loss.item(), adv_loss.item()))
        
    return batch_x_adv

def clairvoyant_each_step_attack_with_range___p(args, model, current_data, true_label, t_start, t_end):
    batch_x = current_data.float().cuda()
    batch_y = true_label.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
    EPSILON = args.eps*ONES_LIKE_FRAME
    STEP_SIZE = args.step_size*ONES_LIKE_FRAME
    ITERS = args.iters
    MAX_VALUE = ONES_LIKE_FRAME*1.0
    MIN_VALUE = ONES_LIKE_FRAME*0.0

    NUM_PREDICTION = args.k

    arg_dict = {"model":model,
            "current_data":batch_x,
            "true_label":batch_y,
            "epsilon":EPSILON,
            "step_size":STEP_SIZE,
            "max_iters":ITERS,
            "min_value":MIN_VALUE,
            "max_value":MAX_VALUE,
            "get_loss_func":  get_loss_func_max if args.max_attack else get_loss_func_mean,
            "pred_func":None,
            "predictive_steps":NUM_PREDICTION,
            "t_start":t_start,
            "t_end":t_end}
    batch_x_adv = attacks.clairvoyant_each_step_attack_with_range___p(**arg_dict)


    print("=== clairvoyant_attack_with_range_test ===")
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
    out_pred_range = out_pred[:, t_start:t_end].reshape(-1, len(LABELS_IN_USE))
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

_, (data, target) = next(enumerate(test_loader))

feature_max = torch.ones((28)).float().cuda()
feature_min = torch.zeros((28)).float().cuda()
scale_factor = feature_max - feature_min

EPS = args.eps
STEP_SIZE = args.step_size
ITERS = args.iters
K = args.k

NUM_TEST_BATCH = args.NUM_TEST_BATCH
COUNT_FOOL = 0
COUNT_DECISION = 0
COUNT_NAT_FOOL = 0
MAX_LOSS_LIST = []
MEAN_LOSS_LIST = []
ADV_MEAN_ERROR_SERIES_LIST = []
BENIGN_MEAN_ERROR_SERIES_LIST = []
for i, (data, target) in enumerate(test_loader):
    data = data.cuda()
    target = target.cuda()
    if args.attack == "clairvoyant":
        x_adv = clairvoyant_attack_with_range_test(args, model, data.cuda(), target.cuda(),
                            args.t_start, args.t_end)
    elif args.attack == "ocp_clairvoyant":
        x_adv = clairvoyant_only_current_perturb_attack_with_range_test(args, model, data.cuda(), target.cuda(), args.t_start, args.t_end)
    elif args.attack == "greedy":
        x_adv = greedy_each_step_attack_with_range_test(args, model, data.cuda(), target.cuda(), args.t_start, args.t_end)
    elif args.attack == "predictive":
        x_adv = predictive_each_step_attack_with_range(args, model, lstmp, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)
    elif args.attack == "clairvoyant_k":
        x_adv = clairvoyant_each_step_attack_with_range___p(args, model, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)                                                 
    elif args.attack == "IID_predictive":
        x_adv = custom_predictive_each_step_attack_with_range(args, model, attacks.get_IID_pred_func, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)                                      
    elif args.attack == "permuted_predictive":
        x_adv = custom_predictive_each_step_attack_with_range(args, model, attacks.get_permuted_pred_func, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)
    elif args.attack == "ocp_predictive":
        x_adv = predictive_only_current_perturb_each_step_attack_with_range(args, model, lstmp, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)
    elif args.attack == "uniform_predictive":
        x_adv = uniform_predictive_each_step_attack_with_range(args, model, lstmp, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)
    elif args.attack == "uniform_ocp_predictive":
        x_adv = uniform_predictive_only_current_perturb_each_step_attack_with_range(args, model, lstmp, data.cuda(), target.cuda(),
                                            args.t_start, args.t_end)
    else:
        assert(False)
    print("Wrong rate (before):", 1 - (model(data.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
    print("Wrong rate (after):", 1 - (model(x_adv.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())

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
            COUNT_NAT_FOOL += (benign_pred[:, t, :].max(dim=1)[1] != target).float().sum()

    if i == 0 and args.save_result_image == True:
        import matplotlib.pyplot as plt
        COLUMNS = 10
        data_ = data.cpu().transpose(1,2).detach().numpy()
        x_adv_ = x_adv.cpu().transpose(1,2).detach().numpy()
        for j in range(COLUMNS):
            #plt.subplots_adjust(left=0.0, right=0.1, top=0.2, bottom=0.1)
            plt.subplot(2, COLUMNS, j + 1)
            plt.imshow(data_[j], cmap='gray', vmin=0, vmax=1)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)

            
            plt.subplot(2, COLUMNS, 1*COLUMNS +j + 1)
            plt.imshow(x_adv_[j], cmap='gray', vmin=0, vmax=1)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.tight_layout(w_pad=0.8, h_pad=0.1)
            plt.show()
            
        plt.savefig("mnist_advplot.jpg")
        plt.savefig("mnist_advplot.pdf")
    
    if i >= NUM_TEST_BATCH - 1:
        break
    
print("Fool rate,", (COUNT_FOOL/(COUNT_DECISION-COUNT_NAT_FOOL)).item(), "max_loss,", torch.mean(torch.tensor(MAX_LOSS_LIST)).item(),
      "mean_loss,", torch.mean(torch.tensor(MEAN_LOSS_LIST)).item())

benign_result_mean_error_series = torch.cat(BENIGN_MEAN_ERROR_SERIES_LIST, dim=0).mean(dim=0)
adv_result_mean_error_series = torch.cat(ADV_MEAN_ERROR_SERIES_LIST, dim=0).mean(dim=0)
print("benign_result_mean_error_series:", benign_result_mean_error_series.cpu().detach().numpy())
print("adv_result_mean_error_series:", adv_result_mean_error_series.cpu().detach().numpy())