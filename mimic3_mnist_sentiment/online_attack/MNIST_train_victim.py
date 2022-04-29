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
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset
from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, LSTMRealTimeRegressor
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_realtime_regression
import config
model_key = "default"
# print(config.MNIST_MODEL_NAME_DICT[model_key])
# print(config.MNIST_MODEL_DICT[model_key])
# print(config.MNIST_MODEL_ARG_DICT[model_key])
# print(config.MNIST_MODEL_DICT[model_key](**config.MNIST_MODEL_ARG_DICT[model_key]))
# sys.exit(0)
# num_hidden = 4

parser = argparse.ArgumentParser()
parser.add_argument("--model_key", default="default", required=False, type=str)
parser.add_argument("--trial", default=0, required=True, type=int)
args = parser.parse_args()

model_key = args.model_key


#print(val_poison_targets)
trans = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST("./data", train=False, transform=trans, download=True)
LABELS_IN_USE = [3, 8] #[0, 1]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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

train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=128, shuffle=False)



def train_realtime_all(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    class_weights = torch.tensor([1.0]*len(LABELS_IN_USE)).cuda()
    best_state_dict = None#model.state_dict()
    best_score = 0
    NUM_LAST_TIMESTAMP = 6 #14 < 6
    for e in range(100):
        print("epoch:", e)
        
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            model.zero_grad()
            out = model(x)
            y_reshaped = y.unsqueeze(dim=1).repeat((1, NUM_LAST_TIMESTAMP)).view(-1, 1).squeeze(1)
            out_reshaped = out[:, -NUM_LAST_TIMESTAMP:, :].reshape(-1, len(LABELS_IN_USE))
            
            logprob = torch.log_softmax(out_reshaped,dim=1)
            loss = F.nll_loss(logprob, y_reshaped, weight=class_weights)
            loss.backward()
            optimizer.step()
    
        print((out[:, -1, :].max(dim=1)[1] == y).float().mean())
    if best_state_dict == None:
        best_state_dict = model.state_dict()
        
    return best_state_dict

model = None
def main(train_func):
    global model, num_hidden
    print("==> training")
    _, (data, target) = next(enumerate(train_loader))
    input_dim = data.shape[1]
    model = config.MNIST_MODEL_DICT[model_key](**config.MNIST_MODEL_ARG_DICT[model_key])
    train_func(model)
    return model


# # # Training 

# # In[32]:


# main(LSTMRegressor, train)


# # In[35]:


# main(LSTMRealTimeRegressor, train_realtime_last)


# # In[58]:
if args.trial == 0:
    torch.manual_seed(0)
main(train_realtime_all)
model_file_name = config.MNIST_MODEL_NAME_DICT[model_key].split(".pt")[0] +f"_trial_{args.trial}.pt"
torch.save(model.state_dict(), f"./tmp/{model_file_name}")
