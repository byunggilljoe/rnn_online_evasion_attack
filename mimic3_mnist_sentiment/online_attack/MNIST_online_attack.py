from __future__ import absolute_import
from __future__ import print_function

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
from mimic3models.in_hospital_mortality.torch.data import create_loader, load_data_48_76
from mimic3models.in_hospital_mortality.torch.eval_func import test_model_regression, test_model_realtime_regression



#print(val_poison_targets)
trans = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST("./data", train=False, transform=trans, download=True)
LABELS_IN_USE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_label_mask = [e in LABELS_IN_USE for e in train_dataset.targets]
test_label_mask = [e in LABELS_IN_USE for e in test_dataset.targets]

train_tensor_dataset = TensorDataset(train_dataset.data[train_label_mask]/128.0, train_dataset.targets[train_label_mask])
test_tensor_dataset = TensorDataset(test_dataset.data[test_label_mask]/128.0, test_dataset.targets[test_label_mask])

train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=128, shuffle=False)


def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    class_weights = torch.tensor([1.0, 1.0]).cuda()
    best_state_dict = None
    best_score = 0
    for e in range(100):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            #print(x.size())
            out = model(x)
            #print(out.size())
            logprob = torch.log_softmax(out, dim=1)
            loss = F.nll_loss(logprob, y, weight=class_weights)
            loss.backward()
            optimizer.step()
            #print(f"loss: {loss}")
        test_model_regression(model, train_loader)
        scores = test_model_regression(model, test_loader)

        if scores['prec1'] > 0.60 and scores['rec1'] > 0.30:
            score = 2/(1.0/scores['rec1'] + 1.0/scores['prec1'])
            if score > best_score:
                best_score = score
                best_state_dict = model.state_dict()
                print("best f1 score :", score)

    if best_state_dict == None:
        best_state_dict = model.state_dict()
        
    return best_state_dict

def train_realtime_last(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    class_weights = torch.tensor([1.0, 1.0]).cuda()
    best_state_dict = None
    best_score = 0
    for e in range(100):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            
            out = model(x)
            
            logprob = torch.log_softmax(out[:, -1, :], dim=1)
            loss = F.nll_loss(logprob, y, weight=class_weights)
            loss.backward()
            optimizer.step()
            
        test_model_realtime_regression(model, train_loader)
        scores = test_model_realtime_regression(model, test_loader)

        if scores['prec1'] > 0.60 and scores['rec1'] > 0.30:
            score = 2/(1.0/scores['rec1'] + 1.0/scores['prec1'])
            if score > best_score:
                best_score = score
                best_state_dict = model.state_dict()
                print("best f1 score :", score)

    if best_state_dict == None:
        best_state_dict = model.state_dict()
        
    return best_state_dict

def train_realtime_all(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    model.cuda()
    class_weights = torch.tensor([1.0]*len(LABELS_IN_USE)).cuda()
    best_state_dict = None#model.state_dict()
    best_score = 0
    NUM_LAST_TIMESTAMP = 28 #6
    for e in range(100):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
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
def main(model_func, train_func):
    global model
    print("==> training")
    _, (data, target) = next(enumerate(train_loader))
    input_dim = data.shape[1]
    model = model_func(input_dim, num_classes=10)

    best_state_dict = train_func(model)

    if not os.path.exists("./checkpoints/logistic_regression/mnist"):
        os.makedirs("./checkpoints/logistic_regression/mnist")
    torch.save(best_state_dict, "./checkpoints/logistic_regression/mnist/lstm.pt")


# # # Training 

# # In[32]:


# main(LSTMRegressor, train)


# # In[35]:


# main(LSTMRealTimeRegressor, train_realtime_last)


# # In[58]:


main(LSTMRealTimeRegressor, train_realtime_all)


# In[59]:


_, (data, target) = next(enumerate(train_loader))
data = data.cuda()
target = target.cuda()
out = model(data)
print((out[:, -1, :].max(dim=1)[1] == target).float().mean())


# # LSTM predictor

# In[60]:


class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = 1
        self.n_hidden = 128#16
        self.num_direction = 1
        assert self.num_direction in [1, 2]
        dropout_p = 0.3
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,                                    bias=True, batch_first=True,                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
        self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 150)
        self.fc2 = torch.nn.Linear(150, 28)
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


# In[8]:


lstmp = LSTMPredictor(28)
lstmp.cuda()
_, (data, target) = next(enumerate(test_loader))
data = data.cuda()
x_hat = lstmp(data)
lstmp.loss(x_hat, data)


# In[9]:


lstmp.train()
optimizer = torch.optim.Adam(lstmp.parameters(), lr=0.0001)
for e in range(1000):
    for i, (data, target) in enumerate(train_loader):
        lstmp.zero_grad()
        data = data.cuda()
        x_hat = lstmp(data)
        #loss = F.mse_loss(data, recons)
        loss = lstmp.loss(data, x_hat)
        loss.backward()
        optimizer.step()
    print(loss.item())


# In[61]:


import matplotlib.pyplot as plt
lstmp.eval()
data = data.cuda()
x_hat = lstmp(data)
x_hat_numpy = x_hat.cpu().detach().numpy()
plt.subplot(2, 1, 1)
plt.imshow(data.cpu().detach().numpy()[0], vmin=0, vmax=1.0)
plt.subplot(2, 1, 2)
plt.imshow(x_hat_numpy[0], vmin=0, vmax=1.0)


# # Greedy Attack at time $t$

# In[62]:


# Given data t=0 t=i, perturb data of t=i to lower attack loss.
def greedy_one_step_attack(model, current_data, true_label, epsilon, step_size, max_iters, min_value, max_value, num_step=1, is_last_step_attack=True):
    x_adv = current_data.clone().detach()
    delta = torch.zeros_like(x_adv)
    mask = torch.zeros_like(x_adv)
    mask[:, -num_step:, :] = 1.0
    model.train()
    x_adv.requires_grad = True
    eye = torch.eye(len(LABELS_IN_USE)).cuda()
    for i in range(max_iters):
        model.zero_grad()
        out = model(x_adv)
        if is_last_step_attack == True:
            last_out = out[:, -1, :]
            true_one_hot = eye[true_label]
            true_logit = (last_out * true_one_hot).max(dim=1)[0]
            wrong_logit = (last_out * (1 - true_one_hot)).sum(dim=1)
            loss = (wrong_logit - true_logit).mean()
        else:
            num_out = out[:, -num_step:, :].reshape(-1, 2)
            true_label_repeat = true_label.unsqueeze(1).repeat((1, num_step)).view(-1)
            #loss = F.nll_loss(torch.log_softmax(num_out, dim=1), true_label_repeat)
            
        loss.backward()
        grad = x_adv.grad.data
        
        #delta = torch.clamp(delta + mask*torch.sign(grad)*step_size, -epsilon, epsilon)
        #x_adv = torch.clamp(current_data + delta, min_value, max_value).detach()
        delta = torch.min(torch.max(delta + mask*torch.sign(grad)*step_size, -epsilon), epsilon)
        x_adv = torch.min(torch.max(current_data + delta, min_value), max_value).detach()
        x_adv.requires_grad = True
    return x_adv    


# In[12]:


feature_max = torch.ones((28)).float().cuda()
feature_min = torch.zeros((28)).float().cuda()
scale_factor = feature_max - feature_min
print(feature_max.max())
print(feature_min.min())
print(scale_factor)

# feature_max = feature_max.max()
# feature_min = feature_min.min()
# scale_factor = feature_max - feature_min


# In[63]:


x_adv = greedy_one_step_attack(model, data.cuda(), target.cuda(), epsilon=0.9*scale_factor, step_size=0.01*scale_factor, max_iters=100, min_value=feature_min, max_value=feature_max, num_step=1)
model.eval()
print("Wrong rate (before):", 1 - (model(data.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
print("Wrong rate (after):", 1 - (model(x_adv.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
plt.subplot(2,1,1)
plt.imshow(data[0].cpu().detach(), vmin=0, vmax=1.0)
plt.subplot(2,1,2)
plt.imshow(x_adv[0].cpu().detach(), vmin=0, vmax=1.0)
(x_adv - data.cuda())[:, :-1, :].square().sum()


# # All perturbation attack (num_step = 48)

# In[64]:


x_adv = greedy_one_step_attack(model, data.cuda(), target.cuda(), epsilon=0.07*scale_factor, step_size=0.01*scale_factor, max_iters=90, min_value=feature_min, max_value=feature_max, num_step=28, is_last_step_attack=True)
model.eval()
print("Wrong rate (before):", 1 - (model(data.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
print("Wrong rate (after):", 1 - (model(x_adv.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
plt.subplot(2,1,1)
plt.imshow(data[0].cpu().detach(), vmin=feature_min.min(), vmax=feature_max.max())
plt.subplot(2,1,2)
plt.imshow(x_adv[0].cpu().detach(), vmin=feature_min.min(), vmax=feature_max.max())


# # Sequential application of the greedy attack

# In[26]:


def greedy_each_step_attack(model, current_data, true_label, epsilon, step_size, max_iters, min_value, max_value):
    adv_data_t = torch.empty(current_data.size(0), 0, current_data.size(2)).cuda()
    for t in range(current_data.size(1)):
        data_t = torch.cat([adv_data_t, current_data[:, t:t+1, :]], dim=1)
        adv_data_t = greedy_one_step_attack(model, data_t, true_label.cuda(), epsilon=epsilon, step_size=step_size, max_iters=max_iters, min_value=min_value, max_value=max_value, num_step=1, is_last_step_attack=True)
        
    return adv_data_t


# In[65]:


x_adv = greedy_each_step_attack(model, data.cuda(), target.cuda(), epsilon=0.07*scale_factor, step_size=0.01*scale_factor, max_iters=90, min_value=feature_min, max_value=feature_max)


# In[66]:


model.eval()
print("Wrong rate (before):", 1 - (model(data.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
print("Wrong rate (after):", 1 - (model(x_adv.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
plt.subplot(2,1,1)
plt.imshow(data[0].cpu().detach(), vmin=feature_min.min(), vmax=feature_max.max())
plt.subplot(2,1,2)
plt.imshow(x_adv[0].cpu().detach(), vmin=feature_min.min(), vmax=feature_max.max())


# # Predictive online attack

# In[67]:


def get_predicted_data(current_data, predictor, predictive_steps):
    predicted_data_only = torch.empty(current_data.size(0), 0, current_data.size(2)).cuda()
    if predictive_steps == 0:
        return torch.cat([current_data, predicted_data_only], dim=1), predicted_data_only
    hidden_and_cell = predictor.get_init_hidden_cell(current_data.size(0))
    #print(predicted_data.size(), hidden_and_cell[0].size())
    out_predictor, hidden_and_cell = predictor.get_one_pred(current_data.clone(), hidden_and_cell)
    next_step = current_data[:, -1:, :]
    predicted_data_only = torch.cat([predicted_data_only, next_step], dim=1)
    
    for i in range(predictive_steps - 1):
        next_step, hidden_and_cell = predictor.get_one_pred(next_step, hidden_and_cell)
        predicted_data_only = torch.cat([predicted_data_only, next_step], dim=1)
    #print(len(torch.cat([current_data, predicted_data_only], dim=1), predicted_data_only))
    return torch.cat([current_data, predicted_data_only], dim=1), predicted_data_only


# In[68]:


# At time t
# Predict k future steps
# Perturb t ~ t + k steps
# Apply perturbation at t
def predictive_each_step_attack(model, current_data, true_label, epsilon, step_size, max_iters, min_value, max_value, predictive_steps=1):
    adv_data = torch.empty((current_data.size(0), 0, current_data.size(2))).cuda()
    for t in range(current_data.size(1)):
        data_t = current_data[:, :t+1, :] # fetch clean input to predict future value
        adv_data = torch.cat([adv_data, current_data[:, t:t+1, :]], dim=1) # fetch current input to start perturbation
        # predicted data
        (_, predicted_data_t_only) = get_predicted_data(data_t, lstmp, predictive_steps=predictive_steps)
        adv_data_t = greedy_one_step_attack(model, torch.cat([adv_data, predicted_data_t_only], dim=1), true_label.cuda(), epsilon=epsilon, step_size=step_size, max_iters=max_iters, min_value=min_value, max_value=max_value, num_step=1 + predictive_steps, is_last_step_attack=True) #num_step = 1 (current_step) + predictive_steps
        adv_data = adv_data_t[:, :t+1, :]
    return adv_data


# In[69]:


predicted_data, predicted_data_only = get_predicted_data(data[:, :10, :].cuda(), lstmp, 20)
print(predicted_data.size())
#plt.subplot(2, 1, 2)
plt.imshow(predicted_data[0].cpu().detach(), vmin=0, vmax=1.0)


# In[70]:


x_adv = predictive_each_step_attack(model, data.cuda(), target, epsilon=0.07*scale_factor, step_size=0.01*scale_factor, max_iters=90, min_value=feature_min, max_value=feature_max, predictive_steps=2)
model.eval()
print("Wrong rate (before):", 1 - (model(data.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
print("Wrong rate (after):", 1 - (model(x_adv.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
plt.subplot(2,1,1)
plt.imshow(data[0].cpu().detach(), vmin=0, vmax=1.0)
plt.subplot(2,1,2)
plt.imshow(x_adv[0].cpu().detach(), vmin=0, vmax=1.0)


# In[71]:


x_adv = predictive_each_step_attack(model, data.cuda(), target, epsilon=0.07*scale_factor, step_size=0.01*scale_factor, max_iters=90, min_value=feature_min, max_value=feature_max, predictive_steps=10)
model.eval()
print("Wrong rate (before):", 1 - (model(data.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
print("Wrong rate (after):", 1 - (model(x_adv.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
plt.subplot(2,1,1)
plt.imshow(data[0].cpu().detach(), vmin=0, vmax=1.0)
plt.subplot(2,1,2)
plt.imshow(x_adv[0].cpu().detach(), vmin=0, vmax=1.0)


# In[72]:


x_adv = predictive_each_step_attack(model, data.cuda(), target, epsilon=0.07*scale_factor, step_size=0.01*scale_factor, max_iters=90, min_value=feature_min, max_value=feature_max, predictive_steps=15)
model.eval()
print("Wrong rate (before):", 1 - (model(data.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
print("Wrong rate (after):", 1 - (model(x_adv.cuda())[:, -1, :].max(dim=1)[1]==target.cuda()).float().mean())
plt.subplot(2,1,1)
plt.imshow(data[0].cpu().detach(), vmin=0, vmax=1.0)
plt.subplot(2,1,2)
plt.imshow(x_adv[0].cpu().detach(), vmin=0, vmax=1.0)


# In[ ]:




