import numpy as np
import torch
import torch.nn.functional as F
from mimic3models.metrics import print_metrics_binary

def test_model_regression(model, data_loader):
    model.eval()
    with torch.no_grad():
        pred = []
        true_y = []
        for i, (x, y) in enumerate(data_loader):
            x = x.cuda()
            y = y.cuda()
            
            out = model(x)
            prob = F.softmax(out, dim=1)[:, 1]

            pred.append(prob.cpu().detach().numpy())
            true_y.append(y.cpu().detach().numpy())
        pred = np.concatenate(pred, axis=0)
        #pred = np.concatenate([np.ones_like(pred) - pred, pred], axis=1)
        true_y = np.concatenate(true_y, axis=0)
        return print_metrics_binary(true_y, pred)
    
def test_model_realtime_regression(model, data_loader):
    model.eval()
    with torch.no_grad():
        pred = []
        true_y = []
        for i, (x, y) in enumerate(data_loader):
            x = x.cuda()
            y = y.cuda()
            
            out = model(x)
            prob = F.softmax(out[:, -1, :], dim=1)

            pred.append(prob.cpu().detach().numpy())
            true_y.append(y.cpu().detach().numpy())
        pred = np.concatenate(pred, axis=0)
        #pred = np.concatenate([np.ones_like(pred) - pred, pred], axis=1)
        true_y = np.concatenate(true_y, axis=0)
        return print_metrics_binary(true_y, pred)
        

def test_model_trigger(model, data_loader):
    model.eval()
    with torch.no_grad():
        pred = []
        true_y = []
        for i, (x, y) in enumerate(data_loader):
            x = x.cuda()
            y = y.cuda()
            
            out = model(x)
            prob = F.softmax(out, dim=1)

            pred.append(prob.cpu().detach().numpy())
            true_y.append(y.cpu().detach().numpy())
        pred = np.concatenate(pred, axis=0)
        #pred = np.concatenate([np.ones_like(pred) - pred, pred], axis=1)
        true_y = np.concatenate(true_y, axis=0)
        pred_y = np.argmax(pred, axis=1)
        success_ratio = (true_y == pred_y).astype(np.float32).mean()
        print("Trigger success ratio:", success_ratio)
        return success_ratio

        