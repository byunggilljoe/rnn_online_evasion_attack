import torch
import torch.nn.functional as F
from mimic3models.in_hospital_mortality.torch.utils.torch import Normalizer, Bounder

class Attack(object):
    def __init__(self, model):
        self.model = model

    #perturb -> compute_grad -> get_loss

    def get_loss(self, x, perturb, y_victim, y_target):
        raise NotImplementedError()

    def compute_grad(self, x_victim, x_adv, y_victim, y_target):
        loss = self.get_loss(x_victim, x_adv, y_victim, y_target)
        loss.backward()
        #print(x_adv.grad.data[0].abs().sum())
        return x_adv.grad.data

    def perturb(self, x_victim, y_victim, y_target, p, epsilon, step_size, max_iters, min_value, max_value, decay=0.00, device="cuda:0", not_change_mode=False):
        x_victim = x_victim.to(device)
        y_victim = y_victim.to(device)
        self.model.to(device)
 
        x_adv = x_victim.clone().to(device)
        momentum = torch.zeros_like(x_adv)
        if not_change_mode == False:
            self.model.eval()
        for i in range(max_iters): 
            x_adv.requires_grad = True
            self.model.zero_grad()
            grad = self.compute_grad(x_victim, x_adv, y_victim, y_target)
            momentum = grad + momentum * decay 
            x_adv = (x_adv + Normalizer.normalize(momentum, p)*step_size)
            if p != "l0":
                perturb = Bounder.bound(x_adv - x_victim, epsilon, p) 
                x_adv = torch.clamp(x_victim + perturb, min=min_value, max=max_value).detach()
            else:
                perturb = Bounder.l0_bound_sparse(x_adv - x_victim, epsilon, x_victim) 
                x_adv = (x_victim + perturb).detach()



        return x_adv

## Real Implementation

class PGDAttack(Attack):
    def __init__(self, model):
        super(PGDAttack, self).__init__(model)  
        #self.criterion = torch.nn.CrossEntropyLoss()
        
    def get_loss(self, x, x_adv, y_victim, y_target, device="cuda:0"):
        #logits = self.model(x_adv)
        #if y_target == None:
        #    return self.criterion(logits, y_victim)
        #else:
        #    return -self.criterion(logits, y_target)

        logits = self.model(x_adv)
        num_classes = logits.size(1)
        if y_target is not None:
            target_one_hot = torch.eye(num_classes)[y_target].to(device)
            other_one_hot = 1.0 - target_one_hot
            target_logit = torch.sum(logits * target_one_hot, dim=1)
            other_logit = torch.max(logits * other_one_hot - target_one_hot*999999, dim=1)[0]
            diff = torch.nn.functional.relu(other_logit - target_logit +10)
            loss = -torch.mean(diff)
        else:
            true_one_hot = torch.eye(num_classes)[y_victim].to(device)
            other_one_hot = 1.0 - true_one_hot 
            true_logit = torch.sum(logits * true_one_hot, dim=1)
            other_logit = torch.max(logits * other_one_hot - true_one_hot*999999, dim=1)[0]
            diff = torch.nn.functional.relu(true_logit  - other_logit +10)
            loss = -torch.mean(diff)

        return loss