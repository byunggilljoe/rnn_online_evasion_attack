import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.append("../")
import attacks
from attack_experiment import AttackExperiment

def get_loss_func(num_loss_step=1, gamma=1.0, max_attack=False):
    if isinstance(gamma, float) == True:
        gamma_array = torch.ones(num_loss_step).cuda()
        for j in range(1, num_loss_step):
            gamma_array[j:]*=gamma
    elif isinstance(gamma, list) == True or isinstance(gamma, np.ndarray) == True :
        gamma_array = torch.tensor(gamma).cuda()
    else:
        assert(False)
# # # Greedy Attack at time $t$
# def get_loss_func(num_loss_step=1, gamma=1.0, max_attack=False):
#     gamma_array = torch.ones(num_loss_step).cuda()
#     for j in range(1, num_loss_step):
#         gamma_array[j:]*=gamma

    def loss_func(out, label, gamma_array=gamma_array, target_label=None):
        # label: BATCH 
        gamma_array = gamma_array.repeat(out.size(0)).float()
        num_out = out[:, -num_loss_step:].reshape((-1)) 
        true_label = label.unsqueeze(1).repeat(1, num_loss_step).reshape(-1)
        loss_values = F.binary_cross_entropy_with_logits(num_out, true_label, reduce='none')*gamma_array
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

def compute_errors(t_start, t_end, out_pred, true_label):
    # out_pred/true: BATCH x TIME
    # mse error
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    LEN = t_end - t_start
    out_pred_range = out_pred[:, t_start:t_end].reshape(-1)
    true_label_repeat = true_label.unsqueeze(1).repeat(1, LEN).view(-1)
    error = criterion(out_pred_range,
                        true_label_repeat)
    error_reshaped = error.view(-1, LEN)
    # mean of max absolute deviation
    mean_max_ad = error_reshaped.max(dim=1)[0].mean()

    # mean of mean absolute deviation
    mean_mean_ad = error_reshaped.mean(dim=1).mean()

    return error, mean_max_ad, mean_mean_ad, error_reshaped.mean(dim=0)
def binary_adv_accuracy(benign_pred, adv_preds, y):
    BOOL_NAT_FOOL = (torch.round(torch.sigmoid(benign_pred)) != y)
    BOOL_ADV_CORRECT = (~BOOL_NAT_FOOL)&(torch.round(torch.sigmoid(adv_preds)) == y)
    COUNT_NAT_FOOL = BOOL_NAT_FOOL.float().sum()
    COUNT_ADV_CORRECT = BOOL_ADV_CORRECT.float().sum()
    acc = COUNT_ADV_CORRECT/(benign_pred.size(0) - COUNT_NAT_FOOL)
    
    return acc

class SentimentAttackExperiment(AttackExperiment):
    def __init__(self, args):
        super().__init__(args)
        
        self.get_loss_func = get_loss_func_mean if args.max_attack == False else get_loss_func_max
        self.get_loss_func_mean = get_loss_func_mean
        self.get_loss_func_max = get_loss_func_max
        self.compute_errors = compute_errors
        self.get_pred_func = get_pred_func

        self.MAX_VALUE = None # set in get_preparation
        self.MIN_VALUE = None # set in get_preparation
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def load_data_and_model(self):
        super().load_data_and_model()
        # sentiment attack parameters should be normalized
        _, batch = next(enumerate(self.train_loader))
        self.embed_dim = 50
        self.feature_max = torch.tensor(np.percentile(self.model.embed(batch.text).view(-1, self.embed_dim).cpu().detach().numpy(), 95, axis=0)).cuda().float()
        self.feature_min = torch.tensor(np.percentile(self.model.embed(batch.text).view(-1, self.embed_dim).cpu().detach().numpy(), 5, axis=0)).cuda().float()
        self.scale_factor = self.feature_max - self.feature_min

        self.MAX_VALUE = self.feature_max
        self.MIN_VALUE = self.feature_min

    def get_preparation(self, args, steering_net, test_generator, start_index):
        batch_x, batch_y, criterion, arg_dict = super().get_preparation(args, steering_net, test_generator, start_index)
        arg_dict["step_size"] = arg_dict["step_size"]*self.scale_factor
        arg_dict["epsilon"]  = arg_dict["epsilon"]*self.scale_factor
        
        model = self.model
        class WrapperNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model.forward_embed(x)

        arg_dict["model"] = WrapperNet().cuda()
        arg_dict["current_data"] = model.embed(arg_dict["current_data"].long())
        return batch_x, batch_y, criterion, arg_dict

    def get_targeted_label(HEIGHT, BATCH):
        return np.array([np.sin(np.linspace(0, 4*np.pi, 20)) * HEIGHT for _ in range(BATCH)])

    def get_adv_results(self, embed_dict, true_label, forward_func, do_print=False):
        self.model.eval()
        
        acc_sum_list = []
        # rg = range(args.t_start, args.t_end) if args.eval_last == False else [-1]
        rg = range(self.args.t_eval_start, self.args.t_eval_end) if self.args.eval_last == False else [-1]
        for t in rg:
            acc_list = [1-binary_adv_accuracy(forward_func(embed_data_p[0])[:, t, :].squeeze(1), 
                                            forward_func(embed_data_p[1])[:, t, :].squeeze(1), 
                                            true_label) for key, embed_data_p in embed_dict.items()]
            if len(acc_sum_list) == 0:
                acc_sum_list = acc_list
            else:
                for i in range(acc_list):
                    acc_sum_list[i] += acc_list[i]
        acc_mean_list = [ase/len(rg) for ase in acc_sum_list]

        
        key_list = [key for key, embed_data_p in embed_dict.items()]
        if do_print:
            for i in range(len(acc_mean_list)):
                print("Fool rate ({}): {}".format(key_list[i], acc_mean_list[i]))
        return acc_mean_list, key_list

    def test_and_finish(self, params_list, batch_x_adv_list, criterion, eval_model, prefix):
        for params, batch_x_adv in zip(params_list, batch_x_adv_list):
            batch_x = params["current_data"]
            batch_y = params["true_label"]
            target_label = params["target_label"]

            NUM_TEST_BATCH = self.args.NUM_TEST_BATCH

            self.get_adv_results({self.args.attack:(batch_x, batch_x_adv)}, batch_y, self.model.forward_embed, do_print=True)