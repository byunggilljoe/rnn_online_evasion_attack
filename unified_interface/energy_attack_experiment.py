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

    def loss_func(out, label, gamma_array=gamma_array, target_label=None):
        # label: BATCH 
        gamma_array = gamma_array.repeat(out.size(0)).float()
        num_out = out[:, -num_loss_step:].reshape((-1,))
        true_label = label[:, -num_loss_step:].reshape((-1,)).float()
        

        if target_label == None:
            loss_values = F.mse_loss(num_out, true_label, reduction='none')*gamma_array
        else:
            target_label = target_label.float()
            loss_values = -F.mse_loss(num_out, target_label.reshape(-1), reduction='none')*gamma_array

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
    criterion = torch.nn.MSELoss(reduction='none')
    LEN = t_end - t_start
    out_pred_range = out_pred[:, t_start:t_end].reshape(-1,)
    true_label = true_label[:, t_start:t_end].reshape(-1,)
    error = criterion(out_pred_range,
                        true_label)
    error_reshaped = error.view(-1, LEN)
    # mean of max absolute deviation
    mean_max_ad = error_reshaped.max(dim=1)[0].mean()

    # mean of mean absolute deviation
    mean_mean_ad = error_reshaped.mean(dim=1).mean()


    return error.mean(), mean_max_ad, mean_mean_ad

class EnergyAttackExperiment(AttackExperiment):
    def __init__(self, args):
        super().__init__(args)
        
        
        self.get_loss_func = get_loss_func_mean if args.max_attack == False else get_loss_func_max
        self.get_loss_func_mean = get_loss_func_mean
        self.get_loss_func_max = get_loss_func_max
        self.compute_errors = compute_errors
        self.get_pred_func = get_pred_func

        self.MAX_VALUE = 8.34 #3.0
        self.MIN_VALUE = -4.9
        self.criterion =torch.nn.MSELoss()
        

    def get_targeted_label(self, BATCH, true_label, t_start, NUM_WAVE):
        HEIGHT = 1.0
        return np.array([np.sin(np.linspace(0, 2*np.pi*NUM_WAVE, 50)).astype(np.float) * HEIGHT for _ in range(BATCH)])

    def test_and_finish(self, param_list, batch_x_adv_list, criterion, eval_model, prefix, elapsed_time):
        
        COUNT_DECISION = 0
        MAX_LOSS_LIST = []
        MEAN_LOSS_LIST = []


        TARGET_ACCORDANCE_ERROR = 0
        with torch.no_grad():
            for params, batch_x_adv in zip(param_list, batch_x_adv_list):
                batch_x = params["current_data"]
                batch_y = params["true_label"]
                target_label = params["target_label"]
                benign_pred = eval_model(batch_x)
                adv_pred = eval_model(batch_x_adv)
                error, mean_max_error, mean_mean_error = compute_errors(self.args.t_eval_start, self.args.t_eval_end, adv_pred, batch_y.cuda())
                MAX_LOSS_LIST.append(mean_max_error)
                MEAN_LOSS_LIST.append(mean_mean_error)
                COUNT_DECISION  += batch_x_adv.size(0)
                if target_label != None:
                    tae = criterion(adv_pred, target_label)
                    TARGET_ACCORDANCE_ERROR += tae*batch_x_adv.size(0)

                if len(MAX_LOSS_LIST) == 1 and self.args.targeted==True:
                    np.savez(f"../experiments/tmp_output/energy_{prefix}_{params['epsilon'][0]:.02f}_visualization_data.npz", batch_x=batch_x.cpu().detach().numpy(), batch_y=batch_y.cpu().detach().numpy(),
                                                                             benign_out=benign_pred.cpu().detach().numpy(),
                                                                             batch_x_adv=batch_x_adv.cpu().detach().numpy(),
                                                                             target_label=target_label.cpu().detach().numpy(),
                                                                             adv_out=adv_pred.cpu().detach().numpy())

        print("MSE,", torch.mean(torch.tensor(MEAN_LOSS_LIST)).item(), "max_loss,", torch.mean(torch.tensor(MAX_LOSS_LIST)).item(),
      "mean_loss,", torch.mean(torch.tensor(MEAN_LOSS_LIST)).item(), "elapsed_time,", elapsed_time)

        if target_label != None:
            print("Target accordance error:", (TARGET_ACCORDANCE_ERROR/COUNT_DECISION).item() )