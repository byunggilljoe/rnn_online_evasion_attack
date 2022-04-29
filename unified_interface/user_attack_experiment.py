import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.append("../")
import attacks
from attack_experiment import AttackExperiment

LABELS_IN_USE = list(range(22))

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
        true_label = label[:, -num_loss_step:].reshape(-1)

        if target_label == None:
            loss_values = F.cross_entropy(num_out, true_label, reduction='none')*gamma_array
        else:
            loss_values = -F.cross_entropy(num_out, target_label.reshape(-1), reduction='none')*gamma_array

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
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    LEN = t_end - t_start
    out_pred_range = out_pred[:, t_start:t_end].reshape(-1, len(LABELS_IN_USE))
    true_label_repeat = true_label[:, t_start:t_end].reshape(-1)
    error = criterion(out_pred_range,
                        true_label_repeat)
    error_reshaped = error.view(-1, LEN)
    # mean of max absolute deviation
    mean_max_ad = error_reshaped.max(dim=1)[0].mean()

    # mean of mean absolute deviation
    mean_mean_ad = error_reshaped.mean(dim=1).mean()

    out_pred_indice = out_pred.max(dim=2)[1]
    true_label_all_repeat = true_label
    error_rate = (out_pred_indice != true_label_all_repeat).float().mean(dim=0)
    return error, mean_max_ad, mean_mean_ad, error_rate

class UserAttackExperiment(AttackExperiment):
    def __init__(self, args):
        super().__init__(args)
        
        
        self.get_loss_func = get_loss_func_mean if args.max_attack == False else get_loss_func_max
        self.get_loss_func_mean = get_loss_func_mean
        self.get_loss_func_max = get_loss_func_max
        self.compute_errors = compute_errors
        self.get_pred_func = get_pred_func

        self.MAX_VALUE = 3.4 #15
        self.MIN_VALUE =-3.4 #-10
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_targeted_label(self, BATCH, true_label, t_start, NUM_WAVE=2):
        NUM_STEP = 2*NUM_WAVE
        LEN_STEP = int(50/NUM_STEP) if NUM_STEP !=0 else 50
        target_label = []
        for i in range(50):
            target_label.append(int(i/LEN_STEP)%2) #3
            #target_label.append(int(i/LEN_STEP)%2 + 3) #3
        # o 
        # /\  5
        # x 3, 2, 1, 0, 4, 6, 7, 8, 9
        return np.array([target_label for _ in range(BATCH)])

    def test_and_finish(self, param_list, batch_x_adv_list, criterion, eval_model, prefix, elapsed_time):
        COUNT_FOOL = 0
        COUNT_DECISION = 0
        COUNT_NAT_FOOL = 0
        MAX_LOSS_LIST = []
        MEAN_LOSS_LIST = []
        ADV_MEAN_ERROR_SERIES_LIST = []
        BENIGN_MEAN_ERROR_SERIES_LIST = []

        COUNT_TARGET_MATCH = 0
        with torch.no_grad():
            for params, batch_x_adv in zip(param_list, batch_x_adv_list):
                batch_x = params["current_data"]
                batch_y = params["true_label"]
                target_label = params["target_label"]
                benign_pred = eval_model(batch_x)
                adv_pred = eval_model(batch_x_adv)
                error, mean_max_error, mean_mean_error, _ = compute_errors(self.args.t_start, self.args.t_end, adv_pred, batch_y.cuda())
                MAX_LOSS_LIST.append(mean_max_error)
                MEAN_LOSS_LIST.append(mean_mean_error)
                
                # for ts_ranged
                _, _, _, benign_mean_error_series = compute_errors(0, benign_pred.size(1), benign_pred, batch_y.cuda())
                _, _, _, adv_mean_error_series = compute_errors(0, adv_pred.size(1), adv_pred, batch_y.cuda())
                BENIGN_MEAN_ERROR_SERIES_LIST.append(benign_mean_error_series.unsqueeze(0))
                ADV_MEAN_ERROR_SERIES_LIST.append(adv_mean_error_series.unsqueeze(0))
                
                #rg = range(args.t_start, args.t_end) if args.eval_last == False else [-1]
                rg = range(self.args.t_eval_start, self.args.t_eval_end) if self.args.eval_last == False else [-1]
                for t in rg:
                    COUNT_DECISION += batch_x.size(0)
                    pred_change = benign_pred[:, t, :].max(dim=1)[1] != adv_pred[:, t, :].max(dim=1)[1]
                    benign_true = benign_pred[:, t, :].max(dim=1)[1] == batch_y[:, t]
                    COUNT_FOOL += (pred_change&benign_true).float().sum()
                    COUNT_NAT_FOOL += (benign_pred[:, t, :].max(dim=1)[1] != batch_y[:, t]).float().sum()
                    
                    # print(adv_pred[0, t, :].max(dim=0)[1], target_label[0, t], batch_y[0])
                    

                    if target_label != None:
                        COUNT_TARGET_MATCH += (adv_pred[:, t, :].max(dim=1)[1] == target_label[:, t]).float().sum()

                if len(BENIGN_MEAN_ERROR_SERIES_LIST) == 1 and self.args.targeted == True:
                    np.savez(f"../experiments/tmp_output/user_{prefix}_{params['epsilon'][0]:.02f}_visualization_data.npz", batch_x=batch_x.cpu().detach().numpy(), batch_y=batch_y.cpu().detach().numpy(),
                                                                             benign_out=benign_pred.cpu().detach().numpy(),
                                                                             batch_x_adv=batch_x_adv.cpu().detach().numpy(),
                                                                             target_label=target_label.cpu().detach().numpy(),
                                                                             adv_out=adv_pred.cpu().detach().numpy())
                                                                             
        print("Fool rate,", (COUNT_FOOL/(COUNT_DECISION-COUNT_NAT_FOOL)).item(), "max_loss,", torch.mean(torch.tensor(MAX_LOSS_LIST)).item(),
      "mean_loss,", torch.mean(torch.tensor(MEAN_LOSS_LIST)).item(), "elapsed_time,", elapsed_time)
      
        if target_label != None:
            print("Target match rate:", (COUNT_TARGET_MATCH/COUNT_DECISION).item() )