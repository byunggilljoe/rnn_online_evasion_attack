import numpy as np
import torch
from attack_experiment import AttackExperiment

class UdacityAttackExperiment(AttackExperiment):
    def __init__(self, args):
        super().__init__(args)
        
        from Adv_attack_and_defense_on_driving_model.BG_codes.bg_utils import get_dataloader_lstm, get_models, prediction_test, get_loss_func_max,\
                    get_loss_func_mean, get_loss_func, compute_errors, get_pred_func, plot_targeted_result
        
        self.get_loss_func = get_loss_func_mean if args.max_attack == False else get_loss_func_max
        self.get_loss_func_mean = get_loss_func_mean
        self.get_loss_func_max = get_loss_func_max
        self.compute_errors = compute_errors
        self.get_pred_func = get_pred_func

        self.MAX_VALUE = 0.5*10
        self.MIN_VALUE = -0.5*10

        self.criterion = torch.nn.MSELoss()
    
    def get_preparation(self, args, steering_net, test_generator, start_index):
        batch_x, batch_y, criterion, arg_dict = super().get_preparation(args, steering_net, test_generator, start_index)
        arg_dict["true_label"] = arg_dict["true_label"].float()

        return batch_x, batch_y, criterion, arg_dict

    def get_targeted_label(self, BATCH, true_label, t_start, NUM_WAVE=2):
        HEIGHT = 1.0
        target_label_all =  np.array([np.sin(np.linspace(0, 2*np.pi*NUM_WAVE, 20)) * HEIGHT for _ in range(BATCH)])
        if t_start > 0:
            target_label_all = np.concatenate([ true_label[:, :t_start].cpu().numpy(), target_label_all[:, t_start:] ], axis=1)       
        return target_label_all

    def test_and_finish(self, param_list, batch_x_adv_list, criterion, eval_model, prefix, elapsed_time):
        benign_loss_list = []
        adv_loss_list = []
        mean_ad_list = []
        max_ad_list = []
        mse_error_list = []
        target_accordance_error_list = []
        total_cnt = 0
        eval_model.eval()
        with torch.no_grad():
            for params, batch_x_adv in zip(param_list, batch_x_adv_list):
                batch_x = params["current_data"]
                batch_y = params["true_label"]
                target_label = params["target_label"]
                benign_out = eval_model(batch_x)
                adv_out = eval_model(batch_x_adv)
                benign_loss = criterion(benign_out, batch_y)
                adv_loss = criterion(adv_out, batch_y)

                mse_error, mean_max_ad, mean_mean_ad, _ = self.compute_errors(self.args.t_eval_start, self.args.t_eval_end, adv_out, batch_y)

                batch_cnt = batch_x.size(0)
                benign_loss_list.append(benign_loss.item() * batch_cnt)
                adv_loss_list.append(adv_loss.item() * batch_cnt)
                mean_ad_list.append(mean_mean_ad * batch_cnt)
                max_ad_list.append(mean_max_ad * batch_cnt)
                mse_error_list.append(mse_error * batch_cnt)
                total_cnt += batch_cnt
                if target_label != None:
                    target_accordance_error = criterion(adv_out, target_label)
                    target_accordance_error_list.append(target_accordance_error.item()*batch_cnt)
                    print("Target comply error: {:.04f}".format(target_accordance_error.item()))
                    # plot_targeted_result(adv_out, target_label, prefix)
                """
                adv_eps = (batch_x_adv - batch_x).abs().max()
                print("adv_eps:", adv_eps.item())
                """
                # for ts_ranged
                _, _, _, benign_error_sequence = self.compute_errors(0, benign_out.size(1), benign_out, batch_y)
                _, _, _, adv_error_sequence = self.compute_errors(0, adv_out.size(1), adv_out, batch_y)
                print("benign_error_sequence:", benign_error_sequence.cpu().detach().numpy())
                print("adv_error_sequence:", adv_error_sequence.cpu().detach().numpy())
                
                if total_cnt == batch_cnt and self.args.targeted == True:
                    np.savez(f"../experiments/tmp_output/udacity_{prefix}_{params['epsilon'][0][0][0]:.02f}_visualization_data.npz", batch_x=batch_x.cpu().detach().numpy(), batch_y=batch_y.cpu().detach().numpy(),
                                                                             benign_out=benign_out.cpu().detach().numpy(),
                                                                             batch_x_adv=batch_x_adv.cpu().detach().numpy(),
                                                                             target_label=target_label.cpu().detach().numpy(),
                                                                             adv_out=adv_out.cpu().detach().numpy())
        
        print("Benign loss: {:.02f}, Adv loss: {:.02f}".format(np.sum(benign_loss_list)/total_cnt, np.sum(adv_loss_list)/total_cnt))
        print("MSE, {:.04f}, mean_max_ad, {:.04f}, mean_mean_ad, {:.04f}, elapsed_time, {:.04f}".format(np.sum(mse_error_list)/total_cnt,
                                                                                 np.sum(max_ad_list)/total_cnt, 
                                                                                 np.sum(mean_ad_list)/total_cnt, elapsed_time))
        if target_label != None:
            print("Target comply error {:.04f}".format(np.sum(target_accordance_error_list)/total_cnt))                                                                                 
        return mse_error, mean_max_ad, mean_mean_ad

    def set_predictive_params(self, default_param):
        default_param["predictive_steps"] = self.args.k
        default_param["is_ts_ranged"] = self.args.ts_ranged
        default_param["ts_neg_gamma"] = self.args.ts_neg_gamma

        default_param["pred_func"] = self.get_pred_func(self.args, self.predictor[0], self.predictor[1])
        default_param["get_loss_func"] = self.get_loss_func_mean if self.args.max_attack == False else self.get_loss_func_max

        return default_param