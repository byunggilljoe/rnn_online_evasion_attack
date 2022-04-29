import sys
sys.path.append("../")
import torch
import data, models
import attacks
import time

class AttackExperiment(object):
    # Dataset dependent implementations
    def get_targeted_label(self, BATCH, true_label, t_start, NUM_WAVE):
        raise NotImplementedError()

    def get_loss_func(self):
        raise NotImplementedError()
    
    def get_loss_func_mean(self):
        raise NotImplementedError()
    
    def get_loss_func_max(self):
        raise NotImplementedError()

    def get_preparation(self):
        raise NotImplementedError()

    def get_pred_func(self, args, predictor):
        raise NotImplementedError()

    def test_and_finish(self):
        raise NotImplementedError()
    


    # Common implementations
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.attack_dict={"clairvoyant": {"set_params":self.set_clairvoyant_params, "attack":attacks.clairvoyant_each_step_attack_with_range___p},
                "predictive":{"set_params":self.set_predictive_params, "attack":attacks.predictive_each_step_attack_with_range},
                "greedy":{"set_params":self.set_greedy_params, "attack":attacks.greedy_each_step_attack_with_range},
                "IID_predictive":{"set_params":self.set_IID_predictive_params, "attack":attacks.predictive_each_step_attack_with_range},
                "uniform_predictive":{"set_params":self.set_uniform_predictive_params, "attack":attacks.predictive_each_step_attack_with_range}}
        
        self.MAX_VALUE = None
        self.MIN_VALUE = None
        self.criterion = None

    def load_data_and_model(self):
        self.train_loader, self.test_loader = data.load_data(self.args.dataset, self.args.batch_size)
        self.model, self.predictor = models.load_model(self.args.dataset, self.args.trial, self.args.batch_size)
        if self.args.transfer_attack == True:
            self.eval_model, _ = models.load_model(self.args.dataset, (self.args.trial+1)%3, self.args.batch_size)
        else:
            self.eval_model = self.model

    def execute(self):
        params_list = []
        batch_x_adv_list = []
        t_start = time.time()
        for i in range(self.args.NUM_TEST_BATCH):
            batch_x, batch_y, criterion, default_params = self.get_preparation(self.args, self.model, self.train_loader if self.args.dataset == "udacity" or self.args.save_realtime_training_data else self.test_loader , i)
            # batch_x, batch_y, criterion, default_params = self.get_preparation(self.args, self.model, self.test_loader , i)
            attack_item = self.attack_dict[self.args.attack]
            params = attack_item["set_params"](default_params)
            batch_x_adv = attack_item["attack"](**params)
            params_list.append(params)
            batch_x_adv_list.append(batch_x_adv.detach())
        t_end = time.time()

        print(f"=== {self.args.attack} ===")
        self.test_and_finish(params_list, batch_x_adv_list, criterion, self.eval_model, self.args.attack, t_end - t_start) # self.model -> self.eval_model (for transfer experiment)
        
        if self.args.save_realtime_training_data == True:
            x_data = torch.cat([p["current_data"].cpu() for p in params_list], axis=0).detach().numpy()
            x_adv_data = torch.cat(batch_x_adv_list, axis=0).cpu().detach().numpy()
            import numpy as np
            np.save("../experiments/tmp_output/{}_{}_{}_realtime_perturb_data.npy".format(self.args.dataset, self.args.attack, self.args.eps), {"x_data":x_data, "x_adv_data":x_adv_data})
        #self.attack_dict[self.args.attack]()
    
    def get_preparation(self, args, steering_net, test_generator, start_index):
        en = enumerate(test_generator)
        _, (batch_x, batch_y) = next(en)
        for i in range(start_index):
            _, (batch_x, batch_y) = next(en)

        batch_x = batch_x.float().cuda()
        batch_y = batch_y.cuda()

        criterion = self.criterion

        ONES_LIKE_FRAME = torch.ones_like(batch_x[0][0]).cuda()
        EPSILON = args.eps*ONES_LIKE_FRAME
        STEP_SIZE = args.step_size*ONES_LIKE_FRAME
        ITERS = args.iters
        MAX_VALUE = ONES_LIKE_FRAME*self.MAX_VALUE
        MIN_VALUE = ONES_LIKE_FRAME*self.MIN_VALUE 

        NUM_PREDICTION = args.k
        target_label = torch.tensor(self.get_targeted_label(BATCH=batch_x.size(0), true_label=batch_y, t_start=args.t_start, NUM_WAVE=args.NUM_WAVE), device="cuda") if args.targeted else None
        arg_dict = {"model":steering_net,
                    "current_data":batch_x,
                    "true_label":batch_y,
                    "epsilon":EPSILON,
                    "step_size":STEP_SIZE,
                    "max_iters":ITERS,
                    "min_value":MIN_VALUE,
                    "max_value":MAX_VALUE,
                    # "loss_func":get_loss_func(num_loss_step=NUM_PREDICTION+1, max_attack=args.max_attack),
                    # "pred_func":get_pred_func(args, encoder, frame_predictor),
                    # "predictive_steps":NUM_PREDICTION,
                    #"t_start":args.t_start,
                    "t_start":0,
                    "t_end":args.t_end,
                    "target_label":target_label}

        return batch_x, batch_y, criterion, arg_dict
        
    
    def set_greedy_params(self, default_param):
        default_param["loss_func"] = self.get_loss_func(num_loss_step = 1)
        return default_param

    def set_clairvoyant_params(self, default_param):
        default_param["predictive_steps"] = self.args.k
        default_param["pred_func"] = None
        default_param["get_loss_func"] = self.get_loss_func_mean if self.args.max_attack == False else self.get_loss_func_max
        return default_param

    def set_predictive_params(self, default_param):
        default_param["predictive_steps"] = self.args.k
        default_param["pred_func"] = self.get_pred_func(self.args, self.predictor)
        default_param["get_loss_func"] = self.get_loss_func_mean if self.args.max_attack == False else self.get_loss_func_max

        default_param["noise_alpha"] = self.args.noise_alpha
        
        default_param["is_ts_ranged"] = self.args.ts_ranged
        default_param["ts_neg_gamma"] = self.args.ts_neg_gamma

        return default_param

    def set_IID_predictive_params(self, default_param):
        return self.set_custom_predictive_params(default_param, attacks.get_IID_pred_func)

    def set_uniform_predictive_params(self, default_param):
        return self.set_custom_predictive_params(default_param, attacks.get_uniform_pred_func)

    def set_custom_predictive_params(self, default_param, gpm):
        default_param["predictive_steps"] = self.args.k
        self.args.max_value = default_param["max_value"]
        self.args.min_value = default_param["min_value"]
        default_param["pred_func"] = gpm(self.args, None)
        default_param["get_loss_func"] = self.get_loss_func_mean if self.args.max_attack == False else self.get_loss_func_max
        return default_param

