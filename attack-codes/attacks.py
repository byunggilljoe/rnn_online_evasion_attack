import torch
import numpy as np
DO_EARLY_LOOKAHEAD_ATTACK = True
# Given data t=0 t=i, perturb data of t=i to lower attack loss.
def greedy_one_step_attack(model, current_data, true_label, epsilon, step_size, max_iters, min_value, max_value,\
                           loss_func, num_perturb_step=1, idc_perturb=None, num_sample=1, temporal_smoothness=None, clean_data=None, target_label=None):
    # Way to specify target output should be one, not two.
    assert((num_perturb_step == None and idc_perturb != None) or (num_perturb_step != None and idc_perturb == None))
    if temporal_smoothness != None:
        assert(clean_data != None)

    x_adv = current_data.clone().detach()
    SIZE = x_adv.size()
    DELTA_SIZE = list(SIZE)
    DELTA_SIZE[0] = DELTA_SIZE[0]//num_sample
    delta = torch.zeros(DELTA_SIZE).cuda()
    mask = torch.zeros(DELTA_SIZE).cuda()
    if num_perturb_step != None:
        mask[:, -num_perturb_step:, :] = 1.0
    else:
        for idx in idc_perturb:
            mask[:, idx, :] = 1.0

    model.train()
    delta.requires_grad = True
    repeat_size = [num_sample] + ([1]*(len(SIZE) - 1))
    x_adv = x_adv + delta.repeat(repeat_size)
    for i in range(max_iters):
        model.zero_grad()
        out = model(x_adv)
        loss = loss_func(out, true_label, target_label = target_label)
        loss.backward(retain_graph=True)
        # loss.backward()

        grad = delta.grad.data
        delta = torch.min(torch.max(delta + mask*torch.sign(grad)*step_size, -epsilon), epsilon)
        
        if temporal_smoothness != None:
            all_delta = (1.0-mask)*(x_adv - clean_data) + mask*delta
            t_start = None
            for t in range(1, all_delta.size(1)):
                prev_time = t-1
                if torch.pow(all_delta[:, prev_time,:], 2).mean() < 1e-8: 
                    continue
                elif t_start == None:
                    t_start = prev_time
                unsqueezed_epsilon = epsilon.unsqueeze(0).repeat(50, 1)
                d_delta_t = torch.min(torch.max(all_delta[:, t, :] - all_delta[:, prev_time, :], -unsqueezed_epsilon*temporal_smoothness), 
                                    unsqueezed_epsilon*temporal_smoothness)
                all_delta[:, t, :] = d_delta_t + all_delta[:, prev_time, :]
            delta = mask*all_delta
        delta = delta.detach()
        delta.requires_grad = True
        delta_repeated = delta.repeat(repeat_size)
        x_adv = torch.min(torch.max(current_data + delta_repeated, min_value), max_value)
    return x_adv  


def greedy_each_step_attack_with_range(model, current_data, true_label, epsilon, step_size, max_iters, min_value, max_value,
                                        loss_func, t_start=0, t_end=48, temporal_smoothness=None, target_label=None):
    empty_size = list(current_data.size())
    empty_size[1] = 0
    adv_data_t = torch.empty(empty_size).cuda()
    for t in range(current_data.size(1)):
        data_t = torch.cat([adv_data_t, current_data[:, t:t+1, :]], dim=1)
        if t >= t_start and t < t_end:
            current_true_label = None
            if len(true_label.size()) > 1:
                # Targeted attack, Udacity
                current_true_label = true_label[:, data_t.size(1) - 1:data_t.size(1)]
            else:
                current_true_label = true_label

            current_target_label = None
            if target_label is not None:
                current_target_label = target_label[:, data_t.size(1) - 1]

            adv_data_t = greedy_one_step_attack(model, data_t, current_true_label.cuda(), epsilon=epsilon, step_size=step_size,
                                                max_iters=max_iters, min_value=min_value, max_value=max_value, num_perturb_step=1,
                                                loss_func=loss_func, temporal_smoothness=temporal_smoothness, clean_data=current_data[:, :t+1, :], target_label = current_target_label)
        elif t < t_start:
            adv_data_t = current_data[:, :t+1, :]
        elif t >= t_end:
            adv_data_t = torch.cat([adv_data_t, current_data[:, t:, :]], dim=1)
            return adv_data_t
    adv_data = torch.cat([adv_data_t, current_data[:, t_end:, :]], dim=1)
    return adv_data


def get_predicted_data(current_data, pred_func, predictive_steps):
    emtpy_size = list(current_data.size())
    emtpy_size[1] = 0
    predicted_data_only = torch.empty(emtpy_size).cuda()
    if predictive_steps == 0:
        return torch.cat([current_data, predicted_data_only], dim=1), predicted_data_only
    
    state = None
    for i in range(predictive_steps):
        next_step, state = pred_func(current_data, predicted_data_only, state)
        predicted_data_only = torch.cat([predicted_data_only, next_step.detach()], dim=1)

    return torch.cat([current_data, predicted_data_only], dim=1), predicted_data_only  


def clairvoyant_each_step_attack_with_range(model, current_data, true_label, epsilon, step_size,
                                max_iters, min_value, max_value, loss_func, t_start=0, t_end=48, temporal_smoothness=None, target_label=None): # last attacked index =  t_end - 1 
    emtpy_size = list(current_data.size())
    emtpy_size[1] = 0
    adv_data = torch.empty(emtpy_size).cuda()
    for t in range(t_end):
        adv_data = torch.cat([adv_data, current_data[:, t:t_end, :]], dim=1) # fetch current input to start perturbation
        perturb_steps = t_end - t

        if len(true_label.size()) > 1:
            # Targeted attack, Udacity
            current_true_label_repeated = true_label[:, :adv_data.size(1)]
        else:
            current_true_label_repeated = true_label

        current_target_label = None
        if target_label is not None:
            current_target_label = target_label[:, :current_true_label_repeated.size(1)]


        adv_data_t = greedy_one_step_attack(model, adv_data, current_true_label_repeated.cuda(),
                                            epsilon=epsilon, step_size=step_size, max_iters=max_iters,
                                            min_value=min_value, max_value=max_value, num_perturb_step=perturb_steps,
                                            loss_func=loss_func, temporal_smoothness=temporal_smoothness, clean_data=current_data, target_label=current_target_label) #num_step = 1 (current_step) + predictive_steps
        adv_data = adv_data_t[:, :t+1, :]
    adv_data = torch.cat([adv_data, current_data[:, t_end:, :]], dim=1)
    return adv_data


# At time t
# Predict k future steps
# Perturb t ~ t + k steps
# Apply perturbation at t
# Given range t_start <= t < t_end
def predictive_each_step_attack_with_range(model, pred_func, current_data, true_label, epsilon, step_size, max_iters, min_value, max_value,
                                           predictive_steps=1, get_loss_func=None, t_start=0, t_end=48, num_sample=1, temporal_smoothness=None, is_ts_ranged=False, ts_neg_gamma=0, target_label=None, noise_alpha=0.0):
    assert(t_start==0)
    assert(get_loss_func != None)
    assert(num_sample == 1)
    assert(temporal_smoothness == None)
    DATA_SIZE = list(current_data.size())
    emtpy_size = DATA_SIZE[:]
    emtpy_size[1] = 0
    
    repeat_size = DATA_SIZE[:]
    repeat_size = [1 for _ in range(len(DATA_SIZE))]
    repeat_size[0] = num_sample

    adv_data = torch.empty(emtpy_size).cuda()
    
    label_repeat_shape = [1 for _ in range(len(list(true_label.size())))]
    label_repeat_shape[0] = num_sample

    true_label_repeated = true_label.repeat(label_repeat_shape)

    #true_label_repeated = true_label.repeat(num_sample)

    # if true_label_repeated.size(0) == 1:
    #     # One label for 1 sequential input.
    #     true_label_repeated = true_label_repeated.squeeze(0)

    def get_sampled_data(data_t, predictive_steps):
        sampled_list = []
        for i in range(num_sample):
            (_, predicted_data_t_only) = get_predicted_data(data_t, pred_func, predictive_steps=predictive_steps)
            sampled_list.append(predicted_data_t_only)
        if False and predictive_steps > 0:
            for i in range(1, num_sample):
                mse_i = torch.nn.functional.mse_loss(sampled_list[0], sampled_list[i])
                print("mse_{}/{}/{}:".format(i, len(sampled_list), predictive_steps), mse_i.item())
        
        return torch.cat(sampled_list, dim=0)

    neg_range_weight= ts_neg_gamma
    for t in range(current_data.size(1)):
        data_t = current_data[:, :t+1, :] # fetch clean input to predict future value
        adv_data = torch.cat([adv_data, current_data[:, t:t+1, :]], dim=1)
        T_START_OFFSET = 0 if DO_EARLY_LOOKAHEAD_ATTACK else -predictive_steps
        if t + predictive_steps + T_START_OFFSET >= t_start and t < t_end:
            if t + predictive_steps < t_end:
                #(_, predicted_data_t_only) = get_predicted_data(data_t, pred_func, predictive_steps=predictive_steps)
                predicted_data_t_only = get_sampled_data(data_t, predictive_steps=predictive_steps)
                
                alpha = 1.0
                noise = torch.rand(predicted_data_t_only.size(), device="cuda")*alpha #[0, 1)
                noise = noise*(max_value-min_value) + min_value
                
                predicted_data_t_only =  predicted_data_t_only*(1.0-noise_alpha) + noise*noise_alpha

                # Test with true data
                #predicted_data_t_only = current_data[:, t+1:t+predictive_steps+1, :]
                assert(predicted_data_t_only.size(1) == predictive_steps)
                num_attack_step = predictive_steps + 1
                num_loss_step = t + predictive_steps + 1 - t_start
                adv_pred_data = torch.cat([adv_data.repeat(*repeat_size), predicted_data_t_only], dim=1)
                adv_pred_data = adv_pred_data[:, :current_data.size(1)]
                
                
                if len(true_label_repeated.size()) > 1:
                    # Targeted attack, Udacity
                    current_true_label_repeated = true_label_repeated[:, :adv_pred_data.size(1)]
                else:
                    current_true_label_repeated = true_label_repeated

                current_target_label = None
                if target_label is not None:
                    current_target_label = target_label[:, :adv_pred_data.size(1)]

                # Ranged attack. Note that we only consider loss before t_start.
                gamma_array = 1.0
                if is_ts_ranged == True:
                    # # Attack should be restrained before t_start.
                    # # Thus, we need all loss steps we attack.
                    # num_loss_step_old = num_loss_step
                    # num_loss_step = num_attack_step 
                    # gamma_array = neg_range_weight*np.ones(num_attack_step)
                    # #gamma_array = -00.0*np.ones(num_attack_step)
                    # gamma_array[-num_loss_step_old:] = 1.0
                    # #print("gamma_array:", gamma_array)
                    gamma_array = neg_range_weight*np.ones(adv_pred_data.size(1))
                    gamma_array[int(gamma_array.shape[0]/4.0*3):] = 1.0
                adv_data_t = greedy_one_step_attack(model, adv_pred_data, current_true_label_repeated.cuda(),
                                                    epsilon=epsilon, step_size=step_size, max_iters=max_iters,
                                                    min_value=min_value, max_value=max_value, num_perturb_step=num_attack_step,
                                                    loss_func=get_loss_func(num_loss_step=num_loss_step, gamma=gamma_array), num_sample=num_sample,
                                                    temporal_smoothness=temporal_smoothness, clean_data=current_data[:,:t+num_attack_step,:], target_label=current_target_label) #num_step = 1 (current_step) + predictive_steps
                adv_data = adv_data_t[:adv_data_t.size(0)//num_sample, :t+1, :]
            elif t + predictive_steps >= t_end:
                num_predictive_step = t_end - t - 1
                #(_, predicted_data_t_only) = get_predicted_data(data_t, pred_func, predictive_steps=num_predictive_step)
                predicted_data_t_only = get_sampled_data(data_t, predictive_steps=num_predictive_step)
                # Test with true data
                #predicted_data_t_only = current_data[:, t+1:t+num_predictive_step+1, :]
                assert(predicted_data_t_only.size(1) == num_predictive_step)
                num_attack_step = t_end - t
                num_loss_step = t_end - t_start
                adv_pred_data = torch.cat([adv_data.repeat(*repeat_size), predicted_data_t_only], dim=1)


                if len(true_label_repeated.size()) > 1:
                    # Targeted attack, Udacity
                    current_true_label_repeated = true_label_repeated[:, :adv_pred_data.size(1)]
                else:
                    current_true_label_repeated = true_label_repeated

                current_target_label = None
                if target_label != None:
                    current_target_label = target_label[:, :adv_pred_data.size(1)]



                # Ranged attack. Note that we only consider loss before t_start.
                gamma_array = 1.0
                if is_ts_ranged == True:
                    # # Attack should be restrained before t_start.
                    # # Thus, we need all loss steps we attack.
                    # num_loss_step_old = num_loss_step
                    # num_loss_step = num_attack_step 
                    # gamma_array =neg_range_weight*np.ones(num_attack_step)
                    # #gamma_array =-00.0*np.ones(num_attack_step)
                    # gamma_array[-num_loss_step_old:] = 1.0
                    # #print("gamma_array:", gamma_array)

                    gamma_array = neg_range_weight*np.ones(adv_pred_data.size(1))
                    gamma_array[int(gamma_array.shape[0]/4.0*3):] = 1.0
                    
                adv_data_t = greedy_one_step_attack(model, adv_pred_data, current_true_label_repeated.cuda(),
                                                    epsilon=epsilon, step_size=step_size, max_iters=max_iters,
                                                    min_value=min_value, max_value=max_value, num_perturb_step=num_attack_step,
                                                    loss_func=get_loss_func(num_loss_step=num_loss_step, gamma=gamma_array), num_sample=num_sample,
                                                    temporal_smoothness=temporal_smoothness, clean_data=current_data[:,:t+num_attack_step,:], target_label=current_target_label) #num_step = 1 (current_step) + predictive_steps
                adv_data = adv_data_t[:adv_data_t.size(0)//num_sample, :t+1, :]
        elif t >= t_end:
            return torch.cat([adv_data, current_data[:, t+1:, :]], dim=1)
        else:
            adv_data = current_data[:, :t+1, :]
        
            
            
    return adv_data


def clairvoyant_each_step_attack_with_range___p(model, pred_func, current_data, true_label, epsilon, step_size, max_iters, min_value, max_value,
                                           predictive_steps=1, get_loss_func=None, t_start=0, t_end=48, temporal_smoothness=None, target_label=None):
    assert(get_loss_func != None)
    emtpy_size = list(current_data.size())
    emtpy_size[1] = 0
    adv_data = torch.empty(emtpy_size).cuda()
    for t in range(current_data.size(1)):
        data_t = current_data[:, :t+1, :] # fetch clean input to predict future value
        adv_data = torch.cat([adv_data, current_data[:, t:t+1, :]], dim=1)
        T_START_OFFSET = 0 if DO_EARLY_LOOKAHEAD_ATTACK else -predictive_steps
        if t + predictive_steps + T_START_OFFSET >= t_start and t < t_end:
            if t + predictive_steps < t_end:
                #assert(False)
                #(_, predicted_data_t_only) = get_predicted_data(data_t, pred_func, predictive_steps=predictive_steps)
                # print("adv_data.size():", adv_data.size())
                predicted_data_t_only = current_data[:, t+1:t+predictive_steps+1, :]###########
                # print("predicted_data_t_only.size():", predicted_data_t_only.size())
                num_attack_step = predictive_steps + 1
                num_loss_step = t + predictive_steps + 1 - t_start
                adv_pred_data = torch.cat([adv_data, predicted_data_t_only], dim=1)
                adv_pred_data = adv_pred_data[:, :current_data.size(1)]

                if len(true_label.size()) > 1:
                    # Targeted attack, Udacity
                    current_true_label = true_label[:, :adv_pred_data.size(1)]
                else:
                    current_true_label = true_label

                current_target_label = None
                if target_label is not None:
                    current_target_label = target_label[:, :adv_pred_data.size(1)]

                adv_data_t = greedy_one_step_attack(model, adv_pred_data, current_true_label.cuda(),
                                                    epsilon=epsilon, step_size=step_size, max_iters=max_iters,
                                                    min_value=min_value, max_value=max_value, num_perturb_step=num_attack_step,
                                                    loss_func=get_loss_func(num_loss_step=num_loss_step),
                                                    temporal_smoothness=temporal_smoothness, clean_data=current_data[:, :adv_pred_data.size(1), :], target_label=current_target_label) #num_step = 1 (current_step) + predictive_steps
                adv_data = adv_data_t[:, :t+1, :]
            elif t + predictive_steps >= t_end:
                # print("t:", t, "t + predictive_steps:", t + predictive_steps)
                num_predictive_step = t_end - t - 1
                #(_, predicted_data_t_only) = get_predicted_data(data_t, pred_func, predictive_steps=num_predictive_step)
                predicted_data_t_only = current_data[:, t+1:t+num_predictive_step+1, :]###########
                num_attack_step = t_end - t
                num_loss_step = t_end - t_start
                adv_pred_data = torch.cat([adv_data, predicted_data_t_only], dim=1)
                adv_pred_data = adv_pred_data[:, :current_data.size(1)]
                # print("----> attack size:", predicted_data_t_only.size())

                if len(true_label.size()) > 1:
                    # Targeted attack, Udacity
                    current_true_label = true_label[:, :adv_pred_data.size(1)]
                else:
                    current_true_label = true_label

                current_target_label = None
                if target_label is not None:
                    current_target_label = target_label[:, :adv_pred_data.size(1)]

                adv_data_t = greedy_one_step_attack(model, adv_pred_data, true_label.cuda(),
                                                    epsilon=epsilon, step_size=step_size, max_iters=max_iters,
                                                    min_value=min_value, max_value=max_value, num_perturb_step=num_attack_step,
                                                    loss_func=get_loss_func(num_loss_step=num_loss_step),
                                                    temporal_smoothness=temporal_smoothness, clean_data=current_data[:, :adv_data.size(1)+predicted_data_t_only.size(1),:], target_label=current_target_label) #num_step = 1 (current_step) + predictive_steps
                adv_data = adv_data_t[:, :t+1, :]
        elif t >= t_end:
            return torch.cat([adv_data, current_data[:, t+1:, :]], dim=1)
        else:
            adv_data = current_data[:, :t+1, :]
            
            
    return adv_data


def get_IID_pred_func(args, predictor):

    def pred_func(current_data, predicted_data_only, state):
        # current_data: BATCH x TIME x CHANNEL x HEIGHT x WIDTH
        # input: prev frame
        # output: next frame and memo for next prediction
        #colcat_data = torch.cat([current_data, predicted_data_only], dim=1)
        rand_index = np.random.randint(current_data.size(1))
        pred_step = current_data[:, rand_index, :].unsqueeze(1)
        # pred_step = pred_step[torch.randperm(pred_step.size()[0])]
        return pred_step, state

    return pred_func

def get_uniform_pred_func(args, predictor):
    def pred_func(current_data, predicted_data_only, state):
        SIZE = list(current_data.size())
        SIZE[1] = 1
        alpha = 0.01 if args.dataset != "udacity" else args.eps*0.01 #0.001
        rand = torch.rand(SIZE, device="cuda")*alpha #[0, 1)
        rand = rand*(args.max_value-args.min_value) + args.min_value
        return rand, state
    return pred_func

def get_permuted_pred_func(args, predictor):

    def pred_func(current_data, predicted_data_only, state):
        # current_data: BATCH x TIME x CHANNEL x HEIGHT x WIDTH
        # input: prev frame
        # output: next frame and memo for next prediction
        #colcat_data = torch.cat([current_data, predicted_data_only], dim=1)
        rand_index = np.random.randint(current_data.size(1))
        rand_permute_idc = torch.randperm(current_data.size(2))
        pred_step = current_data[:, rand_index, rand_permute_idc].unsqueeze(1)
        return pred_step, state

    return pred_func

def get_bottom_k_1_mean(values):
    T = values.size(1)
    return -torch.topk(-values, dim=1, k=T-1)[0].mean()