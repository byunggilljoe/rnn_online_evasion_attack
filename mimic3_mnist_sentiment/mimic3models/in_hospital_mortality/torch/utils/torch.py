import torch
import numpy as np
import math

class Normalizer(object):
    @staticmethod
    def linf_normalize(perturb):
        return torch.sign(perturb)

    @staticmethod
    def l1_normalize(perturb):
        #return perturb/torch.sum(np.abs(np.reshape(perturb, (perturb.shape[0], -1))), axis=1, keepdims=True)
        l1n = torch.max(Normalizer.l1_norm(perturb), torch.ones(perturb.size(0), device="cuda:0")*0.001)
        l1normalized = perturb/l1n.view(-1, 1, 1, 1)
        return l1normalized 

    @staticmethod
    def l0_normalize(perturb):
        # we use l1 normalizer, because max l0 normalization would make it impossible to use perturbations from other pixels.
        #return Bounder.l0_bound(perturb, 1)
        return Normalizer.l1_normalize(perturb)
        #return Normalizer.l2_normalize(perturb)

    @staticmethod
    def l0_norm(perturb):
        return (perturb.view(perturb.size(0), -1).abs() != 0).sum(dim=1).float()

    @staticmethod
    def l1_norm(perturb):
        return perturb.view(perturb.size(0), -1).abs().sum(dim=1)

    @staticmethod
    def l2_norm(perturb):
        norm = torch.norm(perturb.view(perturb.size(0), -1), 2, dim=1)
        return norm

    @staticmethod
    def l2_normalize(perturb):
        l2n = torch.max(Normalizer.l2_norm(perturb), torch.ones(perturb.size(0), device="cuda:0")*0.001)
        perturb = perturb / l2n.view(-1, *([1]*(len(perturb.size()) - 1)))
        return perturb

    @staticmethod 
    def normalize(perturb, p):
        normalizers= {"l0":Normalizer.l0_normalize,"l1":Normalizer.l1_normalize,"l2":Normalizer.l2_normalize, "linf":Normalizer.linf_normalize}
        return normalizers[p](perturb)


class Bounder(object):
    @staticmethod
    def linf_bound(perturb, epsilon):
        return torch.clamp(perturb, min=-epsilon, max=epsilon)

    @staticmethod
    def l0_bound(perturb, epsilon):
        reshaped_perturb = perturb.view(perturb.size(0), -1)
        sorted_perturb = torch.sort(reshaped_perturb, dim=1)[0]
        k = int(math.ceil(epsilon))
        thresholds =sorted_perturb[:, -k]
        target_size = (perturb.size(0), ) + (1,)*(len(perturb.size()) - len(thresholds.size()))
        mask = perturb >= thresholds.view(target_size)
        return perturb*mask
    
    @staticmethod
    def l0_bound_sparse(perturb, epsilon, x_victim):
        #print(epsilon)
        x_victim_permuted = x_victim.permute(0, 2, 3, 1).cpu().detach().numpy()
        perturb_permuted_numpy = perturb.permute(0, 2, 3, 1).cpu().detach().numpy()
        l0_bounded_perturb = torch.from_numpy(Bounder.project_L0_box(perturb_permuted_numpy,\
                int(epsilon), -x_victim_permuted, 1-x_victim_permuted)).float()
        l0_bounded_perturb_permuted =  l0_bounded_perturb.permute(0, 3, 1, 2)
        return l0_bounded_perturb_permuted.cuda()

    @staticmethod
    def project_L0_box(y, k, lb, ub):
        ''' projection of the batch y to a batch x such that:
            - each image of the batch x has at most k pixels with non-zero channels
            - lb <= x <= ub '''
        
        x = np.copy(y)
        p1 = np.sum(x**2, axis=-1)
        p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
        p2 = np.sum(p2**2, axis=-1)
        p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
        x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
        x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
        return x

    @staticmethod
    def l1_bound(perturb, epsilon):
        bounded_s = []
        for i in range(perturb.shape[0]):
            bs = perturb[i].cpu().detach().numpy()
            abs_bs = np.abs(bs)
            if np.sum(abs_bs) > epsilon:
                old_shape = bs.shape
                bs = Bounder.projection_simplex_sort(np.reshape(abs_bs, (abs_bs.size, )), epsilon)
                bs = np.reshape(bs, old_shape)
            bounded_s.append(bs)
        return torch.from_numpy(np.array(bounded_s)).cuda()

    @staticmethod
    def projection_simplex_sort(v, z=1):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    @staticmethod 
    def l2_bound(perturb, epsilon):
        l2_norm = Normalizer.l2_norm(perturb) 
        multiplier = 1.0/torch.max(l2_norm/epsilon, torch.ones_like(l2_norm))
        return perturb * multiplier.view(-1, *([1]*(len(perturb.size()) - 1)))

    @staticmethod 
    def bound(perturb, epsilon, p):
        bounders = {"l0":Bounder.l0_bound,"l1":Bounder.l1_bound,"l2":Bounder.l2_bound, "linf":Bounder.linf_bound}
        return bounders[p](perturb, epsilon)


