#import tensorflow as tf 
import numpy as np
import math

class Normalizer(object):
    @staticmethod
    def l1_normalize(perturb):
        #l1n = np.sum(np.abs(np.reshape(perturb, (perturb.shape[0], -1))), axis=1)
        l1n = Normalizer.l1_norm(perturb)
        return perturb/np.maximum(np.reshape(l1n, (-1,) + (1,)*(len(perturb.shape) - 1)), 1e-10)
        #return perturb/np.maximum(np.reshape(l1n, (-1,) + (1,)*(len(perturb.shape) - 1)), 0.001)

    @staticmethod
    def l0_normalize(perturb):
        # we use l1 normalizer, because max l0 normalization would make it impossible to use perturbations from other pixels.
        return Normalizer.l1_normalize(perturb)

    @staticmethod
    def linf_normalize(perturb):
        return np.sign(perturb)

    @staticmethod
    def l0_norm(perturb):
        return (np.abs(np.reshape(perturb, (perturb.shape[0], -1))) != 0).sum(axis=1).float()

    @staticmethod
    def l1_norm(perturb):
        return np.abs(np.reshape(perturb, (perturb.shape[0], -1))).sum(axis=1)

    @staticmethod
    def l2_norm(perturb):
        size = np.prod(perturb.shape[1:]).astype(np.int64)
        norm = np.linalg.norm(np.reshape(perturb, [-1, size]), 2, axis=1)
        return norm

    @staticmethod
    def l2_normalize(perturb):
        l2n = np.maximum(Normalizer.l2_norm(perturb), 0.0001)
        perturb = perturb / np.reshape(l2n, (-1, ) + (1,)*(len(perturb.shape) - 1))
        return perturb

    @staticmethod 
    def normalize(perturb, p):
        normalizers= {"l0":Normalizer.l0_normalize,"l1":Normalizer.l1_normalize,"l2":Normalizer.l2_normalize, "linf":Normalizer.linf_normalize}
        return normalizers[p](perturb)



class Bounder(object):
    @staticmethod
    def linf_bound(perturb, epsilon):
        return np.clip(perturb, a_min=-epsilon, a_max=epsilon)

    @staticmethod
    def l0_bound(perturb, epsilon):
        reshaped_perturb = np.reshape(perturb, (perturb.shape[0], -1))
        sorted_perturb = np.sort(reshaped_perturb, axis=1)
        k = int(math.ceil(epsilon))
        thresholds =sorted_perturb[:, -k]
        mask = perturb >= np.reshape(thresholds, (perturb.shape[0], ) + (1,)*(len(perturb.shape) - len(thresholds.shape)))
        return perturb*mask

    @staticmethod
    def l1_bound(perturb, epsilon):
        bounded_s = []
        for i in range(perturb.shape[0]):
            bs = perturb[i]
            abs_bs = np.abs(bs)
            if np.sum(abs_bs) > epsilon:
                old_shape = bs.shape
                bs = Bounder.projection_simplex_sort(np.reshape(abs_bs, (abs_bs.size, )), epsilon)
                bs = np.reshape(bs, old_shape)
            bounded_s.append(bs)
        return np.array(bounded_s)

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
        l2_norm = np.expand_dims(Normalizer.l2_norm(perturb), 1)
        #assert((epsilon != 0).all())
        multiplier = 1.0/np.maximum(l2_norm/epsilon, np.ones_like(l2_norm))
        return perturb * np.reshape(multiplier, (-1, ) + (1,)*(len(perturb.shape) - 1))

    @staticmethod 
    def bound(perturb, epsilon, p):
        bounders = {"l0":Bounder.l0_bound,"l1":Bounder.l1_bound,"l2":Bounder.l2_bound, "linf":Bounder.linf_bound}
        return bounders[p](perturb, epsilon)
    


#def euclidean_proj_l1ball(v, s=1):
#    """ Compute the Euclidean projection on a L1-ball
#    Solves the optimisation problem (using the algorithm from [1]):
#        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
#    Parameters
#    ----------
#    v: (n,) numpy array,
#       n-dimensional vector to project
#    s: int, optional, default: 1,
#       radius of the L1-ball
#    Returns
#    -------
#    w: (n,) numpy array,
#       Euclidean projection of v on the L1-ball of radius s
#    Notes
#    -----
#    Solves the problem by a reduction to the positive simplex case
#    See also
#    --------
#    euclidean_proj_simplex
#    """
#    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
#    n, = v.shape  # will raise ValueError if v is not 1-D
#    # compute the vector of absolute values
#    u = np.abs(v)
#    # check if v is already a solution
#    if u.sum() <= s:
#        # L1-norm is <= s
#        return v
#    # v is not already a solution: optimum lies on the boundary (norm == s)
#    # project *u* on the simplex
#    w = euclidean_proj_simplex(u, s=s)
#    # compute the solution to the original problem on v
#    w *= np.sign(v)
#    return w
#
#def euclidean_proj_simplex(v, s=1):
#    """ Compute the Euclidean projection on a positive simplex
#    Solves the optimisation problem (using the algorithm from [1]):
#        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
#    Parameters
#    ----------
#    v: (n,) numpy array,
#       n-dimensional vector to project
#    s: int, optional, default: 1,
#       radius of the simplex
#    Returns
#    -------
#    w: (n,) numpy array,
#       Euclidean projection of v on the simplex
#    Notes
#    -----
#    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
#    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
#    However, this implementation still easily scales to millions of dimensions.
#    References
#    ----------
#    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
#        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
#        International Conference on Machine Learning (ICML 2008)
#        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
#    """
#    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
#    n, = v.shape  # will raise ValueError if v is not 1-D
#    # check if we are already on the simplex
#    if v.sum() == s and np.alltrue(v >= 0):
#        # best projection: itself!
#        return v
#    # get the array of cumulative sums of a sorted (decreasing) copy of v
#    u = np.sort(v)[::-1]
#    cssv = np.cumsum(u)
#    # get the number of > 0 components of the optimal solution
#    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
#    print("e",np.nonzero(u * np.arange(1, n+1) > (cssv - s)))
#    # compute the Lagrange multiplier associated to the simplex constraint
#    theta = float(cssv[rho] - s) / rho
#    # compute the projection by thresholding v using theta
#    w = (v - theta).clip(min=0)
#    return w
#
#def projection_simplex_sort(v, z=1):
#    n_features = v.shape[0]
#    u = np.sort(v)[::-1]
#    cssv = np.cumsum(u) - z
#    ind = np.arange(n_features) + 1
#    cond = u - cssv / ind > 0
#    rho = ind[cond][-1]
#    print("s",u - cssv / ind)
#    print("s",cond)
#    print("s",rho)
#    theta = cssv[cond][-1] / float(rho)
#    w = np.maximum(v - theta, 0)
#    return w
#
#
#def projection_simplex_pivot(v, z=1, random_state=None):
#    rs = np.random.RandomState(random_state)
#    n_features = len(v)
#    U = np.arange(n_features)
#    s = 0
#    rho = 0
#    while len(U) > 0:
#        G = []
#        L = []
#        k = U[rs.randint(0, len(U))]
#        ds = v[k]
#        for j in U:
#            if v[j] >= v[k]:
#                if j != k:
#                    ds += v[j]
#                    G.append(j)
#            elif v[j] < v[k]:
#                L.append(j)
#        drho = len(G) + 1
#        if s + ds - (rho + drho) * v[k] < z:
#            s += ds
#            rho += drho
#            U = L
#        else:
#            U = G
#    theta = (s - z) / float(rho)
#    return np.maximum(v - theta, 0)

#def project_simplex(v, s=1):
#    """
#    Taken from https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246.
#    Compute the Euclidean projection on a positive simplex
#    Solves the optimisation problem (using the algorithm from [1]):
#        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
#    Parameters
#    ----------
#    v: (n,) numpy array,
#       n-dimensional vector to project
#    s: int, optional, default: 1,
#       radius of the simplex
#    Returns
#    -------
#    w: (n,) numpy array,
#       Euclidean projection of v on the simplex
#    Notes
#    -----
#    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
#    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
#    However, this implementation still easily scales to millions of dimensions.
#    References
#    ----------
#    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
#        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
#        International Conference on Machine Learning (ICML 2008)
#        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
#    """
#
#    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
#
#    n, = v.shape  # will raise ValueError if v is not 1-D
#    # check if we are already on the simplex
#    if v.sum() == s and numpy.alltrue(v >= 0):
#        # best projection: itself!
#        return v
#    # get the array of cumulative sums of a sorted (decreasing) copy of v
#    u = numpy.sort(v)[::-1]
#    cssv = numpy.cumsum(u)
#    # get the number of > 0 components of the optimal solution
#    rho = numpy.nonzero(u * numpy.arange(1, n+1) > (cssv - s))[0][-1]
#    # compute the Lagrange multiplier associated to the simplex constraint
#    theta = float(cssv[rho] - s) / rho
#    # compute the projection by thresholding v using theta
#    w = (v - theta).clip(min=0)
#    return w
