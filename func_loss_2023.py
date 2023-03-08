import numpy as np
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy as dc

def delay_embed(data, tau, max_dim):
    """Delay embed data by concatenating consecutive increase delays.
    Parameters
    ----------
    data : array, 1-D
        Data to be delay-embedded.
    tau : int (default=10)
        Delay between subsequent dimensions (units of samples).
    max_dim : int (default=5)
        Maximum dimension up to which delay embedding is performed.
    Returns
    -------
    x : array, 2-D (samples x dim)
        Delay embedding reconstructed data in higher dimension.
    """
    if type(tau) is not int:
        tau = int(tau)

    num_samples = len(data) - tau * (max_dim - 1)
    return np.array([data[dim * tau:num_samples + dim * tau] for dim in range(max_dim)]).T[:,::-1]

def compute_nn_dist(data, tau=1, fixed_dim=3, cut=False, algorithm='kd_tree', metric='chebyshew'):
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)
    if cut:
        data_vec = dc(data_vec[tau * fixed_dim:])
    # compute nearest neighbor index with sklearn.nearestneighbors
    #   here, training and test data is the same set, so it will always
    #   return at least 1 neighbor, which is the point itself, so we want
    #   the second nearest neighbor
    dist, idx = NearestNeighbors(n_neighbors=2, algorithm=algorithm, metric=metric).fit(data_vec).kneighbors(data_vec)

    real_distances = dist[:,1]
    av_distance=np.mean(real_distances)
    return av_distance

def hessian(x):
    #Second derivative along the phase space curve
    return dc(np.gradient(np.gradient(x, axis=0), axis=0))

def grad_ps(x):
    #Second derivative along the phase space curve
    return dc(np.gradient(x, axis=0))

def mean_2nd_derivative(data, tau=10, fixed_dim=3, cut=False):
    # phase space embedding
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)

    #second derivative, squaring, summing for each point, sqare root, mean
    if cut:
        return dc(np.mean(np.sqrt(np.sum(np.square(hessian(data_vec[tau * fixed_dim:])), axis=1))))
    else:
        return dc(np.mean(np.sqrt(np.sum(np.square(hessian(data_vec)), axis=1))))

def var_2nd_derivative(data, tau=1, fixed_dim=3, cut=False):
    # phase space embedding
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)
    if cut:
        return dc(np.var(np.sqrt(np.sum(np.square(hessian(data_vec[tau*fixed_dim:])), axis=1))))
    else:
        return dc(np.var(np.sqrt(np.sum(np.square(hessian(data_vec)), axis=1))))

def mean_1st_derivative(data, tau=10, fixed_dim=3, cut=False):
    # phase space embedding
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)

    # second derivative, squaring, summing for each point, sqare root, mean
    if cut:
        return dc(np.mean(np.sqrt(np.sum(np.square(grad_ps(data_vec[tau*fixed_dim:])), axis=1))))
    else:
        return dc(np.mean(np.sqrt(np.sum(np.square(grad_ps(data_vec)), axis=1))))

def var_1st_derivative(data, tau=10, fixed_dim=3, cut=False):
    # phase space embedding
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)

    # second derivative, squaring, summing for each point, sqare root, mean
    if cut:
        return dc(np.var(np.sqrt(np.sum(np.square(grad_ps(data_vec[tau*fixed_dim:])), axis=1))))
    else:
        return dc(np.var(np.sqrt(np.sum(np.square(grad_ps(data_vec)), axis=1))))

