import nolds
import numpy as np
import six
from copy import deepcopy as dc
import random
from fbm import FBM

def sample_path_batch_single(n_intp, start=0, Hurst=False, change_fac=None):
    if change_fac!=None:
        dt = 1.0 / (n_intp -1) * change_fac                                              #  changed from 1.0 / N
        dt_sqrt = np.sqrt(dt)
    else:
        dt = 1.0 / (n_intp -1)                                          #  changed from 1.0 / N
        dt_sqrt = np.sqrt(dt)
    if Hurst:
        dt_sqrt = dc(dt_sqrt*round(random.uniform(0.01, 100), 10))
        #dt_sqrt = dc(dt_sqrt*0.01*pow(10, round(random.uniform(0, 4), 10)))
    B = np.empty(n_intp, dtype=float)
    B[0] = start
    for n in six.moves.range(n_intp - 2):                                           # changed from "for n in six.moves.range(N - 1)"
         t = n * dt
         xi = np.random.randn() * dt_sqrt
         B[n + 1] = B[n] * (1 - dt / (1 - t)) + xi
    B[-1] = 0
    return B

def interpolate_data(Y, n_intp, X=None, Hurst=False, change_fac=None):
    if X == None:
        X = np.array(list(range(len(Y))))
    Y_interpolated = list()
    for i in range(len(Y)-1):
        orig_points = dc(Y[i:i+2])
        if orig_points[0] == orig_points[1]:
            interp_points = dc(sample_path_batch_single(n_intp+2, start=0, Hurst=Hurst, change_fac=change_fac))
            interp_points[:] = dc(interp_points[:] + orig_points[0])
        elif orig_points[0] > orig_points[1]:
            interp_points = dc(sample_path_batch_single(n_intp+2, start=(orig_points[0]-orig_points[1]), Hurst=Hurst, change_fac=change_fac))
            interp_points[:] = dc(interp_points[:] + orig_points[1])
        elif orig_points[1] > orig_points[0]:
            interp_points = dc(sample_path_batch_single(n_intp+2, start=(orig_points[1]-orig_points[0]), Hurst=Hurst, change_fac=change_fac))
            #reverse order of array
            interp_points[:] = dc(interp_points[:] + orig_points[0])
            interp_points = dc(interp_points[::-1])
        for ii in range(len(interp_points)-1):
            Y_interpolated.append(interp_points[ii])
    Y_interpolated.append(Y[-1])
    Y_out = dc(np.array(Y_interpolated))
    X_out = ((np.array(list(range(len(Y_out)))))/(len(Y_out) - 1 ))*(len(Y)-1)
    return X_out, Y_out

def interpolate_data_jf(Y, n_intp, Hurst="rand"):
    if Hurst == "orig":
        print('Adjusting Hurst exponent to the datas inherent Hurst exponent')
        H = 0
        for i in range(100):
            H = H + nolds.hurst_rs(Y)
        H = H/100
        print('Hurst set to ' + str(H))
    elif Hurst == "rand":
        print('Setting random Hurst exponent')
        H = random.uniform(0, 1)
        if H == 0:
            H = 0.001
        if H == 1:
            H == 0.999
        print('Hurst set to ' + str(H))
    else:
        try:
            print('Hurst exponent preset to ' + str(float(Hurst)))
            H = float(Hurst)
        except:
            print('Error, Hurst exponent set to 0.5')
            H = 0.5

    offset = Y[0]
    X_i = dc(Y[:] - offset)
    N_coarse = dc(len(Y))
    N_fine = N_coarse * (n_intp + 1)
    X = FBM(n=N_fine, hurst=H, length=1)
    t = X.times()
    fine_points = np.arange(t.size)
    t_i = t[::N_fine // N_coarse]
    t_i = t_i[1:]
    coarse_points = fine_points[::N_fine // N_coarse]
    coarse_points = coarse_points[1:]
    Sigma = np.zeros((N_coarse, N_coarse))
    for ii in range(N_coarse):
        for jj in range(N_coarse):
            Sigma[ii, jj] = 0.5 * (t_i[ii] ** (2 * H) + t_i[jj] ** (2 * H) - np.abs(t_i[ii] - t_i[jj]) ** (2 * H))
    Sigma_inv = np.linalg.inv(Sigma)

    X_sample = X.fbm()
    X_bridge = X_sample.copy()
    for ii in range(N_coarse):
        for jj in range(N_coarse):
            X_bridge += -(X_sample[coarse_points[ii]] - X_i[ii]) \
                        * Sigma_inv[ii, jj] * 0.5 * (t_i[jj] ** (2 * H) + t ** (2 * H) - np.abs(t_i[jj] - t) ** (2 * H))

    X_bridge = dc(X_bridge[(n_intp + 1):])
    Y_out = dc(np.array(X_bridge) + offset)
    X_out = ((np.array(list(range(len(Y_out)))))/(len(Y_out) - 1 ))*(len(Y)-1)
    return X_out, Y_out



