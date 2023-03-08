import numpy as np
import random
from copy import deepcopy as dc

def mating(fitness_a, genes_a, fitness_b, genes_b, p_mating):
    offspring = list()

    #fitness must be smaller as we want to minimize something

    if fitness_a < fitness_b:
        for i in range(len(genes_a[:,0])):
            if random.uniform(0,1) < p_mating:
                offspring.append(genes_a[i,:])
            else:
                offspring.append(genes_b[i,:])

    if fitness_b < fitness_a:
        for i in range(len(genes_a[:,0])):
            if random.uniform(0,1) < p_mating:
                offspring.append(genes_b[i,:])
            else:
                offspring.append(genes_a[i,:])

    if fitness_b == fitness_a:
        for i in range(len(genes_a[:,0])):
            if random.uniform(0,1) < 0.5:
                offspring.append(genes_b[i,:])
            else:
                offspring.append(genes_a[i,:])


    return np.array(offspring)

def restructure_interpolated_array(array, n_intp):
    switch = True
    out_arr = list()
    i = 0
    while switch:
        out_arr.append(list(array[i:i+n_intp+2]))
        i = i + n_intp + 1
        if i >= len(array):
            switch = False
    return dc(np.array(out_arr))

def restructure_interpolated_array_2(array, n_intp):
    switch = Trues
    out_arr = list()
    i = 0
    while switch:
        in_list = list()
        for ii in range(n_intp+2):
            in_list.append(array[i+ii])
        out_arr.append(in_list)
        i = i + n_intp + 1
        if i >= (len(array)-1):
            switch = False
    return dc(np.array(out_arr))

def interpolated_time_series_to_genes(array, n_intp):
    switch = True
    out_arr = list()
    i = 0
    while switch:
        in_list = list()
        for ii in range(n_intp+2):
            in_list.append(array[i+ii])
        out_arr.append(in_list)
        i = i + n_intp + 1
        if i >= (len(array)-1):
            switch = False
    return dc(np.array(out_arr))

def genes_to_time_series(array):
    out_list = list()
    for i in range(len(array[:,0])):
        for ii in range(len(array[0,:])-1):
            out_list.append(array[i,ii])
    out_list.append(array[-1,-1])
    return np.array(out_list)

def fit_sort_s(fit_arr, off_arr): #sort by smaller fitness
    for i in range(len(fit_arr)):
        for ii in range(len(fit_arr)-1):
            if fit_arr[ii+1]<fit_arr[ii]:
                save_fit = dc(fit_arr[ii])
                fit_arr[ii] = dc(fit_arr[ii+1])
                fit_arr[ii+1] = dc(save_fit)

                save_off = dc(off_arr[:,ii])
                off_arr[:,ii] = dc(off_arr[:,ii + 1])
                off_arr[:,ii + 1] = dc(save_off)

    return dc(fit_arr), dc(off_arr)


def fit_sort_l(fit_arr, off_arr): #sort by larger fitness
    for i in range(len(fit_arr)):
        for ii in range(len(fit_arr) - 1):
            if fit_arr[ii + 1] > fit_arr[ii]:
                save_fit = dc(fit_arr[ii])
                fit_arr[ii] = dc(fit_arr[ii + 1])
                fit_arr[ii + 1] = dc(save_fit)

                save_off = dc(off_arr[:, ii])
                off_arr[:, ii] = dc(off_arr[:, ii + 1])
                off_arr[:, ii + 1] = dc(save_off)

    return dc(fit_arr), dc(off_arr)

