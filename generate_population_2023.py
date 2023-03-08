"""
This program generates a population of randomly interpolated data as done in:
Raubitzek, S.; Neubauer, T.; Friedrich, J.; Rauber, A. Interpolating Strange Attractors via Fractional Brownian Bridges. Entropy 2022, 24, 718. https://doi.org/10.3390/e24050718
(https://www.mdpi.com/1099-4300/24/5/718)
The data set that's part of the folders is the monthly international airline passengers data set from the
Time Series Data Library by Hyndman et al. You can also load the car sales data set from the Time Series Data Library
All you need to do is to set the number of interpolation points, the size of the population and load the data set.
-Sebastian Raubitzek, 08.03.2023
"""

import numpy as np
from func_bb_interpolation_2023 import *
import random
from func_orga_2023 import *

########################################################################################################################
# MAIN PARAMETERS ######################################################################################################
########################################################################################################################
data_name = "airline_passengers" #data name
#data_name = "car_sales" #data name
n_intp = 3 #nr of additional (new) inteprolated data points
n_pop = 1000 # 1000 is default, for a quick run n_pop=100, size of the population for the following genetic algorithm
#embd_dim = 3 #embedding dimension for the reconstructed phase space, find via e.g. false nearest neighbors
#time_delay = 1 #time delay for the reconstruted phase space, find via e.g. minimal mutual information
interpolation_method = 0 #you can choose between 0 or 1, 0 is the one from the paper, 1 is the one I prgrammed which is
                            #just a regular fractional Brownian Bridge
create_folder_structure(data_name)
change_fac=0.02 #vertical scaling factor sort of, applies only if interpolation_method=1
########################################################################################################################
# GENERATE POPULATION ##################################################################################################
########################################################################################################################
data = np.genfromtxt("./DATA_ORIGINAL/" + str(data_name) + ".csv", delimiter=',')
# print(data)
data = data[:, 1]

collect_n_pop = list()

for i in range(n_pop):
    print(i)
    random.seed(i)
    if interpolation_method == 0:
        X, Y = dc(interpolate_data_jf(Y=data, n_intp=n_intp, Hurst="rand"))
    if interpolation_method == 1:
        X, Y = dc(interpolate_data(Y=data, n_intp=n_intp, Hurst=True, change_fac=change_fac))

    if i==0:
        collect_n_pop.append(X)
        collect_n_pop.append(Y)
    else:
        collect_n_pop.append(Y)

collect_n_pop = dc(np.array(collect_n_pop))

np.savetxt('./DATA_POPULATION/' + str(data_name) + "_population_" + str(n_intp) + ".csv", collect_n_pop.T,  fmt='%s', delimiter=",")
