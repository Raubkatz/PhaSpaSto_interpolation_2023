"""
This program performs a genetic algorithm on a population of randomly interpolated data as done in:
Raubitzek, S.; Neubauer, T.; Friedrich, J.; Rauber, A. Interpolating Strange Attractors via Fractional Brownian Bridges. Entropy 2022, 24, 718. https://doi.org/10.3390/e24050718
(https://www.mdpi.com/1099-4300/24/5/718)
The data set that's part of the folders is the monthly international airline passengers data set from the
Time Series Data Library by Hyndman et al. You can also load the car sales data set from the Time Series Data Library
All you need to do is to set the number of interpolation points, number of generations, the phase space embedding and load the data set.
-Sebastian Raubitzek, 08.03.2023
"""

import numpy as np
from func_bb_interpolation_2023 import *
import random
import func_gen_alg_2023
from sklearn.preprocessing import MinMaxScaler
from func_orga_2023 import *
import func_loss_2023
from datetime import datetime
from matplotlib import pyplot as plt

########################################################################################################################
# MAIN PARAMETERS ######################################################################################################
########################################################################################################################
data_name = "airline_passengers" #data name
#data_name = "car_sales" #data name
n_intp = 3 #nr of additional (new) inteprolated data points
n_generations = 1000
x_label = "Months"
y_label = "International Airline Passengers"
embd_dim = 3 #embedding dimension for the reconstructed phase space, find via e.g. false nearest neighbors
time_delay = 1 #time delay for the reconstruted phase space, find via e.g. minimal mutual information
########################################################################################################################
# NOT SO MAIN PARAMETERS ######################################################################################################
########################################################################################################################
mutation = True #switch mutation on off
p_mut = 0.5 #mutaiton probability
scale = False #scales the data before loss function, doesn't change too much, but might make a significant difference for certain data sets
detrend = True #set true for time series if a clear linear trend is visible
detrend_ps_plot = True #if reconstructed phase space plot is detrended, should be true if data set set has visible trend

set_fine_grained_time_delay = False #If True then the the time delay of the inteprolated data set is the reference point
time_delay_fg = 4 #if set_fine_grained_time_delay=True then time_delay_fg will overwrite time_delay, if false then vice versa
loss_function = 4 # 0=nn distance, 1=first derivative mean, 2=first derivative var, 3=second derivative mean, 4=second derivative var; default is 4, the variance of second derivatives
multi_loss = False
check_abort = True #criterion for abortion of genetic algorithm if no changes for 10 generations
check_5 = True #criterion for abortion of genetic algorithm if no changes for 5 generations, check_abort must be true for chekc 5 to trigger
loss_list = [0,2,4] # selection of loss functions to be used together
p_mating = 0.5  # weighting the genes towards the fitter parent
scale_low = 0.0
scale_high = 1.0
parent_pop = 0.5  # only upper 50% are allowed to mate
cut_edges = True #cut off artifacts from phase space embeddings, i.e. the parts where the end is attached to the beginning of the time series
########################################################################################################################
# GENETIC ALGORITHM ####################################################################################################
########################################################################################################################
start = dc(datetime.now())
print('#################')
print('PhaSpaSto Interpolation Start: ', start)
print('#################')
#load original data
orig_data = np.genfromtxt("./DATA_ORIGINAL/" + str(data_name) + ".csv", delimiter=',')
orig_data_X = dc(orig_data[:,0])
orig_data = dc(orig_data[:,1]) #separate into x and y component
#load population
population = np.genfromtxt('./DATA_POPULATION/' + str(data_name) + "_population_" + str(n_intp) + ".csv", delimiter=',', dtype=float)
work_data_X = dc(population[:, 0]) #separate into x and y component
work_data = dc(population[:, 1:])

tau_to_use_fg = time_delay * (n_intp + 1) #calculate time delay for interpolated time series
if set_fine_grained_time_delay:
    tau_to_use_fg = time_delay_fg
    time_delay = int(round(time_delay_fg/(n_intp + 1)))

#Y_lin = list()
if detrend: #subtract a linear fit
    Y_lin_orig = dc(linear_fit(orig_data_X, orig_data, orig_data_X))
    Y_lin_interp = dc(linear_fit(orig_data_X, orig_data, work_data_X))
    orig_data = dc(orig_data - Y_lin_orig)
    for i in range(len(work_data[0, :])): # do for each interpolated time series in the population
        work_data[:, i] = dc(work_data[:, i] - Y_lin_interp)

if scale: # normalize the dataset
    scaler_orig = dc(MinMaxScaler(feature_range=(scale_low, scale_high)))
    orig_data = dc(un_brace(scaler_orig.fit_transform(re_brace(orig_data))))
    for i in range(len(work_data[0, :])): # do for each interpolated time series in the population
        work_data[:, i] = dc(un_brace(scaler_orig.transform(re_brace(work_data[:, i]))))

check_arr = dc(np.empty(10, dtype=float)) # counting the number of same occurences, to abort the program
multi_count = 0 #applies only if theres more than one loss function, i.e. multicount=True
parent_size = dc((int(len(work_data[0,:])*parent_pop))) #number of parents used to generate new generation, not all parents are but rather only half of them
fin_gen = 0 # final generation
fin_mean_loss = 0 # final loss on average, if abortion criteria trigger, then this is the value of the interpolation

for gen in range(n_generations): # generation loop
    print('gen ' + str(gen))
    #caculate fitness for all data sets
    fit_list = list()
    if multi_loss: #more than one loss function?
        loss_function = dc(loss_list[multi_count])
        print('multi loss ' + str(loss_function))
        multi_count = dc(multi_count + 1)
        if multi_count == len(loss_list):
            multi_count = 0

    if mutation: #mutation yes/no
        mut_prob = dc(random.uniform(0,1))
        if mut_prob < p_mut:
            print('Mutation \n\n          ƪ(`▿▿▿▿ ´ƪ)          \n ')
            samp_mut = dc(random.sample(range(len(work_data[0,:])), 1))
            X, Y = dc(interpolate_data_jf(Y=orig_data, n_intp=n_intp, Hurst="rand"))
            work_data[:, samp_mut[0]] = dc(Y)

    if loss_function == 0:
        for i in range(len(work_data[0, :])):
            fit_list.append(func_loss_2023.compute_nn_dist(work_data[:, i], tau=tau_to_use_fg, fixed_dim=embd_dim, cut=cut_edges))
    elif loss_function == 1:
        for i in range(len(work_data[0, :])):
            fit_list.append(func_loss_2023.mean_1st_derivative(work_data[:, i], tau=tau_to_use_fg, fixed_dim=embd_dim, cut=cut_edges))
    elif loss_function == 2:
        for i in range(len(work_data[0, :])):
            fit_list.append(func_loss_2023.var_1st_derivative(work_data[:, i], tau=tau_to_use_fg, fixed_dim=embd_dim, cut=cut_edges))
    elif loss_function == 3:
        for i in range(len(work_data[0, :])):
            fit_list.append(func_loss_2023.mean_2nd_derivative(work_data[:, i], tau=tau_to_use_fg, fixed_dim=embd_dim, cut=cut_edges))
    elif loss_function == 4:
        for i in range(len(work_data[0, :])):
            fit_list.append(func_loss_2023.var_2nd_derivative(work_data[:, i], tau=tau_to_use_fg, fixed_dim=embd_dim, cut=cut_edges))

    fit_list = dc(np.array(fit_list))

    check_arr = dc(np.roll(check_arr, 1)) #sort the array to check for abortion
    check_arr[0] = dc(np.average(fit_list)) # add latest element
    print(check_arr[0]) # print out latest average of fitness

    fit_list, work_data = dc(func_gen_alg_2023.fit_sort_s(fit_list, work_data))
    fin_gen = dc(gen)
    fin_mean_loss = dc(check_arr[0])
    if gen == n_generations:
        print('Last run')
        print('Done')
        break

    # generate a new population for the next generation
    for hab in range(parent_size):
        #pick two random parents but only from the upper percentage, defined by parent_size
        samp = dc(random.sample(range(parent_size),2))
        #generate offspring
        offspring = dc(func_gen_alg_2023.mating(fit_list[samp[0]], func_gen_alg_2023.interpolated_time_series_to_genes(work_data[:,samp[0]], n_intp), fit_list[samp[1]], func_gen_alg_2023.interpolated_time_series_to_genes(work_data[:,samp[1]], n_intp), p_mating))
        #put offspring into correct structure
        offspring = dc(func_gen_alg_2023.genes_to_time_series(offspring))
        #append offspring
        work_data[:,-(hab+1)] = dc(offspring) # filling from the other side, because it will be sorted anyways

    if check_abort:
        if check_5: #abort if 5 consecutive fitness averages are the same
            if (check_arr[0] == check_arr[1]) and (check_arr[1] == check_arr[2]) and (check_arr[2] == check_arr[3]) and (check_arr[3] == check_arr[4]):
                break
        else:  #abort if 10 consecutive fitness averages are the same
            if (check_arr[0] == check_arr[1]) and (check_arr[1] == check_arr[2]) and (check_arr[2] == check_arr[3]) and (check_arr[3] == check_arr[4]) and (check_arr[4] == check_arr[5] and (check_arr[5] == check_arr[6]) and (check_arr[6] == check_arr[7])and (check_arr[7] == check_arr[8]) and (check_arr[8] == check_arr[9])):
                break

########################################################################################################################
# SAVE RESULTS #########################################################################################################
########################################################################################################################
if scale:
    # normalize the dataset
    for i in range(len(work_data[0, :])):
        work_data[:, i] = dc(un_brace(scaler_orig.inverse_transform(re_brace(work_data[:, i ]))))

if detrend:
    orig_data = dc(orig_data + Y_lin_orig)
    for i in range(len(work_data[0, :])):
        work_data[:, i] = dc(work_data[:, i] + Y_lin_interp)

now = dc(datetime.now())
end = dc(now - start)

print('#################')
print('Finished: ', now)
print('Runtime: ', end)
print('#################')

gen_out_txt = str(data_name) + "\n"
gen_out_txt = gen_out_txt + "fin_gen:" +  "\n" + str(fin_gen) + "\n" + "fin_mean_loss:" + "\n" + str(fin_mean_loss) + "\n"
gen_out_txt = gen_out_txt + "Mutation:" + "\n" + str(mutation) + "\n" + "Mutation probability:" + "\n" + str(p_mut) + "\n" + "Maximal Generations" + "\n" + str(n_generations) + "\n" + 'Runtime:' + "\n" + str(end)

work_data_out = np.empty((len(work_data[:, 0]), len(work_data[0, :]) + 1), dtype=float)
work_data_out[:,0] = dc(work_data_X)
work_data_out[:,1:] = dc(work_data)
np.savetxt("./DATA_GEN_ALG_IMPROVED/" + data_name + "_" + str(n_intp) + "_gen_improved.csv", work_data_out,  fmt='%s', delimiter=",")
InfoPath = "./DATA_GEN_ALG_IMPROVED/" + data_name + "_" + str(n_intp) + "_gen_improved_info.txt"
InfoFile = open(InfoPath, "w+")
InfoFile.write(gen_out_txt)
InfoFile.close()
########################################################################################################################
# PLOT INTERPOLATION ###################################################################################################
########################################################################################################################
plt.plot(orig_data_X, orig_data, '--', color="blue")
plt.scatter(orig_data_X, orig_data, label="original data", color="black")
plt.plot(work_data_X, work_data[:,0], label="PhaSpaSto interpolation", color="orange")
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig('./PLOTS/plot_phaspasto_int_' + data_name + str(n_intp) +'_nintp.png')  #save png
plt.savefig('./PLOTS/plot_phaspasto_int_' + data_name + str(n_intp) +'_nintp.eps')  #save eps, use always eps for journals
plt.show()

if detrend_ps_plot: #if detrended before plot, for data with visible trends this should be done
    orig_data = dc(orig_data - Y_lin_orig)
    work_data[:,0] = dc(work_data[:,0] - Y_lin_interp)
#Reconstructed phase space fine grained and interpolated data plot
data_lag0 = dc(orig_data.flatten())
data_lag1 = dc(np.roll(orig_data, time_delay).flatten())
data_lag2 = dc(np.roll(orig_data, 2 * time_delay).flatten())
data_lag0 = dc(data_lag0[embd_dim*time_delay:])
data_lag1 = dc(data_lag1[embd_dim*time_delay:])
data_lag2 = dc(data_lag2[embd_dim*time_delay:])
# Plot time delay embedding
fig = plt.figure()
cm = dc(plt.get_cmap("RdYlGn"))
col = dc([cm(float(iii) / (len(data_lag0))) for iii in range(len(data_lag0))])
ax = fig.add_subplot(111, projection='3d')
# plt.title(' Attractor ' + str(workingtitle))
for ii in range(len(data_lag0) - 1):
    ax.plot(data_lag0[ii:ii + 2], data_lag1[ii:ii + 2], data_lag2[ii:ii + 2], color=plt.cm.jet(int(255 * ii / (len(data_lag0)))), linewidth=0.8)
plt.savefig('./PLOTS/rec_phase_space_plot_' + data_name + '.png')  # , dpi=150)
plt.savefig('./PLOTS/rec_phase_space_plot_' + data_name + '.eps')  # , dpi=150)
plt.show()

#Reconstructed phase space fine grained and interpolated data plot
data_lag0 = dc(work_data[:,0].flatten())
data_lag1 = dc(np.roll(work_data[:,0], tau_to_use_fg).flatten())
data_lag2 = dc(np.roll(work_data[:,0], 2 * tau_to_use_fg).flatten())
data_lag0 = dc(data_lag0[embd_dim*tau_to_use_fg:])
data_lag1 = dc(data_lag1[embd_dim*tau_to_use_fg:])
data_lag2 = dc(data_lag2[embd_dim*tau_to_use_fg:])

# Plot time delay embedding
fig = plt.figure()
cm = plt.get_cmap("RdYlGn")
col = [cm(float(iii) / (len(data_lag0))) for iii in range(len(data_lag0))]
ax = fig.add_subplot(111, projection='3d')
for ii in range(len(data_lag0) - 1):
    ax.plot(data_lag0[ii:ii + 2], data_lag1[ii:ii + 2], data_lag2[ii:ii + 2], color=plt.cm.jet(int(255 * ii / (len(data_lag0)))), linewidth=0.8)
plt.savefig('./PLOTS/rec_phase_space_plot_' + data_name + '_' + str(n_intp) + '_fg.png')
plt.savefig('./PLOTS/rec_phase_space_plot_' + data_name + '_' + str(n_intp) + '_fg.eps')
plt.show()
