import os
import numpy as np
from copy import deepcopy as dc

def create_folder_structure(workingtitle):
    # create folder structure
    if not os.path.exists('./DATA_POPULATION'):
        os.makedirs('./DATA_POPULATION')
    if not os.path.exists('./DATA_GEN_ALG_IMPROVED'):
        os.makedirs('./DATA_GEN_ALG_IMPROVED')
    if not os.path.exists('./PLOTS'):
        os.makedirs('./PLOTS')
    print("\n" + "folders created" + "\n")

def re_brace(dataset): # adds an additional brackst to the inner values
    Out_Arr = []
    for i in range(len(dataset)):
        Out_Arr.append(dataset[(i):(i+1)])
    return np.array(Out_Arr)

def un_brace(dataset): # removes the inner bracket
    Out_Arr = np.empty([len(dataset)])
    for i in range(len(dataset)):
        Out_Arr[i] = dataset[i,0]
    return Out_Arr

def linear_fit(train_set_X, train_set_Y, full_set_X):  # do a linear fit on all points
    coefficients = np.polyfit(train_set_X, train_set_Y, 1) #determine coefficients
    lin_Y = list()
    for i in range(len(full_set_X)):
        y = Lin_Model(coefficients[0], coefficients[1], full_set_X[i])
        lin_Y.append(y)
    return dc(np.array(lin_Y))

def cubic_fit(train_set_X, train_set_Y, full_set_X):  # do a cubic fit on all points
    coefficients = np.polyfit(train_set_X, train_set_Y, 3) #determine coefficients
    lin_Y = list()
    for i in range(len(full_set_X)):
        y = Cubic_Model(coefficients[0], coefficients[1], coefficients[2], coefficients[3], full_set_X[i])
        lin_Y.append(y)
    return np.array(lin_Y)

def five_fit(train_set_X, train_set_Y, full_set_X):  # do a cubic fit on all points
    coefficients = np.polyfit(train_set_X, train_set_Y, 5) #determine coefficients
    lin_Y = list()
    for i in range(len(full_set_X)):
        y = Five_Model(coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4], coefficients[5], full_set_X[i])
        lin_Y.append(y)
    return np.array(lin_Y)

def Lin_Model(a, b, x):  # linear model for linear interpolation
    return a * x + b

def Cubic_Model(a,b,c,d,x):
    return a*x*x*x + b*x*x + c*x + d

def Five_Model(a,b,c,d,e,f,x):
    return a*x*x*x*x*x + b*x*x*x*x + c*x*x*x + d*x*x + e*x + f



