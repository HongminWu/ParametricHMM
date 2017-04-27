#!/usr/bin/env python -W ignore::DeprecationWarning
# @ Hongmin Wu
# April 16, 2017
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import *
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
import warnings
import shutil
import matplotlib.animation as animation
import glob
import ipdb
warnings.filterwarnings("ignore", category= DeprecationWarning)

testing_dataset       = 'REAL_HIRO_ONE_SA_SUCCESS' #REAL_HIRO_ONE_SA_ERROR_CHARAC
testing_file_id       = ['15','16','17','18','19','20']
trained_model_id      = ['15','16','17','18','19','20']                  #define the trained_model

trained_models_path   = '/home/birl/npBayesHMM/python/state_estimation/trained_models'
rootDATApath          = '/home/birl/npBayesHMM/HIRO_SA_DATA/'
dataType              = ['R_Torques.dat','R_Angles.dat'] #'R_Angles.dat','R_CartPos.dat' tuples can not editedtime_step
scale_length          = 10

def scaling(X):
    _index, _column = X.shape
    Data_scaled     = []
    for i in range(scale_length, _index-scale_length-2):
        scaled      = preprocessing.scale(X[i-scale_length:i+scale_length + 1, :])
        Data_scaled.append(scaled[scale_length,:])

    scaled_array    = np.asarray(Data_scaled)
    return scaled_array

def load_one_trial(dataPath, id):
    sensor            = pd.DataFrame()
    for dat_file in dataType:
        for dat_path in glob.glob(os.path.join(dataPath, "*" + id + "/" + dat_file)):
            raw_data  = pd.read_csv(dat_path, sep='\s+', header=None, skiprows=1, usecols = range(1,7))
            sensor    = raw_data.transpose().append(sensor) # transpose to [ndim x length]
    return  dat_path, sensor.transpose().values, scaling(sensor.transpose().values)

def testing(testing_dataPath, testing_file_id, model_id):
    #load testing file
    testing_folder_name, \
    testing_raw_data, \
    testing_sensor = load_one_trial(testing_dataPath, testing_file_id)
    #load trained model
    train_model    = joblib.load(os.path.join(trained_models_path, "tainedModel_" + model_id + ".pkl"))
    #incremental testing
    log_likelihood = []
    obs            = []
    for obs_idx in range(0, len(testing_sensor), scale_length): #range(start, end, step)
        print "testing: " + str(obs_idx) + "/" + str(len(testing_sensor))
        obs.append(testing_sensor[obs_idx, :])
        log_likelihood.append(train_model.score(obs))
        # logprob, state_sequence =train_model.decode(obs) #find most likely state sequence corresponding to obs
        # print logprob, state_sequence
    return testing_folder_name, testing_raw_data, testing_sensor, log_likelihood, train_model

if __name__ == '__main__':
    # for testing
    #assume we have N trained model, then (N - 1) log-likelihood vector for each trained model
    #step_2
    CV_likelihood = []
    for model_id in trained_model_id:
        temp_ = []
        temp_likelihood = []
        minLen = []
        for file_id in testing_file_id:
            if file_id == model_id: #leave one out cross validation
                continue
            testing_folder_name, \
            testing_raw_data,    \
            testing_sensor,      \
            log_likelihood,      \
            train_model          = testing(rootDATApath + testing_dataset, file_id, model_id)
            temp_.append(log_likelihood) #collect all the initial likelihood
            minLen.append(len(log_likelihood))
        minIdx = np.argmin(minLen)
        minLen = min(minLen)
        std = pd.Series(temp_[minIdx])
        for align in range(len(temp_)):
             other = pd.Series(temp_[align])
             other = std.align(other,join='inner')
             temp_likelihood.append(other[1].tolist())
        CV_likelihood.append(temp_likelihood)
    #step_3: calculate the mean and standard derivation of each trained model#convert the list to ndarray
    mean_sd_likelihood = []
    finial_mean        = []
    for nmodel in range(len(CV_likelihood) ):
        temp_ = []
        temp_mean = np.asarray(CV_likelihood[nmodel][:]).mean(axis= 0)        #mean
        temp_.append(temp_mean)
        finial_mean.append(temp_mean[-1]) # for select the optimal train

        temp_var = np.sqrt(np.asarray(CV_likelihood[nmodel][:]).var(axis= 0)) #standard derivation
        temp_.append(temp_var)
        mean_sd_likelihood.append(temp_)
    #step_4 each trained model is collected with one mean and one standard derivation
    optimalModel = np.argmax(np.array(finial_mean))
    #calculate the threshold of likelihood
    thresthod =  np.array(mean_sd_likelihood[optimalModel][0] - 2 * mean_sd_likelihood[optimalModel][1])
    expected_likelihood = np.array(mean_sd_likelihood[optimalModel][0])

    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    plt.plot(thresthod,'r--',linewidth=3, label="thresthod")
    plt.grid(True)
    # plt.title("The testing  signals in " + testing_dataset + "/" + testing_folder_name)

    #ax_2 = fig_1.add_subplot(212)
    # load trained model
   # plt.grid(True)
   # plt.title("expected_likelihood")
    plt.plot(expected_likelihood, 'b-', linewidth=3, label="expected_likelihood")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right', frameon=True, shadow=True, ncol=1)
    # plt.savefig("fig_wrench_angles_state_"
    #             + str(train_model.n_components)
    #             + "_Covariance_" + train_model. covariance_type
    #             + "_Transition_fixed_"
    #             + "_test_" + testing_file_id
    #             + ".eps",
    #             format="eps")
    # plt.annotate('Hidden States:' + str(train_model.n_components) + ", "
    #              + ' GaussianHMM_cov=' + train_model.covariance_type + ", "
    #              + "fixed transition matrix",
    #              xy=(0, 0), xycoords='data',
    #              xytext=(+10, -30), textcoords='offset points', fontsize=16,
    #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.show()