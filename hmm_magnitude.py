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
import ipdb
import glob

warnings.filterwarnings("ignore", category= DeprecationWarning)

training_dataset      = 'REAL_HIRO_ONE_SA_SUCCESS'
training_file_id      = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

testing_dataset       = 'REAL_HIRO_ONE_SA_SUCCESS'
testing_file_id       = ['06']

rootDATApath          = '/home/birl/npBayesHMM/HIRO_SA_DATA/'
trained_models_path   = '/home/birl/npBayesHMM/python/state_estimation/hmm_magnitude'
dataType              = ['FT_magnitude.dat'] #'R_Torques.dat', 'R_Angles.dat','R_CartPos.dat' tuples can not edited
time_step             = 0.005
n_components          = 20
covariance_type       = "diag"
Niter                 = 1000
scale_length          = 10

def scaling(X):
    _index, _column = X.shape
    Data_scaled     = []
    for i in range(scale_length, _index-scale_length-2):
        scaled      = preprocessing.scale(X[i-scale_length:i+scale_length + 1, :])
        Data_scaled.append(scaled[scale_length,:])

    scaled_array    = np.asarray(Data_scaled)
    return scaled_array

def load_one_trial(dataPath, file_id):
    sensor            = pd.DataFrame()
    minLen            = np.zeros(len(file_id))
    for idx in range(len(file_id)):
        for dat_file in glob.glob(os.path.join(dataPath,"20121127-HIROSA-S-" + file_id[idx] + "/" + "FT_magnitude.dat")):
            raw_data  = pd.read_csv(dat_file , sep='\s+', header=None, skiprows=1)
            sensor    = raw_data.transpose().append(sensor) # transpose to [ndim x length]
            minLen[idx] = raw_data.shape[0]
    sensor = sensor.transpose().values
    sensor = np.delete(sensor, np.arange(min(minLen), max(minLen)),0) #aligment
    return  sensor

def training(training_dataPath):
    if os.path.isdir(trained_models_path ): # if exit
        shutil.rmtree(trained_models_path, ignore_errors= False, onerror= None)
    training_sensor = load_one_trial(training_dataPath, training_file_id)

    print "Training the normal execution"

    start_prob = np.zeros(n_components)
    start_prob[0] = 1
    train_model = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                          params="mct", init_params="cmt", n_iter=Niter)
    train_model.startprob_ = start_prob


    train_model = train_model.fit(training_sensor)
    #save the models
    if not os.path.isdir(trained_models_path ):
        os.makedirs(trained_models_path )
    joblib.dump(train_model, trained_models_path  + "/tainedModel"  +".pkl")
    return training_sensor

def testing(testing_dataPath):
    #load testing file
    testing_sensor= load_one_trial(testing_dataPath, testing_file_id)

    #load trained model
    train_model = joblib.load(os.path.join(trained_models_path, "tainedModel" + ".pkl"))
    log_likelihood = np.zeros(len(testing_sensor))
    obs = []
    for obs_idx in range(0, len(testing_sensor)):
        print "testing: " + str(obs_idx) + "/" + str(len(testing_sensor))
        obs.append(testing_sensor[obs_idx, :])
        log_likelihood[obs_idx, 0] = train_model.score(obs)
    return testing_folder_name, testing_raw_data, testing_sensor, log_likelihood
if __name__ == '__main__':
    # for training
    training_sensor = training(rootDATApath + training_dataset)
    #training_sensor = load_one_trial(rootDATApath + training_dataset)
   # raw_input("Press ENTER to testing the trail from different experiment: ")

    # for testing
    testing_sensor, log_likelihood  = testing(rootDATApath + testing_dataset)
    plt.figure(1)
    ax_1 = plt.subplot(411)
    plt.plot(training_raw_data)
    plt.grid(True)
    plt.title("The training signals in " + training_dataset + "/" + training_folder_name)

    ax_2 = plt.subplot(412)
    plt.plot(testing_raw_data)
    plt.grid(True)
    plt.title("The testing  signals in " + testing_dataset + "/" + testing_folder_name)

    ax_3 = plt.subplot(413)
    plt.plot(testing_sensor)
    plt.grid(True)
    plt.title("The testing  signals after scaling preprocessing")

    ax_4= plt.subplot(414)
    #np.savetxt('s_threshold.out', np.amax(log_likelihood, axis=1)) # save the standard_thhreshold by training and testing the same successful trial.
    #plt.plot(np.loadtxt('Standard_threshold.out'), 'b^', linewidth=2, label = 'Standard threshold')
    plt.plot(log_likelihood, linewidth=3, label = "standard_threshold")
    plt.grid(True)
    plt.title("Log-Likelihood")
    plt.legend(bbox_to_anchor = (0., 1.02, 1.,  .102), loc='lower right', frameon=True, shadow = True, ncol = 1)
    plt.savefig("standard_threshold", format="eps")
    plt.show()

