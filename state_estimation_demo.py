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
import glob
import ipdb

warnings.filterwarnings("ignore", category= DeprecationWarning)

training_dataset      = 'REAL_HIRO_ONE_SA_SUCCESS'
training_file_id      = '09'

testing_dataset       = 'REAL_HIRO_ONE_SA_SUCCESS'
testing_file_id       = '10'

rootDATApath          = '/home/birl/npBayesHMM/HIRO_SA_DATA/'
trained_models_path   = '/home/birl/npBayesHMM/python/state_estimation/learned_models'
dataType              = ['R_Torques.dat','R_Angles.dat'] #'R_Angles.dat','R_CartPos.dat' tuples can not edited
label_str             = ['approach', 'rotation', 'insertion', 'mating']
time_step             = 0.005
n_components          = 25 # number of states in the model
n_mix                 =  2 # number of components in the GMM
covariance_type       = "diag"
Niter                 = 2000
scale_length          = 10
HMM_TYPE              = "GaussianHMM" #GMMHMM

def scaling(X):
    _index, _column = X.shape
    Data_scaled     = []
    for i in range(scale_length, _index-scale_length-2):
        scaled      = preprocessing.scale(X[i-scale_length:i+scale_length + 1, :])
        Data_scaled.append(scaled[scale_length,:])

    scaled_array    = np.asarray(Data_scaled)
    return scaled_array

def load_one_trial(dataPath,id):
    sensor            = pd.DataFrame()
    Rstate            = pd.DataFrame()
    #folders           = [d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath,d))]
    for folders in glob.glob(os.path.join(dataPath, "*" + id)):
        for dat_file in os.listdir(folders):
            if dat_file in dataType:
                raw_data  = pd.read_csv(folders +'/' + dat_file, sep='\s+', header=None, skiprows=1, usecols = range(1,7))
                sensor    = raw_data.transpose().append(sensor) # transpose to [ndim x length]
            elif dat_file == 'R_State.dat':
                Rstate    = pd.read_csv(folders + '/' + dat_file, sep='\s+', header=None, skiprows=1)
                Rstate    = Rstate.values / time_step
                Rstate    = np.insert(Rstate, 0, 0, axis= 0)
                Rstate    = np.append(Rstate, sensor.shape[1])
    return  folders, raw_data.values, scaling(sensor.transpose().values) , Rstate

def training(training_dataPath, training_file_id):
    if os.path.isdir(trained_models_path ): # if exit
        shutil.rmtree(trained_models_path, ignore_errors= False, onerror= None)
    training_folder_name, training_raw_data, training_sensor, Rstate = load_one_trial(training_dataPath, training_file_id)
    for istate in range(len(Rstate) - 1):
        print "Training: state_" + str(istate)
        start_prob = np.zeros(n_components)
        start_prob[0] = 1
        if HMM_TYPE == "GaussianHMM":
            train_model = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                               params="mct", init_params="cmt", n_iter=Niter)
        elif HMM_TYPE == "GMMHMM":
            train_model = GMMHMM(n_components=n_components,n_mix= n_mix, covariance_type=covariance_type,
                               params="tmcw", init_params="tmcw", n_iter=Niter)
        train_model.startprob_ = start_prob
        train_model = train_model.fit(training_sensor[int(Rstate[istate]):int(Rstate[istate + 1]),:])
        #save the models
        if not os.path.isdir(trained_models_path ):
            os.makedirs(trained_models_path )
        joblib.dump(train_model, trained_models_path  + "/state_" + str(istate)+".pkl")
    return training_folder_name, training_raw_data, training_sensor

def testing(testing_dataPath, testing_file_id):
    #load testing file
    testing_folder_name, testing_raw_data, testing_sensor, Rstate = load_one_trial(testing_dataPath, testing_file_id)

    #load trained model
    folders = os.listdir(trained_models_path)
    log_likelihood = np.zeros((len(testing_sensor),len(folders)))
    obs = []
    for obs_idx in range(0, len(testing_sensor)):
        print "testing: " + str(obs_idx) + "/" + str(len(testing_sensor))
        if obs_idx in Rstate:
            obs = []
        obs.append(testing_sensor[obs_idx, :])
        for imodel in range(len(folders)):
            train_model = joblib.load(os.path.join(trained_models_path, "state_"+str(imodel)+".pkl"))
            log_likelihood[obs_idx, imodel] = train_model.score(obs)
    return testing_folder_name, testing_raw_data, testing_sensor, log_likelihood
if __name__ == '__main__':
    # for training
    training_folder_name, training_raw_data, training_sensor = training(rootDATApath + training_dataset, training_file_id)

#    raw_input("Press ENTER to testing the trail from different experiment: ")

    # for testing
    testing_folder_name, testing_raw_data, testing_sensor, log_likelihood  = testing(rootDATApath + testing_dataset, testing_file_id)
    plt.figure(1)
    ax_1 = plt.subplot(311)
    plt.plot(training_raw_data)
    plt.grid(True)
    plt.title("The training signals in " + training_dataset + "/" + training_folder_name)

    ax_2 = plt.subplot(312)
    plt.plot(testing_raw_data)
    plt.grid(True)
    plt.title("The testing  signals in " + testing_dataset + "/" + testing_folder_name)

    ax_3= plt.subplot(313)
    #np.savetxt('s_threshold.out', np.amax(log_likelihood, axis=1)) # save the standard_thhreshold by training and testing the same successful trial.
    #plt.plot(np.loadtxt('Standard_threshold.out'), 'b^', linewidth=2, label = 'Standard threshold')
    for istate in range(log_likelihood.shape[1]):
        plt.plot(log_likelihood[:,istate], linewidth=3, label = label_str[istate])
        plt.grid(True)
        plt.title("Log-Likelihood")
    plt.legend(bbox_to_anchor = (0., 1.02, 1.,  .102), loc='lower right', frameon=True, shadow = True, ncol = log_likelihood.shape[1])
    plt.annotate('Hidden States:' + str(n_components) + ", "
                 + HMM_TYPE +'_cov=' + covariance_type + ", "
                 + "fixed transition matrix",
                 xy=(0, 0), xycoords='data',
                 xytext=(+10, -30), textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.savefig("fig_wrench_angles_state_"
                + str(n_components)
                + "_Covariance_" + covariance_type
                + "train_" + str(training_file_id)
                + "_test_" + str(testing_file_id)
                + ".eps",
                format="eps")
    plt.show()

