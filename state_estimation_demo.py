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

warnings.filterwarnings("ignore", category= DeprecationWarning)

trained_models_path   = '/home/birl/npBayesHMM/python/state_estimation/learned_models'
training_dataPath        = '/home/birl/npBayesHMM/HIRO_SA_DATA/REAL_HIRO_ONE_SA_SUCCESS'
training_file_id  = 5

testing_dataPath        = '/home/birl/npBayesHMM/HIRO_SA_DATA/REAL_HIRO_ONE_SA_FAILURE'
testing_file_id = 3

dataType        = ['R_Torques.dat'] #,'R_State.dat' ,'R_Angles.dat','R_CartPos.dat' tuples can not edited
label_str       = ['approach', 'rotation', 'insertion', 'mating']
time_step       = 0.005
n_components    = 6
covariance_type = "diag"
Niter           = 1000
scale_length = 10

def scaling(X):
    _index, _column = X.shape
    Data_scaled = []
    scale_length = 10
    for i in range(scale_length, _index-scale_length-2):
        scaled = preprocessing.scale(X[i-scale_length:i+scale_length + 1, :])
        Data_scaled.append(scaled[scale_length,:])

    scaled_array = np.asarray(Data_scaled)
    return scaled_array

def load_one_trial(dataPath,id):
    sensor = pd.DataFrame()
    Rstate = pd.DataFrame()
    folders = [d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath,d))]
    for dat_file in os.listdir(os.path.join(dataPath,folders[id])):
        if dat_file in dataType:
            raw_data = pd.read_csv(os.path.join(dataPath,folders[id])+'/' + dat_file, sep='\s+', header=None, skiprows=1, usecols = range(1,7))
            sensor = raw_data.transpose().append(sensor) # transpose to [ndim x length]
        elif dat_file == 'R_State.dat':
            Rstate = pd.read_csv(os.path.join(dataPath, folders[id]) + '/' + dat_file, sep='\s+', header=None, skiprows=1)
            Rstate = Rstate.values / time_step
            Rstate = np.insert(Rstate, 0, 0, axis= 0)
            Rstate = np.append(Rstate, sensor.shape[1])
    return  raw_data.values, scaling(sensor.transpose().values), Rstate

def training(training_dataPath, training_file_id):
    if os.path.isdir(trained_models_path ): # if exit
        shutil.rmtree(trained_models_path, ignore_errors= False, onerror= None)
    training_raw_data, training_sensor, Rstate = load_one_trial(training_dataPath, training_file_id)
    for istate in range(len(Rstate) - 1):
        start_prob = np.zeros(n_components)
        start_prob[0] = 1
        train_model = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                              params="mct", init_params="cmt", n_iter=Niter)
        train_model.startprob_ = start_prob
        train_model = train_model.fit(training_sensor[Rstate[istate]:Rstate[istate + 1],:])
        #save the models
        if not os.path.isdir(trained_models_path ):
            os.makedirs(trained_models_path )
        joblib.dump(train_model, trained_models_path  + "/state_" + str(istate)+".pkl")
    return training_raw_data, training_sensor

def testing(testing_dataPath, testing_file_id):
    #load testing file
    testing_raw_data, testing_sensor, Rstate = load_one_trial(testing_dataPath, testing_file_id)

    #load trained model
    folders = os.listdir(trained_models_path)
    log_likelihood = np.zeros((len(testing_sensor),len(folders)))
    obs = []
    for obs_idx in range(1, len(testing_sensor)):
        print "testing: " + str(obs_idx) + "/" + str(len(testing_sensor))
        if obs_idx in Rstate:
            obs = []
        obs.append(testing_sensor[obs_idx, :])
        for imodel in range(len(folders)):
            train_model = joblib.load(os.path.join(trained_models_path, "state_"+str(imodel)+".pkl"))
            log_likelihood[obs_idx, imodel] = train_model.score(obs)
    return testing_raw_data, testing_sensor, log_likelihood
if __name__ == '__main__':
    # for training
    training_raw_data, training_sensor = training(training_dataPath, training_file_id)

    # for testing
    testing_raw_data, testing_sensor, log_likelihood  = testing(testing_dataPath, testing_file_id)
    plt.figure(1)
    ax_1 = plt.subplot(411)
    plt.plot(training_raw_data)
    plt.grid(True)
    plt.title("The training signals")

    ax_2 = plt.subplot(412)
    plt.plot(testing_raw_data)
    plt.grid(True)
    plt.title("The testing  signals")

    ax_3 = plt.subplot(413)
    plt.plot(testing_sensor)
    plt.grid(True)
    plt.title("The testing  signals after scaling preprocessing")

    ax_4= plt.subplot(414)
    for istate in range(log_likelihood.shape[1]):
        plt.plot(log_likelihood[:,istate], linewidth=3, label = label_str[istate])
        plt.grid(True)
        plt.title("Log-Likelihood")
    plt.legend(loc='lower left', frameon=True)
    plt.savefig("scale_preprocessing.eps", format="eps")
    plt.show()

