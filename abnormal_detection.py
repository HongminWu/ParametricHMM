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

training_dataset      = 'REAL_HIRO_ONE_SA_SUCCESS'
testing_dataset       = 'REAL_HIRO_ONE_SA_SUCCESS' #REAL_HIRO_ONE_SA_ERROR_CHARAC
#training_file_id     = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'] #only one trial
training_file_id      = '03'
testing_file_id       = '03'                       #only one trial
model_id              = '03'                  #define the trained_model

rootDATApath          = '/home/birl/npBayesHMM/HIRO_SA_DATA/'
trained_models_path   = '/home/birl/npBayesHMM/python/state_estimation/trained_models'
dataType              = ['R_Torques.dat','R_Angles.dat'] #'R_Angles.dat','R_CartPos.dat' tuples can not edited
time_step             = 0.005
n_components          = 20
covariance_type       = "diag"
Niter                 = 1000
scale_length          = 5

def fixed_transMatrix():
    self_trans = 0.5
    transMatrix = np.identity(n_components) * self_trans
    transMatrix[- 1, - 1] = 1
    for i in np.arange(n_components - 1):
        for j in np.arange(n_components - i - 2):
            transMatrix[i, i + j + 1] = 1. / (n_components - 1. - i) - j / ((n_components - 1. - i) * (n_components - 2. - i))
    transMatrix[n_components - 2, -1] = self_trans
    return transMatrix

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
    return  "20121127-HIROSA-S-" + id, sensor.transpose().values, scaling(sensor.transpose().values)

def training(training_dataPath, training_file_id):
    training_folder_name, training_raw_data, training_sensor = load_one_trial(training_dataPath, training_file_id)

    print "Training the normal execution"

    train_model = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                              init_params="cm",  params="cmt", n_iter=Niter)
    start_prob = np.zeros(n_components)
    start_prob[0] = 1
    train_model.startprob_ = start_prob
    train_model.transmat_  = fixed_transMatrix()

    train_model = train_model.fit(training_sensor)
    #save the models
    if not os.path.isdir(trained_models_path ):
        os.makedirs(trained_models_path )
    joblib.dump(train_model, trained_models_path  + "/tainedModel_" + training_file_id +".pkl")
    print "..."
    return training_folder_name, training_raw_data, training_sensor

def testing(testing_dataPath, testing_file_id, model_id):
    #load testing file
    testing_folder_name, testing_raw_data, testing_sensor= load_one_trial(testing_dataPath, testing_file_id)

    #load trained model
    train_model = joblib.load(os.path.join(trained_models_path, "tainedModel_" + model_id + ".pkl"))
    #incremental testing
    log_likelihood = np.zeros((len(testing_sensor)))
    obs = []
    for obs_idx in range(0, len(testing_sensor)):
        print "testing: " + str(obs_idx) + "/" + str(len(testing_sensor))
        obs.append(testing_sensor[obs_idx, :])
        log_likelihood[obs_idx] = train_model.score(obs)
        # logprob, state_sequence =train_model.decode(obs) #find most likely state sequence corresponding to obs
        # print logprob, state_sequence
    return testing_folder_name, testing_raw_data, testing_sensor, log_likelihood

if __name__ == '__main__':
    # for training
    # for file_id in training_file_id:
    #     training_folder_name, training_raw_data, training_sensor = training(rootDATApath + training_dataset, file_id)

    # for testing
    testing_folder_name, testing_raw_data, testing_sensor, log_likelihood  = testing(rootDATApath + testing_dataset, testing_file_id, model_id)
    training_folder_name, training_raw_data, training_sensor = load_one_trial(rootDATApath + training_dataset, model_id) #only for plot

    fig_1  = plt.figure()
    ax_1 = fig_1.add_subplot(311)
    plt.plot(training_raw_data)
    plt.grid(True)
    plt.title("The training signals in " + training_dataset + "/" + training_folder_name)
    ax_2 = fig_1.add_subplot(312)
    plt.plot(testing_raw_data)
    plt.grid(True)
    plt.title("The testing  signals in " + testing_dataset + "/" + testing_folder_name)

    ax_3 = fig_1.add_subplot(313)
    # load trained model
    plt.grid(True)
    plt.title("Log-Likelihood")
    train_model = joblib.load(os.path.join(trained_models_path, "tainedModel_" + model_id + ".pkl"))
    if testing_file_id == training_file_id: ## save the standard_thhreshold by training and testing the same successful trial.
        np.savetxt(                    trained_models_path + '/tainedModel_threshold_' +model_id+'.out', log_likelihood)
    tainedModel_threshold = np.loadtxt(trained_models_path + '/tainedModel_threshold_' + model_id + '.out')
    plt.plot(tainedModel_threshold, 'r--', linewidth=2, label='expected_likelihood')
    plt.plot(log_likelihood, 'b-', linewidth=3, label="obs_likelihood")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right', frameon=True, shadow=True, ncol=1)
    plt.savefig("fig_wrench_angles_state_"
                + str(n_components)
                + "_Covariance_" + covariance_type
                + "_Transition_fixed_"
                + "train_" + training_file_id
                + "_test_" + testing_file_id
                + ".eps",
                format="eps")
    plt.annotate('Hidden States:' + str(n_components) + ", "
                 + ' GaussianHMM_cov=' + covariance_type + ", "
                 + "fixed transition matrix",
                 xy=(0, 0), xycoords='data',
                 xytext=(+10, -30), textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.show()




