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
training_file_id     = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'] #only one trial

rootDATApath          = '/home/birl/npBayesHMM/HIRO_SA_DATA/'
trained_models_path   = '/home/birl/npBayesHMM/python/state_estimation/trained_models'
dataType              = ['R_Torques.dat','R_Angles.dat'] #'R_Angles.dat','R_CartPos.dat' tuples can not edited
time_step             = 0.005
n_components          = 25
covariance_type       = "diag"
Niter                 = 2000
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
    return  dat_path , sensor.transpose().values, scaling(sensor.transpose().values)

def training(training_dataPath, training_file_id):
    training_folder_name, training_raw_data, training_sensor = load_one_trial(training_dataPath, training_file_id)

    print "Training the normal execution"

    train_model = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                              init_params="cmt",  params="cmt", n_iter=Niter)
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

if __name__ == '__main__':
    # for training multiple trials
    for file_id in training_file_id:
        training_folder_name, training_raw_data, training_sensor = training(rootDATApath + training_dataset, file_id)



