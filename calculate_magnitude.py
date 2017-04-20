#!/usr/bin/env python -W ignore::DeprecationWarning
# @ Hongmin Wu
# April 16, 2017
import sys
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
from hmmlearn.hmm import *
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
import warnings
import shutil
import ipdb
from time import sleep


warnings.filterwarnings("ignore", category= DeprecationWarning)

rootDATApath          = '/home/birl/npBayesHMM/HIRO_SA_DATA/'
dataset      = 'SIM_HIRO_ONE_SA_SUCCESS'
dataType              = ['R_Torques.dat']

def magnitude(x): #vector x
    return math.sqrt(sum(i**2 for i in x))

def load_one_trial(dataPath):
    folders           = [d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath,d))]
    nfolder = 0
    while (nfolder < len(folders)):
        for dat_file in os.listdir(os.path.join(dataPath,folders[nfolder])):
            if dat_file in dataType:
                raw_data  = pd.read_csv(os.path.join(dataPath,folders[nfolder])+'/' + dat_file, sep='\s+', header=None, skiprows=0, usecols = range(1,7))
                sensor = raw_data.values
                mag = np.zeros(sensor.shape[0])
                for idx in range(0, sensor.shape[0]):
                    mag[idx] = magnitude(sensor[idx,:])
                np.savetxt(dataPath + "/" + folders[nfolder]+ "/FT_magnitude.dat",mag)
                plt.plot(mag,label = folders[nfolder])
                plt.legend(bbox_to_anchor = (0., 1.0), loc='best', frameon=True, shadow = True, ncol = 1)
                plt.show()
        nfolder = nfolder + 1
if __name__ == '__main__':
    load_one_trial(rootDATApath + dataset)
    raw_input("Press ENTER to testing the trail from different experiment: ")
