#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from hmmlearn.hmm import *
from sklearn.externals import joblib
import ipdb
from sklearn.preprocessing import scale
import warnings

warnings.filterwarnings("ignore", category= DeprecationWarning)
def matplot_list(list_data, figure_index, title,save=False,
                 label_string= ['Approach Motion',
                                'Rotation Motion',
                                'Insertion Motion',
                                'Mating Motion']):
    # if you want to save, title is necessary as a save name.
    
    global n_state
    global covariance_type_string
    plt.figure(figure_index, figsize=(40,30), dpi=80)
    ax = plt.subplot(111)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    plt.grid(True)
    i = 0
    for data in list_data:
        i = i + 1
        index = np.asarray(data).shape
        O = (np.arange(index[0])).tolist()
        plt.plot(O, data, label=label_string[i-1],linewidth='3.0')
    plt.legend(loc='lower left', frameon=True)

    plt.title(title)

    plt.annotate('State=4 Sub_State='+str(n_state)+' GaussianHMM_cov='+covariance_type_string,
             xy=(0, 0), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    if save:
        plt.savefig("Subtask1.eps", format="eps")


def scaling(X):
    _index, _column = X.shape
    Data_scaled = []
    scale_length = 10
    for i in range(scale_length, _index-scale_length-2):
        scaled = scale(X[i-scale_length:i+scale_length + 1, :])
        Data_scaled.append(scaled[scale_length,:])

    scaled_array = np.asarray(Data_scaled)
    return scaled_array


def load_data(path, path_index, preprocessing_scaling=True):
    
    df1 = pd.read_csv(path+'/R_Angles.dat', sep='\s+', header=None, skiprows=1)
    df2 = pd.read_csv(path+'/R_CartPos.dat', sep='\s+', header=None, skiprows=1)
    df3 = pd.read_csv(path+'/R_Torques.dat', sep='\s+', header=None, skiprows=1)
    df4 = pd.read_csv(path+'/worldforce-'+str(path_index)+".dat", sep='\s+', header=None, skiprows=1)
    df5 = pd.read_csv(path+'/R_State.dat', sep='\s+', header=None, skiprows=1)

    df1.columns = ['time','s0','s1','s2','s3','s4','s5']
    df2.columns = ['time','x','y','z','R','P','Y']
    df3.columns = ['time','Mx','My','Mz','MR','MP','MY']
    df4.columns = ['time','Fx','Fy','Fz','FR','FP','FY']

    df = pd.merge(df1,df2, how='outer', on='time')
    df = pd.merge(df,df3, how='outer', on='time')
    df = pd.merge(df,df4, how='outer',on='time')
    df = df.fillna(method='ffill')

    df5.columns=['time']
    df5['state']=[1,2,3]
    df5.ix[3] = [0.005,0]
    df = pd.merge(df,df5, how='outer', on='time')
    df = df.fillna(method='ffill')
 
    X_1 = df.values[df.values[:,-1] ==0]
    index_1,column_1 = X_1.shape
    X_2 = df.values[df.values[:,-1] ==1]
    index_2,column_2 = X_2.shape
    X_3 = df.values[df.values[:,-1] ==2]
    index_3,column_3 = X_3.shape
    X_4 = df.values[df.values[:,-1] ==3]
    index_4,column_4 = X_4.shape
    
    index = [index_1,index_2,index_3,index_4]

    X_1_ = X_1[:,1:]
    X_2_ = X_2[:,1:]
    X_3_ = X_3[:,1:]
    X_4_ = X_4[:,1:]
    
    Data = [X_1_,X_2_,X_3_,X_4_]

    if preprocessing_scaling:
        scaled_X_1 = scaling(Data[0])
        scaled_X_2 = scaling(Data[1])
        scaled_X_3 = scaling(Data[2])
        scaled_X_4 = scaling(Data[3])

        index_1, column_1 = scaled_X_1.shape
        index_2, column_2 = scaled_X_1.shape
        index_3, column_3 = scaled_X_1.shape
        index_4, column_4 = scaled_X_1.shape

        Data = []
        Data = [scaled_X_1, scaled_X_2, scaled_X_3, scaled_X_4]
        
        index = [index_1,index_2,index_3,index_4]

    return Data, index
    
def main():
    #ipdb.set_trace()

    global n_state
    global covariance_type_string
    
    n_state = 3

    covariance_type_string = 'diag'

    preprocessing_scaling = False

    train_path = "/home/birl/npBayesHMM/HIRO_SA_DATA/REAL_HIRO_ONE_SA_SUCCESS/20121127-HIROSA-S-02"

    test_path = "/home/birl/npBayesHMM/HIRO_SA_DATA/REAL_HIRO_ONE_SA_SUCCESS/20121127-HIROSA-S-06"

    train_path_index = 2

    test_path_index = 6


    #ipdb.set_trace()

    train_Data, train_index = load_data(path=train_path, path_index=train_path_index, preprocessing_scaling=False)

    My_Data = train_Data[0]
    My_Data = My_Data[:,:4]
    My_Data = My_Data.T
    My_Data_list = My_Data.tolist()
    
    matplot_list(My_Data_list, figure_index=7, title="train data set", save=True, label_string=['s0','s1','s2','s3'])

    train_Data, train_index = load_data(path=train_path, path_index=train_path_index, preprocessing_scaling=preprocessing_scaling)
    
    My_Data = train_Data[0]
    My_Data = My_Data[:,:4]
    My_Data = My_Data.T
    My_Data_list = My_Data.tolist()
    matplot_list(My_Data_list, figure_index=8, title="scaled train data set", save=True, label_string=['s0','s1','s2','s3'])
    
    
    # df1 = pd.read_csv('~/ML_data/REAL_HIRO_ONE_SA_SUCCESS/20121127-HIROSA-S-02/R_Angles.dat', sep='\s+', header=None, skiprows=1)
    # df2 = pd.read_csv('~/ML_data/REAL_HIRO_ONE_SA_SUCCESS/20121127-HIROSA-S-02/R_CartPos.dat', sep='\s+', header=None, skiprows=1)
    # df3 = pd.read_csv('~/ML_data/REAL_HIRO_ONE_SA_SUCCESS/20121127-HIROSA-S-02/R_Torques.dat', sep='\s+', header=None, skiprows=1)
    # df4 = pd.read_csv('~/ML_data/REAL_HIRO_ONE_SA_SUCCESS/20121127-HIROSA-S-02/worldforce-2.dat', sep='\s+', header=None, skiprows=1)
    # df5 = pd.read_csv('~/ML_data/REAL_HIRO_ONE_SA_SUCCESS/20121127-HIROSA-S-02/R_State.dat', sep='\s+', header=None, skiprows=1)

    # df1.columns = ['time','s0','s1','s2','s3','s4','s5']
    # df2.columns = ['time','x','y','z','R','P','Y']
    # df3.columns = ['time','Mx','My','Mz','MR','MP','MY']
    # df4.columns = ['time','Fx','Fy','Fz','FR','FP','FY']

    # df = pd.merge(df1,df2, how='outer', on='time')
    # df = pd.merge(df,df3, how='outer', on='time')
    # df = pd.merge(df,df4, how='outer',on='time')
    # df = df.fillna(method='ffill')

    # df5.columns=['time']
    # df5['state']=[1,2,3]
    # df5.ix[3] = [0.005,0]
    # df = pd.merge(df,df5, how='outer', on='time')
    # df = df.fillna(method='ffill')
 
    # X_1 = df.values[df.values[:,-1] ==0]
    # index_1,column_1 = X_1.shape
    # X_2 = df.values[df.values[:,-1] ==1]
    # index_2,column_2 = X_2.shape
    # X_3 = df.values[df.values[:,-1] ==2]
    # index_3,column_3 = X_3.shape
    # X_4 = df.values[df.values[:,-1] ==3]
    # index_4,column_4 = X_4.shape
    
    # index = [index_1,index_2,index_3,index_4]

    # X_1_ = X_1[:,1:]
    # X_2_ = X_2[:,1:]
    # X_3_ = X_3[:,1:]
    # X_4_ = X_4[:,1:]


    # Data = [X_1_,X_2_,X_3_,X_4_]

    # if preprocessing_scaling:
    #     scaled_X_1 = scaling(Data[0])
    #     scaled_X_2 = scaling(Data[1])
    #     scaled_X_3 = scaling(Data[2])
    #     scaled_X_4 = scaling(Data[3])

    #     index_1, column_1 = scaled_X_1.shape
    #     index_2, column_2 = scaled_X_1.shape
    #     index_3, column_3 = scaled_X_1.shape
    #     index_4, column_4 = scaled_X_1.shape

    #     train_Data = []
    #     train_Data = [scaled_X_1, scaled_X_2, scaled_X_3, scaled_X_4]

    start_prob = np.zeros(n_state)
    start_prob[0] = 1
    model_1 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                         params="mct", init_params="cmt", n_iter=1000)
    model_1.startprob_ = start_prob
    model_1 = model_1.fit(train_Data[0])

    model_2 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                         params="mct", init_params="cmt", n_iter=1000)
    model_2.startprob_ = start_prob
    model_2 = model_2.fit(train_Data[1])

    model_3 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                         params="mct", init_params="cmt", n_iter=1000)
    model_3.startprob_ = start_prob
    model_3 = model_3.fit(train_Data[2])

    model_4 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                         params="mct", init_params="cmt", n_iter=1000)
    model_4.startprob_ = start_prob
    model_4 = model_4.fit(train_Data[3])



    # save the models
    if not os.path.isdir(train_path+"/model/"):
        os.makedirs(train_path+"/model/")
    
    joblib.dump(model_1, train_path+"/model/model_s1.pkl")
    joblib.dump(model_2, train_path+"/model/model_s2.pkl")
    joblib.dump(model_3, train_path+"/model/model_s3.pkl")
    joblib.dump(model_4, train_path+"/model/model_s4.pkl")

    log_1_data = []
    log_2_data = []
    log_3_data = []
    log_4_data = []
    log_1_full_data = []
    log_2_full_data = []
    log_3_full_data = []
    log_4_full_data = []

    test_Data, test_index = load_data(path=test_path, path_index=test_path_index, preprocessing_scaling=preprocessing_scaling)
    
    ix_list = test_index
    for j in range(4): #nstate
        log_1 =[]
        log_2 =[]
        log_3 =[]
        log_4 =[]
        for i in range(1,ix_list[j]): #length
            log = model_1.score(test_Data[j][:i])
            log_1_full_data.append(log)
          #  ipdb.set_trace()
            
            log = model_2.score(test_Data[j][:i])
            log_2.append(log)
            log_2_full_data.append(log)
            
            log = model_3.score(test_Data[j][:i])
            log_3.append(log)
            log_3_full_data.append(log)
            
            log = model_4.score(test_Data[j][:i])
            log_4.append(log)
            log_4_full_data.append(log)
            
        log_1_data.append(log_1)
        log_2_data.append(log_2)
        log_3_data.append(log_3)
        log_4_data.append(log_4)

    subtask_log_data = []

    for i in range(4):
        subtask_log_data.append([log_1_data[i],log_2_data[i],log_3_data[i],log_4_data[i]])

    full_log_data = [log_1_full_data, log_2_full_data,log_3_full_data, log_4_full_data]

    if preprocessing_scaling:
        prefix_title = "scaled_data"
    else:
        prefix_title = "data"

    Subtask_name = ['Approach Motion',
                    'Rotation Motion',
                    'Insertion Motion',
                    'Maintain Motion']
        
    #plt.figure(1,figsize=(40,30), dpi=80)
    matplot_list(subtask_log_data[0], figure_index=1, title=Subtask_name[0]+" "+prefix_title+" "+"trained HMM Model S0~St log probability", save=True)
    #plt.savefig("Subtask1.eps", format="eps")

    #plt.figure(2,figsize=(40,30), dpi=80)
    #plt.title(Subtask_name[1]+" "+prefix_title+" "+"trained HMM Model S0~St log probability ")
    matplot_list(subtask_log_data[1], figure_index=2, title=Subtask_name[1]+" "+prefix_title+" "+"trained HM Model S0~St log probability", save=True)
    #plt.savefig("Subtask2.eps", format="eps")

    #plt.figure(3,figsize=(40,30), dpi=80)
    #plt.title(Subtask_name[2]+" "+prefix_title+" "+"trained HMM Model S0~St log probability ")
    matplot_list(subtask_log_data[2], figure_index=3, title=Subtask_name[2]+" "+prefix_title+" "+"trained HMM Model S0~St log probability", save=True)
    #plt.savefig("Subtask3.eps", format="eps")    

    #plt.figure(4,figsize=(40,30), dpi=80)
    #plt.title(Subtask_name[3]+" "+prefix_title+" "+"trained HMM Model S0~St log probability ")
    matplot_list(subtask_log_data[3], figure_index=4, title=Subtask_name[3]+" "+prefix_title+" "+"trained HMM Model S0~St log probability", save=True)
    #plt.savefig("Subtask4.eps", format="eps")


    
    # ax = plt.subplot(111)
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.spines['bottom'].set_position(('data',0))
    # ax.yaxis.set_ticks_position('left')
    # ax.spines['left'].set_position(('data',0))
    #plt.title("Full scaledData HMM Model S0~St log probability")
    #plt.grid(True)
    #matplot_list(full_log_data)
    matplot_list(full_log_data, figure_index=5, title=prefix_title+"Full scaledData HMM Model S0~St log probability", save=False)
    index_cum = 2
    for i in range(4):
        index_cum = ix_list[i] + index_cum -1
        plt.plot([index_cum, index_cum], [-1*10**4, 5*10], color ='grey', linewidth=2,linestyle="--")
        print "subtask data index:%d"%index_cum
    plt.savefig("Full_sum_log.eps", format="eps")

    
   #  plt.figure(2,figsize=(40,30), dpi=100)
    
   #  #plt.subplot(2,1,2)
   #  plt.title("Subtask 2 -trained HMM Model log-function ")
   #  # Max = max(max(log_1),max(log_2),max(log_3),max(log_4),max(log_5))
   #  #  Min = min(min(log_1),min(log_2),min(log_3),min(log_4),min(log_5))
   #  #plt.ylim(Min*1.1, Max*1.1)
   #  plt.plot(O, log_1_data[1][:], label="Subtask 1",linewidth='3.0')
   #  plt.plot(O, log_2_data[1][:], label="Subtask 2",linewidth='3.0')
   #  plt.plot(O, log_3_data[1][:], label="Subtask 3",linewidth='3.0')
   #  plt.plot(O, log_4_data[1][:], label="Subtask 4",linewidth='3.0')
   #          # plt5 = plt.plot(O, log_5, label="Subtask 5",linewidth='3.0')
   #  plt.legend(loc='lower left', frameon=True)
   #  plt.savefig("Subtask2.eps", format="eps")
    
   #  plt.figure(3,figsize=(40,30), dpi=80)
   # # plt.subplot(2,1,3)
   #  plt.title("Subtask 3 -trained HMM Model log-function ")
   #  # Max = max(max(log_1),max(log_2),max(log_3),max(log_4),max(log_5))
   #  #  Min = min(min(log_1),min(log_2),min(log_3),min(log_4),min(log_5))
   #  #plt.ylim(Min*1.1, Max*1.1)
   #  plt.plot(O, log_1_data[2][:], label="Subtask 1",linewidth='3.0')
   #  plt.plot(O, log_2_data[2][:], label="Subtask 2",linewidth='3.0')
   #  plt.plot(O, log_3_data[2][:], label="Subtask 3",linewidth='3.0')
   #  plt.plot(O, log_4_data[2][:], label="Subtask 4",linewidth='3.0')
   #          # plt5 = plt.plot(O, log_5, label="Subtask 5",linewidth='3.0')
   #  plt.legend(loc='lower left', frameon=True)
   #  plt.savefig("Subtask3.eps", format="eps")
    
   #  plt.figure(4,figsize=(40,30), dpi=80)
   #  #plt.subplot(2,1,4)
   #  plt.title("Subtask 4 -trained HMM Model log-function ")
   #  # Max = max(max(log_1),max(log_2),max(log_3),max(log_4),max(log_5))
   #  #  Min = min(min(log_1),min(log_2),min(log_3),min(log_4),min(log_5))
   #  #plt.ylim(Min*1.1, Max*1.1)
   #  plt.plot(O, log_1_data[3][:], label="Subtask 1",linewidth='3.0')
   #  plt.plot(O, log_2_data[3][:], label="Subtask 2",linewidth='3.0')
   #  plt.plot(O, log_3_data[3][:], label="Subtask 3",linewidth='3.0')
   #  plt.plot(O, log_4_data[3][:], label="Subtask 4",linewidth='3.0')
   #          # plt5 = plt.plot(O, log_5, label="Subtask 5",linewidth='3.0')
   #  plt.legend(loc='lower left', frameon=True)
   #  plt.savefig("Subtask4.eps", format="eps")


   #  O1 = np.arange()

   #  plt.figure(4,figsize=(40,30), dpi=80)
   #  #plt.subplot(2,1,4)
   #  plt.title("Subtask 4 -trained HMM Model log-function ")
   #  # Max = max(max(log_1),max(log_2),max(log_3),max(log_4),max(log_5))
   #  #  Min = min(min(log_1),min(log_2),min(log_3),min(log_4),min(log_5))
   #  #plt.ylim(Min*1.1, Max*1.1)
   #  plt.plot(O, log_1_data[3][:], label="Subtask 1",linewidth='3.0')
   #  plt.plot(O, log_2_data[3][:], label="Subtask 2",linewidth='3.0')
   #  plt.plot(O, log_3_data[3][:], label="Subtask 3",linewidth='3.0')
   #  plt.plot(O, log_4_data[3][:], label="Subtask 4",linewidth='3.0')
   #          # plt5 = plt.plot(O, log_5, label="Subtask 5",linewidth='3.0')
   #  plt.legend(loc='lower left', frameon=True)
   #  plt.savefig("Subtask4.eps", format="eps")

    
    plt.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())
