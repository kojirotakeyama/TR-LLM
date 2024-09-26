import numpy as np
import os
import glob
import setting as st
import yaml
import cv2
import copy
import utility as util


if __name__ == "__main__":
        
    data_type_list = ([['_all_seq','_all_seq2','_all_seq3']])
    data_type_list.append(['_woconv_seq','_woconv_seq2','_woconv_seq3'])
    data_type_list.append(['_woconv-history','_woconv-history2','_woconv-history3'])
    
    data = []
    data_all = []
    data_mean = []
    data_act = []
    data_act_all = []
    data_act_mean = []
    
    for data_type_list_sub in data_type_list:
        for data_type in data_type_list_sub:
            data.append(np.genfromtxt('Eval'+data_type+'/res_eval_pct.csv', delimiter=',')[0])
            data_all.append(np.genfromtxt('Eval'+data_type+'/res_eval_pct.csv', delimiter=',')[0])
            data_act.append(np.genfromtxt('Eval'+data_type+'/res_eval_pct_act.csv', delimiter=',')[0])
            data_act_all.append(np.genfromtxt('Eval'+data_type+'/res_eval_pct_act.csv', delimiter=',')[0])
            
        data_mean.append(np.mean(np.array(data),axis=0))
        data_act_mean.append(np.mean(np.array(data_act),axis=0))
        data = []
        data_act = []
    
    data_all = np.array(data_all)
    data_mean = np.array(data_mean)
    data_act_all = np.array(data_act_all)
    data_act_mean = np.array(data_act_mean)
    
    header = np.array(sum(data_type_list, []) + ['all_seq_mean','woconv_seq_mean','woconv-history_mean'])
    
    a = np.concatenate([data_all,data_mean],axis=0)
    b = np.concatenate([header.reshape(-1,1),a],axis=1)
    
    np.savetxt("./Eval_total.csv", b, delimiter=',', fmt='%s')
    
    a = np.concatenate([data_act_all,data_act_mean],axis=0)
    b = np.concatenate([header.reshape(-1,1),a],axis=1)
    
    np.savetxt("./Eval_total_act.csv", b, delimiter=',', fmt='%s')