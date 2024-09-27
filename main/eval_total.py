import numpy as np
import os
import glob
import setting as st
import yaml
import cv2
import copy
import utility as util


if __name__ == "__main__":
        
    data_type_list = ([['']])
    #data_type_list.append(['_woconv_seq','_woconv_seq2','_woconv_seq3'])
    #data_type_list.append(['_woconv-history','_woconv-history2','_woconv-history3'])
    
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
    data_act_all = np.array(data_act_all)
    
    header = [' ','LLM_top1','LLM_top3','LLM_top5']
    header += ['Traj_top1_d>1m','Traj_top1_d>2m','Traj_top1_d>3m']
    header += ['Traj_top3_d>1m','Traj_top3_d>2m','Traj_top3_d>3m']
    header += ['Traj_top5_d>1m','Traj_top5_d>2m','Traj_top5_d>3m']
    header += ['Ours_top1_d>1m','Ours_top1_d>2m','Ours_top1_d>3m']
    header += ['Ours_top3_d>1m','Ours_top3_d>2m','Ours_top3_d>3m']
    header += ['Ours_top5_d>1m','Ours_top5_d>2m','Ours_top5_d>3m']
    header = np.array(header).reshape(1,-1)
    
    header2 = np.array(['Accuracy[%]']).reshape(1,1)
    
    a = data_all.reshape(1,-1)
    b = np.concatenate([header2,a],axis=1)
    c = np.concatenate([header,b],axis=0)
    
    np.savetxt("./Eval_total.csv", c, delimiter=',', fmt='%s')
    
    a = data_act_all.reshape(1,-1)
    b = np.concatenate([header2,a],axis=1)
    c = np.concatenate([header,b],axis=0)
    
    np.savetxt("./Eval_total_act.csv", c, delimiter=',', fmt='%s')