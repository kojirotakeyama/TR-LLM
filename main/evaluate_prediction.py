import setting as st
import utility as util
import prompt_manager as pt
import os
import pickle
from SIM_ACT import ACT
import csv
import copy
import numpy as np

def read_csv_as_list(csv_file_path):
    
    # Initialize lists to store the data of the second, third, fifth, and sixth columns
    rows = [1,2,3,4,6,7,8,9]
    out = [[] for _ in range(8)]
    
    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        
        # Skip the first three rows
        for _ in range(3):
            next(reader)
        
        # Iterate over the CSV rows
        for row in reader:
            
            for i in range(len(rows)):
                out[i].append(row[rows[i]])
            
    return out

def eval_pred(scenario_id):
    
    folder_path = './Result/scenario_id=' + str(f"{scenario_id:05}") 
    file_path = folder_path + '/prediction.csv'
    data_in = read_csv_as_list(file_path)
    data_out = copy.deepcopy(data_in)
    data_out.insert(2,[""]*len(data_out[0]))
    data_out.insert(5,[""]*len(data_out[0]))
    data_out.insert(8,[""]*len(data_out[0]))
    data_out.insert(11,[""]*len(data_out[0]))
    
    for i in range(len(data_in[0])):
        
        if data_in[0][i]!="" and data_in[1][i]!="":
            score = pt.get_similarity(data_in[0][i],data_in[1][i])
            data_out[2][i]=score
            print(data_in[0][i],data_in[1][i],score)
        
        if data_in[2][i]!="" and data_in[3][i]!="":
            score = pt.get_similarity(data_in[2][i],data_in[3][i])
            data_out[5][i]=score
            print(data_in[2][i],data_in[3][i],score)
        
        if data_in[4][i]!="" and data_in[5][i]!="":
            score = pt.get_similarity(data_in[4][i],data_in[5][i])
            data_out[8][i]=score
            print(data_in[4][i],data_in[5][i],score)
        
        if data_in[6][i]!="" and data_in[7][i]!="":
            score = pt.get_similarity(data_in[6][i],data_in[7][i])
            data_out[11][i]=score
            print(data_in[6][i],data_in[7][i],score)
    
    data_out[0].append("")
    data_out[1].append("")
    data_out[2].append(np.mean(np.array(data_out[2])[np.array(data_out[2])!=""].astype(np.float32)))
    data_out[3].append("")
    data_out[4].append("")
    data_out[5].append(np.mean(np.array(data_out[5])[np.array(data_out[5])!=""].astype(np.float32)))
    data_out[6].append("")
    data_out[7].append("")
    data_out[8].append(np.mean(np.array(data_out[8])[np.array(data_out[8])!=""].astype(np.float32)))
    data_out[9].append("")
    data_out[10].append("")
    data_out[11].append(np.mean(np.array(data_out[11])[np.array(data_out[11])!=""].astype(np.float32)))
    
    util.save_data_csv(np.array(data_out).transpose(),folder_path+"/score_similarity.csv")

def eval_pred2(scenario_id):
    
    folder_path = './Result/scenario_id=' + str(f"{scenario_id:05}") 
    file_path = folder_path + '/prediction2.csv'
    data_in = read_csv_as_list(file_path)
    data_out = copy.deepcopy(data_in)
    data_out.insert(2,[""]*len(data_out[0]))
    data_out.insert(5,[""]*len(data_out[0]))
    data_out.insert(8,[""]*len(data_out[0]))
    data_out.insert(11,[""]*len(data_out[0]))
    
    for i in range(len(data_in[0])):
        
        if data_in[0][i]!="" and data_in[1][i]!="":
            score = pt.get_similarity(data_in[0][i],data_in[1][i])
            data_out[2][i]=score
            print(data_in[0][i],data_in[1][i],score)
        
        if data_in[2][i]!="" and data_in[3][i]!="":
            score = pt.get_similarity(data_in[2][i],data_in[3][i])
            data_out[5][i]=score
            print(data_in[2][i],data_in[3][i],score)
    
    data_out[0].append("")
    data_out[1].append("")
    data_out[2].append(np.mean(np.array(data_out[2])[np.array(data_out[2])!=""].astype(np.float32)))
    data_out[3].append("")
    data_out[4].append("")
    data_out[5].append(np.mean(np.array(data_out[5])[np.array(data_out[5])!=""].astype(np.float32)))
    data_out[6].append("")
    data_out[7].append("")
    data_out[8].append("")
    data_out[9].append("")
    data_out[10].append("")
    data_out[11].append("")
    
    util.save_data_csv(np.array(data_out).transpose(),folder_path+"/score_similarity2.csv")
    
if __name__ == "__main__":
    
    for n in st.scenario_id:
        eval_pred(n)
    
    
    
    
