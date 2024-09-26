import numpy as np
import os
import glob
import setting as st
import yaml
import cv2
import copy
import utility as util
import prompt_manager as pt

def exclude_items(items):
        
    items_out = copy.deepcopy(items)
    for obs in list(items.keys()):
        if obs.endswith(" "):
                
            if not obs.rstrip() in st.list_obj_exclude:
                items_out[obs.rstrip()] = copy.deepcopy(items[obs])
            del items_out[obs]
        elif obs in st.list_obj_exclude:
            del items_out[obs]
        
    return items_out
  
def save_prob_obj_llm(data_llm, scene_id, act_id, n_act, prob_scale, flag_save):
    
    obj_state = {}
    obj_state['obj_list'] = exclude_items(util.get_sem_pos(scene_id))
    obj_candidate = list(obj_state['obj_list'].keys())
        
    #map_texture = cv2.imread(st.path_texture_map + f'{scene_id:03}'+'.png')
    map_binary = np.repeat(cv2.imread(st.path_binary_map + f'{scene_id:03}' + '.png')[:,:,0:1],3,2)
    map_binary[np.where(map_binary>0)]=100
        
    dir_name = './Eval' + data_type + '/pred_obj_llm_{:0=3}'.format(scene_id)
    file_name = '/map_predobj_'+ f'{scene_id:03}_' + f'{act_id:03}_' + f'{n_act:03}' + '.png'
     
    obj_prob = {}
    for i in range(len(obj_candidate)):
        obj_prob[obj_candidate[i]] = []
        for j in range(len(obj_state['obj_list'][obj_candidate[i]])):
            obj_prob[obj_candidate[i]].append(data_llm[i])
        obj_prob[obj_candidate[i]] = np.array(obj_prob[obj_candidate[i]])
    util.disp_prob_obj(map_in=map_binary, obj_candidate=obj_candidate, obj_prob=obj_prob, obj_list=obj_state['obj_list'], dir_name=dir_name, file_name=file_name, prob_scale=prob_scale, flag_save=1)
  
def save_prob_obj_ours(data_llm, data_traj, scene_id, act_id, n_act, prob_scale, flag_save):
    
    obj_state = {}
    obj_state['obj_list'] = exclude_items(util.get_sem_pos(scene_id))
    obj_candidate = list(obj_state['obj_list'].keys())
    
    #map_texture = cv2.imread(st.path_texture_map + f'{scene_id:03}'+'.png')
    
    for t in range(data_traj.shape[1]):
        
        path_map_traj = './Result' + data_type + '/TRAJ/pred_obj_traj_' + f'{scene_id:03}/' + 'map_predobj_'+ f'{scene_id:03}_' + f'{act_id:03}_' + f'{n_act:03}_' + f'{t:03}' + '.png'
        if not os.path.exists(path_map_traj):
            continue
        
        map_traj = cv2.imread(path_map_traj)
        map_traj = cv2.resize(map_traj, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
        
        dir_name = './Eval' + data_type + '/pred_obj_ours_{:0=3}'.format(scene_id)
        file_name = '/map_predobj_'+ f'{scene_id:03}_' + f'{act_id:03}_' + f'{n_act:03}_' + f'{t:03}' + '.png'
         
        obj_prob = {}
        idx = 0
        for i in range(len(obj_candidate)):
            obj_prob[obj_candidate[i]] = []
            for j in range(len(obj_state['obj_list'][obj_candidate[i]])):
                obj_prob[obj_candidate[i]].append(data_llm[i]*data_traj[idx,t])
                idx += 1
            obj_prob[obj_candidate[i]] = np.array(obj_prob[obj_candidate[i]])
        util.disp_prob_obj(map_in=map_traj, obj_candidate=obj_candidate, obj_prob=obj_prob, obj_list=obj_state['obj_list'], dir_name=dir_name, file_name=file_name, prob_scale=prob_scale, flag_save=1)

def modify_target_obj(t_obj, ref_obj):
    
    with open(st.path_act + 'object_flexibility.yaml') as file:
        obj_flexibility = yaml.load(file, Loader=yaml.FullLoader)['object flexibility']

    target_obj = obj_flexibility[t_obj]
    target_obj_out = []
        
    for obj in target_obj:
        if obj in ref_obj:
            target_obj_out = obj
            break
    
    #print(t_obj, target_obj_out)
    return target_obj_out

if __name__ == "__main__":
        
    data_type_list = ['_all','_woconv','_woconv-history']
                      
    flag_viz_llm = 0
    flag_viz_ours = 0
        
    gt = [[] for _ in range(20)]
    for i in range(20):
        gt[i] = [[] for _ in range(20)]
    
    gt[0][1] = 'coffee table'
    gt[0][3] = 'fridge'
    gt[1][3] = 'fridge'
    gt[1][9] = 'dining table'
    gt[2][1] = 'table'
    gt[2][3] = 'sink'
    gt[3][1] = 'table'
    gt[3][3] = 'couch'
    gt[4][1] = 'cupboard'
    gt[4][3] = 'sink'
    gt[4][5] = 'stove'
    gt[5][1] = 'sink'
    gt[5][4] = 'dining table'
    gt[6][1] = 'curtain'
    gt[6][3] = 'sink'
    gt[6][5] = 'plant'
    gt[7][1] = 'dining table'
    gt[7][3] = 'dishwasher'
    gt[8][2] = 'trash bin'
    gt[9][1] = 'cupboard'
    gt[9][3] = 'book'
    gt[9][5] = 'couch'
    
    for data_type in data_type_list:
        
        score = np.zeros([20,20])-1
    
        base_dir = './Result' + data_type + '/'
        folder_path_llm = base_dir + '/LLM/'
        folder_path_eval = './Eval' + data_type + '/'
        if not os.path.exists(folder_path_eval):
            os.mkdir(folder_path_eval)
        
        f_llm = os.path.join(folder_path_llm, f"*{'llm'}*")
        f_llm2 = os.path.join(folder_path_llm, f"*{'llm_input'}*")
        
        # List all files matching the pattern
        fname_llm_ = sorted(glob.glob(f_llm))
        fname_llm2 = sorted(glob.glob(f_llm2))
        fname_llm = [item for item in fname_llm_ if item not in fname_llm2]
              
        count = 0
        for fp_llm in fname_llm:
            
            scene_id = int(fp_llm[-15:-12])
            situation_id = int(fp_llm[-11:-8])
            act_id = int(fp_llm[-7:-4])
            
            if score[situation_id,act_id] != -1:
                continue
            else:
                score[situation_id,act_id] = 0
            
            with open(fname_llm2[count], 'r') as file:
                # Read the content of the file
                prompt = file.read()
            count += 1
            
            with open(st.path_situation + 's_{:0=5}'.format(situation_id) + '.yaml','rb') as file:
                s= yaml.load(file, Loader=yaml.FullLoader)
            
            with open(st.path_act + 'act_{:0=3}'.format(situation_id) + '.yaml') as file:
                a = yaml.load(file, Loader=yaml.FullLoader)
                gt_act = a['P1']['action'][act_id+1]
            
            if a['object states'][act_id] != None:
                a['object states'][act_id] = a['object states'][act_id].replace('\"P1\"',s['P1']['name'])
                        
                prompt = prompt + "\n[additional information]\n" + a['object states'][act_id]
            
            gt_obj = gt[situation_id][act_id]
            
            for i in range(10):
                act_pred = pt.pred_act_based_on_obj(prompt, gt_obj, s['P1']['name'])['content']['action']
                score[situation_id,act_id] += int(pt.eval_score(act_pred, gt_act)['content']['score'])
                print(act_pred, gt_act, score[situation_id,act_id])
            score[situation_id,act_id] = score[situation_id,act_id]/10
            
            #np.savetxt("./pred_act/pred_act_{:0=3}".format(situation_id) + "_{:0=3}".format(act_id) + ".csv", act_pred, delimiter=',', fmt='%s')
        print(score)
        np.savetxt("./pred_act/pred_act_score" + data_type + ".csv", score, delimiter=',', fmt='%s')
        np.save("./pred_act/pred_act_score" + data_type + ".npy", score)
            
            
            
