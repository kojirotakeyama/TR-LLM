import numpy as np
import os
import glob
import setting as st
import yaml
import cv2
import copy
import utility as util

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
        
    data_type_list = ['']
                      
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
        
        pred_score = np.load('./pred_act/pred_act_score_all.npy')
    
        base_dir = './Result' + data_type + '/'
        folder_path_llm = base_dir + '/LLM/'
        folder_path_traj = base_dir + '/TRAJ/'
        folder_path_eval = './Eval' + data_type + '/'
        if not os.path.exists(folder_path_eval):
            os.mkdir(folder_path_eval)
        
        f_llm = os.path.join(folder_path_llm, f"*{'llm'}*")
        f_llm2 = os.path.join(folder_path_llm, f"*{'llm_input'}*")
        f_traj = os.path.join(folder_path_traj, f"*{'traj'}*")
        
        # List all files matching the pattern
        fname_llm_ = sorted(glob.glob(f_llm))
        fname_llm2 = sorted(glob.glob(f_llm2))
        fname_llm = [item for item in fname_llm_ if item not in fname_llm2]
        
        fname_traj = sorted(glob.glob(f_traj))
        
        
        gt_order_llm = []
        gt_order_traj = []
        gt_order_ours = []
        gt_rel_llm = []
        gt_rel_traj = []
        gt_rel_ours = []
        
        acc_top1_traj = [[],[],[]]
        acc_top3_traj = [[],[],[]]
        acc_top5_traj = [[],[],[]]
        acc_top1_ours = [[],[],[]]
        acc_top3_ours = [[],[],[]]
        acc_top5_ours = [[],[],[]]
        acc_top1_llm = []
        acc_top3_llm = []
        acc_top5_llm = []
        acc_all_traj = [[],[],[]]
        acc_all_ours = [[],[],[]]
        acc_all_llm = []
        
        rel_top1_traj = [[],[],[]]
        rel_top3_traj = [[],[],[]]
        rel_top5_traj = [[],[],[]]
        rel_top1_ours = [[],[],[]]
        rel_top3_ours = [[],[],[]]
        rel_top5_ours = [[],[],[]]
        rel_top1_llm = []
        rel_top3_llm = []
        rel_top5_llm = []
        rel_all_traj = [[],[],[]]
        rel_all_ours = [[],[],[]]
        rel_all_llm = []
        
        for fp_llm in fname_llm:
            
            fp_traj = glob.glob(folder_path_traj+'/result_traj' + fp_llm[-16:])
            fp_traj2 = glob.glob(folder_path_traj+'/traj' + fp_llm[-16:])
            
            scene_id = int(fp_llm[-15:-12])
            situation_id = int(fp_llm[-11:-8])
            act_id = int(fp_llm[-7:-4])
            
            if len(fp_traj)==0:
                continue
            else:
                fp_traj = fp_traj[0]
                fp_traj2 = fp_traj2[0]
            
            # Read the header
            with open(fp_llm, 'r') as f:
                header = f.readline().strip().split(',')
            with open(fp_traj, 'r') as f:
                header2 = f.readline().strip().split(',')
            
            # Read the data, skipping the header row
            data_llm = np.genfromtxt(fp_llm, delimiter=',', skip_header=1)
            data_traj = np.genfromtxt(fp_traj, delimiter=',', skip_header=1).transpose()
            data_traj2 = np.genfromtxt(fp_traj2, delimiter=',').transpose()
            
            data_ours = np.zeros(data_traj.shape)
            for i in range(data_traj.shape[0]):
                obj_idx = header.index(header2[i])
                for j in range(data_traj.shape[1]):
                    data_ours[i,j] = data_traj[i,j]*data_llm[obj_idx]
            
            d = np.sqrt(np.sum((data_traj2[0:2,:] - data_traj2[2:4,:])**2, axis=0))/128*10
            
            if np.min(d)<0.5:
                s_idx = np.min(np.where(d<0.5))
            else:
                s_idx = d.shape[0]
            
            data_traj = data_traj[:,:s_idx]
            
            dpos_ = np.sqrt(np.sum((data_traj2[0:2,1:] - data_traj2[0:2,:-1])**2, axis=0))/128*10
            dpos = np.zeros(data_traj2.shape[1])
            for i in range(dpos_.shape[0]):
                dpos[i+1] = dpos[i] + dpos_[i]
            
            t_obj = gt[int(fp_traj[-11:-8])][int(fp_traj[-7:-4])]
            if t_obj == []:
                continue
            obj_gt = modify_target_obj(t_obj, header)
                
            if obj_gt != []:
                gt_idx = header.index(obj_gt)
                gt_idx2 = np.array([i for i, x in enumerate(header2) if x == obj_gt])
            else:
                continue
            
            if flag_viz_llm==1:
                save_prob_obj_llm(data_llm, int(fp_llm[-15:-12]), int(fp_traj[-11:-8]), int(fp_traj[-7:-4]), prob_scale=20, flag_save=1)
            if flag_viz_ours==1:
                save_prob_obj_ours(data_llm, data_traj, int(fp_llm[-15:-12]), int(fp_traj[-11:-8]), int(fp_traj[-7:-4]), prob_scale=10, flag_save=1)
        
            if np.max(dpos)>1.0:
                d1 = np.min(np.where(dpos>1.0))
            else:
                d1 = data_traj.shape[1]
            
            if np.max(dpos)>2.0:
                d2 = np.min(np.where(dpos>2.0))
            else:
                d2 = data_traj.shape[1]
            
            if np.max(dpos)>3.0:
                d3 = np.min(np.where(dpos>3.0))
            else:
                d3 = data_traj.shape[1]
            
            order_llm = np.argsort(-data_llm)
            order_traj = np.argsort(-data_traj, axis=0)
            order_ours = np.argsort(-data_ours, axis=0)
            
            w = pred_score[situation_id,act_id]
            
            rel_llm = data_llm/np.sum(data_llm)
            rel_traj = (data_traj/np.sum(data_traj))
            rel_ours = (data_ours/np.sum(data_ours))
            
            rel_top1_llm.append(rel_llm[order_llm[0]]*w)
            rel_top3_llm.append(np.sum(rel_llm[order_llm[0:3]])*w)
            rel_top5_llm.append(np.sum(rel_llm[order_llm[0:5]])*w)
            
            acc_top1_llm.append(np.sum(gt_idx==order_llm[0])*w)
            acc_top3_llm.append(np.sum(gt_idx==order_llm[0:3])*w)
            acc_top5_llm.append(np.sum(gt_idx==order_llm[0:5])*w)
            
            for i in range(order_traj.shape[1]):
                
                if i >= d1: 
                    acc_top1_traj[0].append(np.any(np.in1d(gt_idx2, order_traj[0,i]))*w)
                    acc_top3_traj[0].append(np.any(np.in1d(gt_idx2, order_traj[0:3,i]))*w)
                    acc_top5_traj[0].append(np.any(np.in1d(gt_idx2, order_traj[0:5,i]))*w)
                    
                    acc_top1_ours[0].append(np.any(np.in1d(gt_idx2, order_ours[0,i]))*w)
                    acc_top3_ours[0].append(np.any(np.in1d(gt_idx2, order_ours[0:3,i]))*w)
                    acc_top5_ours[0].append(np.any(np.in1d(gt_idx2, order_ours[0:5,i]))*w)
                    
                    rel_top1_traj[0].append(rel_traj[order_traj[0,i],i]*w)
                    rel_top3_traj[0].append(np.sum(rel_traj[order_traj[0:3,i],i])*w)
                    rel_top5_traj[0].append(np.sum(rel_traj[order_traj[0:5,i],i])*w)
                    rel_top1_ours[0].append(rel_ours[order_ours[0,i],i]*w)
                    rel_top3_ours[0].append(np.sum(rel_ours[order_ours[0:3,i],i])*w)
                    rel_top5_ours[0].append(np.sum(rel_ours[order_ours[0:5,i],i])*w)
                    
                if i >= d2:
                    acc_top1_traj[1].append(np.any(np.in1d(gt_idx2, order_traj[0,i]))*w)
                    acc_top3_traj[1].append(np.any(np.in1d(gt_idx2, order_traj[0:3,i]))*w)
                    acc_top5_traj[1].append(np.any(np.in1d(gt_idx2, order_traj[0:5,i]))*w)
                    
                    acc_top1_ours[1].append(np.any(np.in1d(gt_idx2, order_ours[0,i]))*w)
                    acc_top3_ours[1].append(np.any(np.in1d(gt_idx2, order_ours[0:3,i]))*w)
                    acc_top5_ours[1].append(np.any(np.in1d(gt_idx2, order_ours[0:5,i]))*w)
                    
                    rel_top1_traj[1].append(rel_traj[order_traj[0,i],i]*w)
                    rel_top3_traj[1].append(np.sum(rel_traj[order_traj[0:3,i],i])*w)
                    rel_top5_traj[1].append(np.sum(rel_traj[order_traj[0:5,i],i])*w)
                    rel_top1_ours[1].append(rel_ours[order_ours[0,i],i]*w)
                    rel_top3_ours[1].append(np.sum(rel_ours[order_ours[0:3,i],i])*w)
                    rel_top5_ours[1].append(np.sum(rel_ours[order_ours[0:5,i],i])*w)
                    
                if i >= d3:
                    acc_top1_traj[2].append(np.any(np.in1d(gt_idx2, order_traj[0,i]))*w)
                    acc_top3_traj[2].append(np.any(np.in1d(gt_idx2, order_traj[0:3,i]))*w)
                    acc_top5_traj[2].append(np.any(np.in1d(gt_idx2, order_traj[0:5,i]))*w)
                    
                    acc_top1_ours[2].append(np.any(np.in1d(gt_idx2, order_ours[0,i]))*w)
                    acc_top3_ours[2].append(np.any(np.in1d(gt_idx2, order_ours[0:3,i]))*w)
                    acc_top5_ours[2].append(np.any(np.in1d(gt_idx2, order_ours[0:5,i]))*w)
                    
                    rel_top1_traj[2].append(rel_traj[order_traj[0,i],i]*w)
                    rel_top3_traj[2].append(np.sum(rel_traj[order_traj[0:3,i],i])*w)
                    rel_top5_traj[2].append(np.sum(rel_traj[order_traj[0:5,i],i])*w)
                    rel_top1_ours[2].append(rel_ours[order_ours[0,i],i]*w)
                    rel_top3_ours[2].append(np.sum(rel_ours[order_ours[0:3,i],i])*w)
                    rel_top5_ours[2].append(np.sum(rel_ours[order_ours[0:5,i],i])*w)
                    
        out_acc_pct_llm1 = np.zeros([1,10])
        out_acc_pct_llm3 = np.zeros([1,10])
        out_acc_pct_llm5 = np.zeros([1,10])
        out_acc_pct_traj1 = np.zeros([3,10])
        out_acc_pct_traj3 = np.zeros([3,10])
        out_acc_pct_traj5 = np.zeros([3,10])
        out_acc_pct_ours1 = np.zeros([3,10])
        out_acc_pct_ours3 = np.zeros([3,10])
        out_acc_pct_ours5 = np.zeros([3,10])
        out_acc_pct_llm_all = np.zeros([1,10])
        out_acc_pct_traj_all = np.zeros([3,10])
        out_acc_pct_ours_all = np.zeros([3,10])
        
        
        out_acc_rel_llm1 = np.zeros([1,10])
        out_acc_rel_llm3 = np.zeros([1,10])
        out_acc_rel_llm5 = np.zeros([1,10])
        out_acc_rel_traj1 = np.zeros([3,10])
        out_acc_rel_traj3 = np.zeros([3,10])
        out_acc_rel_traj5 = np.zeros([3,10])
        out_acc_rel_ours1 = np.zeros([3,10])
        out_acc_rel_ours3 = np.zeros([3,10])
        out_acc_rel_ours5 = np.zeros([3,10])
        out_acc_rel_llm_all = np.zeros([1,10])
        out_acc_rel_traj_all = np.zeros([3,10])
        out_acc_rel_ours_all = np.zeros([3,10])
        
        out_rel_rel_llm1 = np.zeros([1,10])
        out_rel_rel_llm3 = np.zeros([1,10])
        out_rel_rel_llm5 = np.zeros([1,10])
        out_rel_rel_traj1 = np.zeros([3,10])
        out_rel_rel_traj3 = np.zeros([3,10])
        out_rel_rel_traj5 = np.zeros([3,10])
        out_rel_rel_ours1 = np.zeros([3,10])
        out_rel_rel_ours3 = np.zeros([3,10])
        out_rel_rel_ours5 = np.zeros([3,10])
        out_rel_rel_llm_all = np.zeros([1,10])
        out_rel_rel_traj_all = np.zeros([3,10])
        out_rel_rel_ours_all = np.zeros([3,10])
        
        acc_all_llm = np.array(acc_top1_llm + acc_top3_llm + acc_top5_llm)
        rel_all_llm = np.array(rel_top1_llm + rel_top3_llm + rel_top5_llm)
        
        acc_top1_llm = np.array(acc_top1_llm)
        acc_top3_llm = np.array(acc_top3_llm)
        acc_top5_llm = np.array(acc_top5_llm)
        rel_top1_llm = np.array(rel_top1_llm)
        rel_top3_llm = np.array(rel_top3_llm)/3
        rel_top5_llm = np.array(rel_top5_llm)/5
        
        for j in range(3):
            
            acc_all_traj[j] = np.array(acc_top1_traj[j] + acc_top3_traj[j] + acc_top5_traj[j])
            acc_all_ours[j] = np.array(acc_top1_ours[j] + acc_top3_ours[j] + acc_top5_ours[j])
            rel_all_traj[j] = np.array(rel_top1_traj[j] + rel_top3_traj[j] + rel_top5_traj[j])
            rel_all_ours[j] = np.array(rel_top1_ours[j] + rel_top3_ours[j] + rel_top5_ours[j])
            
            acc_top1_traj[j] = np.array(acc_top1_traj[j])
            acc_top3_traj[j] = np.array(acc_top3_traj[j])
            acc_top5_traj[j] = np.array(acc_top5_traj[j])
            acc_top1_ours[j] = np.array(acc_top1_ours[j])
            acc_top3_ours[j] = np.array(acc_top3_ours[j])
            acc_top5_ours[j] = np.array(acc_top5_ours[j])
            
            
            rel_top1_traj[j] = np.array(rel_top1_traj[j])
            rel_top3_traj[j] = np.array(rel_top3_traj[j])
            rel_top5_traj[j] = np.array(rel_top5_traj[j])
            rel_top1_ours[j] = np.array(rel_top1_ours[j])
            rel_top3_ours[j] = np.array(rel_top3_ours[j])
            rel_top5_ours[j] = np.array(rel_top5_ours[j])
            
        N_thre = 10
        for i in range(N_thre):
            
            thre = 100/N_thre*i
            
            out_acc_pct_llm1[0,i] = np.sum(acc_top1_llm[rel_top1_llm>np.percentile(rel_top1_llm,thre)])/np.sum(rel_top1_llm>np.percentile(rel_top1_llm,thre))*100
            out_acc_pct_llm3[0,i] = np.sum(acc_top3_llm[rel_top3_llm>np.percentile(rel_top3_llm,thre)])/np.sum(rel_top3_llm>np.percentile(rel_top3_llm,thre))*100
            out_acc_pct_llm5[0,i] = np.sum(acc_top5_llm[rel_top5_llm>np.percentile(rel_top5_llm,thre)])/np.sum(rel_top5_llm>np.percentile(rel_top5_llm,thre))*100
            out_acc_pct_llm_all[0,i] = np.sum(acc_all_llm[rel_all_llm>np.percentile(rel_all_llm,thre)])/np.sum(rel_all_llm>np.percentile(rel_all_llm,thre))*100
            
            for j in range(3):
                out_acc_pct_traj1[j,i] = np.sum(acc_top1_traj[j][rel_top1_traj[j]>np.percentile(rel_top1_traj[j],thre)])/np.sum(rel_top1_traj[j]>np.percentile(rel_top1_traj[j],thre))*100
                out_acc_pct_traj3[j,i] = np.sum(acc_top3_traj[j][rel_top3_traj[j]>np.percentile(rel_top3_traj[j],thre)])/np.sum(rel_top3_traj[j]>np.percentile(rel_top3_traj[j],thre))*100
                out_acc_pct_traj5[j,i] = np.sum(acc_top5_traj[j][rel_top5_traj[j]>np.percentile(rel_top5_traj[j],thre)])/np.sum(rel_top5_traj[j]>np.percentile(rel_top5_traj[j],thre))*100
                out_acc_pct_traj_all[j,i] = np.sum(acc_all_traj[j][rel_all_traj[j]>np.percentile(rel_all_traj[j],thre)])/np.sum(rel_all_traj[j]>np.percentile(rel_all_traj[j],thre))*100
                out_acc_pct_ours1[j,i] = np.sum(acc_top1_ours[j][rel_top1_ours[j]>np.percentile(rel_top1_ours[j],thre)])/np.sum(rel_top1_ours[j]>np.percentile(rel_top1_ours[j],thre))*100
                out_acc_pct_ours3[j,i] = np.sum(acc_top3_ours[j][rel_top3_ours[j]>np.percentile(rel_top3_ours[j],thre)])/np.sum(rel_top3_ours[j]>np.percentile(rel_top3_ours[j],thre))*100
                out_acc_pct_ours5[j,i] = np.sum(acc_top5_ours[j][rel_top5_ours[j]>np.percentile(rel_top5_ours[j],thre)])/np.sum(rel_top5_ours[j]>np.percentile(rel_top5_ours[j],thre))*100
                out_acc_pct_ours_all[j,i] = np.sum(acc_all_ours[j][rel_all_ours[j]>np.percentile(rel_all_ours[j],thre)])/np.sum(rel_all_ours[j]>np.percentile(rel_all_ours[j],thre))*100
        """
        N_thre = 10
        for i in range(N_thre):
            
            thre = 100/N_thre*i
            
            out_acc_rel_llm1[0,i] = np.sum(acc_top1_llm[rel_top1_llm>np.percentile(rel_top1_llm,thre)])/np.sum(rel_top1_llm>np.percentile(rel_top1_llm,thre))*100
            out_acc_rel_llm3[0,i] = np.sum(acc_top3_llm[rel_top3_llm>np.percentile(rel_top3_llm,thre)])/np.sum(rel_top3_llm>np.percentile(rel_top3_llm,thre))*100
            out_acc_rel_llm5[0,i] = np.sum(acc_top5_llm[rel_top5_llm>np.percentile(rel_top5_llm,thre)])/np.sum(rel_top5_llm>np.percentile(rel_top5_llm,thre))*100
            out_acc_rel_llm_all[0,i] = np.sum(acc_all_llm[rel_all_llm>np.percentile(rel_all_llm,thre)])/np.sum(rel_all_llm>np.percentile(rel_all_llm,thre))*100
            
            out_rel_rel_llm1[0,i] = np.percentile(rel_top1_llm,thre)
            out_rel_rel_llm3[0,i] = np.percentile(rel_top3_llm,thre)
            out_rel_rel_llm5[0,i] = np.percentile(rel_top5_llm,thre)
            out_rel_rel_llm_all[0,i] = np.percentile(rel_all_llm,thre)
            
            for j in range(3):
                out_acc_rel_traj1[j,i] = np.sum(acc_top1_traj[j][rel_top1_traj[j]>np.percentile(rel_top1_traj[j],thre)])/np.sum(rel_top1_traj[j]>np.percentile(rel_top1_traj[j],thre))*100
                out_acc_rel_traj3[j,i] = np.sum(acc_top3_traj[j][rel_top3_traj[j]>np.percentile(rel_top3_traj[j],thre)])/np.sum(rel_top3_traj[j]>np.percentile(rel_top3_traj[j],thre))*100
                out_acc_rel_traj5[j,i] = np.sum(acc_top5_traj[j][rel_top5_traj[j]>np.percentile(rel_top5_traj[j],thre)])/np.sum(rel_top5_traj[j]>np.percentile(rel_top5_traj[j],thre))*100
                out_acc_rel_traj_all[j,i] = np.sum(acc_all_traj[j][rel_all_traj[j]>np.percentile(rel_all_traj[j],thre)])/np.sum(rel_all_traj[j]>np.percentile(rel_all_traj[j],thre))*100
                out_acc_rel_ours1[j,i] = np.sum(acc_top1_ours[j][rel_top1_ours[j]>np.percentile(rel_top1_ours[j],thre)])/np.sum(rel_top1_ours[j]>np.percentile(rel_top1_ours[j],thre))*100
                out_acc_rel_ours3[j,i] = np.sum(acc_top3_ours[j][rel_top3_ours[j]>np.percentile(rel_top3_ours[j],thre)])/np.sum(rel_top3_ours[j]>np.percentile(rel_top3_ours[j],thre))*100
                out_acc_rel_ours5[j,i] = np.sum(acc_top5_ours[j][rel_top5_ours[j]>np.percentile(rel_top5_ours[j],thre)])/np.sum(rel_top5_ours[j]>np.percentile(rel_top5_ours[j],thre))*100
                out_acc_rel_ours_all[j,i] = np.sum(acc_all_ours[j][rel_all_ours[j]>np.percentile(rel_all_ours[j],thre)])/np.sum(rel_all_ours[j]>np.percentile(rel_all_ours[j],thre))*100
                
                out_rel_rel_traj1[j,i] = np.percentile(rel_top1_traj[j],thre)
                out_rel_rel_traj3[j,i] = np.percentile(rel_top3_traj[j],thre)
                out_rel_rel_traj5[j,i] = np.percentile(rel_top5_traj[j],thre)
                out_rel_rel_traj_all[j,i] = np.percentile(rel_all_traj[j],thre)
                out_rel_rel_ours1[j,i] = np.percentile(rel_top1_ours[j],thre)
                out_rel_rel_ours3[j,i] = np.percentile(rel_top3_ours[j],thre)
                out_rel_rel_ours5[j,i] = np.percentile(rel_top5_ours[j],thre)
                out_rel_rel_ours_all[j,i] = np.percentile(rel_all_ours[j],thre)
        """
        out_pct = np.concatenate([out_acc_pct_llm1,out_acc_pct_llm3,out_acc_pct_llm5,out_acc_pct_traj1,out_acc_pct_traj3,out_acc_pct_traj5,out_acc_pct_ours1,out_acc_pct_ours3,out_acc_pct_ours5],axis=0).transpose()
        out_pct_all = np.concatenate([out_acc_pct_llm_all,out_acc_pct_traj_all,out_acc_pct_ours_all],axis=0).transpose()
        #out_rel = np.concatenate([out_rel_rel_llm1,out_acc_rel_llm1,out_rel_rel_llm3,out_acc_rel_llm3,out_rel_rel_llm5,out_acc_rel_llm5,out_rel_rel_traj1,out_acc_rel_traj1,out_rel_rel_traj3,out_acc_rel_traj3,out_rel_rel_traj5,out_acc_rel_traj5,out_rel_rel_ours1,out_acc_rel_ours1,out_rel_rel_ours3,out_acc_rel_ours3,out_rel_rel_ours5,out_acc_rel_ours5],axis=0).transpose()
        #out_rel_all = np.concatenate([out_rel_rel_llm_all,out_acc_rel_llm_all,out_rel_rel_traj_all,out_acc_rel_traj_all,out_rel_rel_ours_all,out_acc_rel_ours_all],axis=0).transpose()
                   
        np.savetxt(folder_path_eval + "res_eval_pct_act.csv", out_pct, delimiter=',', fmt='%s')
        np.savetxt(folder_path_eval + "res_eval_pct_all_act.csv", out_pct_all, delimiter=',', fmt='%s')