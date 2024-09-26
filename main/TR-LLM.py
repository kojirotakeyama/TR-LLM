import setting as st
import utility as util
import prompt_manager as pt
import os
import pickle
import copy
import yaml
import numpy as np
import shutil
import cv2
import json
import path_generator
import sys
import torch

class ACT:

    def __init__(self):
        self.obj_state = {}
        self.person_state = {}
        self.scene_state = []
        self.obj_pt = []
        self.person_pt = []
        self.person_obs_pt = []
        self.scene_pt = []
        
        self.initial_time = 0
        self.list_name2id = {}
        self.scene_id = st.scene_id
        self.trajectory = np.zeros([35,2])   #Interval of the time-series position is 333ms (3Hz)
        self.act_scenario = []
        
        self.obj_candidate_llm = []
        self.obj_prob_llm = []
        
        self.conv_pt = []
        self.assistant_pt = []
        
        self.t_act = 0
        self.t = 0
        self.n_act = 0
        
        self.img_template = torch.tensor(util.get_temp_img(0.2, 128)).unsqueeze(0).type(torch.float32)
    
    def exclude_items(self,items):
        
        items_out = copy.deepcopy(items)
        for obs in list(items.keys()):
            if obs.endswith(" "):
                
                if not obs.rstrip() in st.list_obj_exclude:
                    items_out[obs.rstrip()] = copy.deepcopy(items[obs])
                del items_out[obs]
            elif obs in st.list_obj_exclude:
                del items_out[obs]
        
        return items_out
                
    def init_obj_state(self):
        
        self.obj_state['obj_list'] = self.exclude_items(util.get_sem_pos(self.scene_id))
        self.obj_candidate_llm = list(self.obj_state['obj_list'].keys())
        
    def get_goal_area(self):
        img_size = 128
        map_binary = cv2.imread(st.path_binary_map + f'{self.scene_id:03}' + '.png')[:,:,0:1]
        map_binary[np.where(map_binary>0)]=1

        traj = np.zeros([35,2])
        if self.t_act <= self.trajectory.shape[0]:
            if self.t_act<=35:
                traj[-self.t_act:,:] = self.trajectory[:self.t_act,:]
                traj[:-self.t_act,:] = self.trajectory[0,:]
            else:
                traj = self.trajectory[self.t_act-35:self.t_act,:]
        
        traj_map = util.conv_points2img(traj, distribution=[0.5,0.5], img_size=img_size)

        self.traj_tmp = traj
        self.goal_area = util.get_goal_area(map_binary, traj_map, img_size)
        self.goal_area = cv2.GaussianBlur(self.goal_area, ksize=(35,35), sigmaX=7)
    
    def load_map(self):
        
        map_binary = cv2.imread(st.path_binary_map + f'{self.scene_id:03}' + '.png')[:,:,0:1]
        map_binary[np.where(map_binary>0)]=1
        self.map_binary = map_binary
        
        self.map_texture = cv2.imread(st.path_texture_map + f'{self.scene_id:03}'+'.png')
    
    def get_traj(self):
        
        print('Trajectory prediction: scene: ' + str(sc_id) + ' action: ' + str(a_id) + ' n: ' + str(self.n_act))
        
        img_size = 128
        traj_obs = (np.zeros([15,2]) + self.person_state['P1']['internal state']['start pos'])*img_size/10
        traj_obs2 = traj_obs*0
        
        self.goal = self.person_state['P1']['internal state']['goal pos']*img_size/10
    
        traj = path_generator.get_trajectory2(self.map_binary, traj_obs, traj_obs2, self.goal, map_size=img_size, flag_fix=0)
        
        if traj.shape[0]<15:
            print('Trajectory prediction failed: Length of the trajectory is too short...')
            return 0
        
        traj_obs = np.zeros([15,2]) + traj[0]
        goal = traj[-1]
        traj2 = path_generator.get_trajectory2(self.map_binary, traj_obs, traj_obs2, goal, map_size=img_size, flag_fix=1)
        
        if traj2.shape[0]<15:
            print('Trajectory prediction failed: Length of the trajectory is too short...')
            return 0
        
        self.trajectory = traj2
        
        print('Trajectory prediction succeeded: number of trajectory step is ' + str(traj.shape[0]))
        return 1
        
    def pred_obj_llm(self):
        print('LLM prediction: scene: ' + str(sc_id) + ' action: ' + str(a_id) + ' n: ' + str(self.n_act))
        
        assistant_pt_tmp = []
        
        if flag_history==1:
            self.update_pt(keys_remove=[['observable state','action history short']])
            assistant_pt_tmp.append(self.get_pt(st.list_person_use))
        
        if flag_conversation==1:
            self.update_pt()
            assistant_pt_tmp.append(self.get_pt(st.list_person_use))
        
        self.update_pt()
        assistant_pt_tmp.append(self.get_pt(st.list_person_use))
        
        if len(self.conv_pt)!=0:
            #assistant_pt_tmp[-1] = assistant_pt_tmp[-1] + self.conv_pt
            assistant_pt_tmp[-1] = assistant_pt_tmp[-1] + ['[' + act.person_state['P1']['persona']['name'] + ' and ' + act.person_state['P2']['persona']['name'] + 's conversation]\n' + self.conv_pt2[0]]
        
        self.obj_pt = pt.get_obj_pt(self.obj_state)
    
        obj_ref = copy.deepcopy(self.obj_state['obj_list'])
        obj_pt = copy.deepcopy(self.obj_pt[0])
        
        conv_flag = 0
        count = 0
        while not conv_flag:
            
            past_pt = []
            if st.flag_seq==1:
                for assistant_pt in assistant_pt_tmp:
                    pred_pt, past_pt = pt.pred_act(assistant_pt, past_pt, name=act.person_state['P1']['persona']['name'])
                
            else:
                pred_pt, past_pt = pt.pred_act(assistant_pt_tmp[-1], past_pt, name=act.person_state['P1']['persona']['name'])
                
            assistant_pt = assistant_pt_tmp[-1]
            
            t_obj = pt.get_target_object(assistant_pt, obj_pt, pred_pt, name=act.person_state['P1']['persona']['name'], res_type="json_object")
            
            conv_flag = 1
            
            if not "content" in list(t_obj.keys()):
                conv_flag = 0
                print('error: there is a problem with the output format...')
            
            if not list(t_obj['content'].keys()) == list(obj_ref.keys()):
                conv_flag = 0
                print('error: there is a problem with keys...')
                
            count += 1
            if count>10:
                return 0
        
        #self.obj_candidate_llm = []
        self.obj_prob_llm = {}
        A = 15.0
        B = 10.0
        C = 5.0
        D = 1.0

        for i in range(len(list(t_obj['content'].keys()))):
            #self.obj_candidate_llm.append(list(obj_ref.keys())[i])
            obj = list(t_obj['content'].keys())[i]
            if t_obj['content'][obj]=="A":
                self.obj_prob_llm[obj] = A
            if t_obj['content'][obj]=="B":
                self.obj_prob_llm[obj] = B
            if t_obj['content'][obj]=="C":
                self.obj_prob_llm[obj] = C
            if t_obj['content'][obj]=="D":
                self.obj_prob_llm[obj] = D
        
        dir_name = './Result/LLM/pred_obj_llm_{:0=3}'.format(self.scene_id)
        file_name = '/map_predobj_'+ f'{self.scene_id:03}_' + f'{self.act_id:03}_' + f'{self.n_act:03}_' + f'{self.t_act:03}' + '.png'
        map_binary = np.repeat(copy.deepcopy(self.map_binary),3,2)
        map_binary[map_binary==1]=100
        
        #util.disp_prob_obj(map_in=map_binary, obj_candidate=self.obj_candidate_llm, obj_prob=self.obj_prob_llm, obj_list=self.obj_state['obj_list'], dir_name=dir_name, file_name=file_name, prob_scale=20, flag_save=1)

        self.assistant_pt = assistant_pt
        #for i in range(len(self.obj_prob_llm)):
        #    self.obj_prob_llm[i] = self.obj_prob_llm[i]/prob_sum
            
        return 1

    def get_conversation_llm(self):
        print('LLM get conversation: scene: ' + str(sc_id) + ' action: ' + str(a_id) + ' n: ' + str(self.n_act))
        
        self.update_pt(mode='conversation')
        assistant_pt = self.get_pt(st.list_person_use, obs_type=1)
        self.obj_pt = pt.get_obj_pt(self.obj_state)
    
        self.conv_pt = [pt.get_conversation(assistant_pt, name1=act.person_state['P1']['persona']['name'], name2=act.person_state['P2']['persona']['name'])]
        
    def pred_obj_traj(self):
        
        print('Goal prediction: scene: ' + str(sc_id) + ' action: ' + str(a_id) + ' n: ' + str(self.n_act))
        
        prob = {}
        
        for obj in list(self.obj_state['obj_list'].keys()):
            prob[obj] = []
            for i in range(len(self.obj_state['obj_list'][obj])):
                obj_area = self.obj_state['obj_list'][obj][i]
            
                goal_area = cv2.resize(self.goal_area, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
                goal_area = goal_area/np.sum(goal_area)
    
                prob[obj].append(np.sum(goal_area[obj_area[:,1],obj_area[:,0]])/obj_area.shape[0])
            prob[obj] = np.array(prob[obj])
        
        dir_name = './Result/TRAJ/pred_obj_traj_{:0=3}'.format(self.scene_id)
        file_name = '/map_predobj_'+ f'{self.scene_id:03}_' + f'{self.act_id:03}_' + f'{self.n_act:03}_' + f'{self.t_act:03}' + '.png'
        
        #map_texture = np.clip(cv2.resize(self.map_texture, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.int16)-100, 0, 255).astype(np.uint8)
        map_binary = np.repeat(copy.deepcopy(self.map_binary),3,2)
        map_binary[map_binary==1]=100
        
        map_obj_prob = util.disp_prob_obj(map_in=map_binary, obj_candidate=self.obj_candidate_llm, obj_prob=prob, obj_list=self.obj_state['obj_list'], dir_name=dir_name, file_name=file_name, prob_scale=25, flag_save=0)
        
        #map_debug = cv2.resize(map_obj_prob, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
        self.map_pred_goal = util.disp_pred_goal(map_obj_prob, self.traj_tmp, self.goal_area, dir_name=dir_name, file_name=file_name)
        
        print('Goal prediction done.')
              
        return prob
        
    def get_location(self):

        obj_area = self.obj_state['obj_list'][self.person_state['P1']['internal state']['start obj'][0]][0]
        self.person_state['P1']['internal state']['start pos'] = np.mean(obj_area,axis=0)/1024*10
        obj_area = self.obj_state['obj_list'][self.person_state['P1']['internal state']['goal obj'][0]][0]
        self.person_state['P1']['internal state']['goal pos'] = np.mean(obj_area,axis=0)/1024*10
        
        if np.sqrt(np.sum((self.person_state['P1']['internal state']['start pos'] - self.person_state['P1']['internal state']['goal pos'])**2)) < 0.5:
            return 0
        else:
            return 1

    def load_situation(self, file_idx):

        with open(st.path_situation + 's_{:0=5}'.format(file_idx) + '.yaml','rb') as file:
            s= yaml.load(file, Loader=yaml.FullLoader)

        for i in range(len(st.list_person_use)):
            p = st.list_person_use[i]
            self.person_state[p] = {'persona': {'name': 'nan', 'agent type': 'nan', 'nationality': 'nan', 'age': 'nan', 'gender': 'nan', 'relationship': ['nan'], 'personality': ['nan'], 'preference': ['nan'], 'daily routine': ['nan']}, 'observable state': {'clothes': ['nan'], 'holding object': {'left hand': 'nan', 'right hand': 'nan'}, 'position': {'x': 'nan', 'y': 'nan'}, 'velocity': {'x': 'nan', 'y': 'nan'}, 'heading': {'theta': 'nan'}}, 'internal state': {'high-level task': 'nan', 'low-level task': 'nan', 'action plan long': ['nan'], 'action plan short': ['nan'], 'target position': {'x': 'nan', 'y': 'nan'}, 'target heading': {'theta': 'nan'}}}
            self.person_state[p]['persona'] = copy.deepcopy(s[p])
            del self.person_state[p]['observable state']['clothes']
        
        self.scene_state = copy.deepcopy(s['Scene'])
        self.scene_state['time'] = pt.conv_time_str2float(self.scene_state['time'])
    
    def modify_target_obj(self, t_obj, obj_flexibility, ref_obj):
        
        target_obj = {key: obj_flexibility[key] for key in t_obj}
        
        target_obj_out = copy.deepcopy(target_obj)
        
        for obj in list(target_obj.keys()):

            for obj2 in target_obj[obj]:
                if obj2 in ref_obj:
                    target_obj_out[obj] = obj2
                    break
                
        return target_obj_out
    
    def get_conv_pt2(self, conv, name1, name2):
        
        for i in range(len(conv)):
            
            if conv[i] == None:
                continue

            if 'P1' in conv[i]:
                conv[i] = conv[i].replace('P1',name1)
            if 'P2' in conv[i]:
                conv[i] = conv[i].replace('P2',name2)
    
        return conv
    
    def replace_obj_name(self, name_list, name_list2):
        
        name_list_out = []
        
        for txt in name_list:
            parts = txt.split('"')
            name_list_tmp = ""
            
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    if part in name_list2.keys():
                        if not isinstance(name_list2[part], list):
                            name_list_tmp += name_list2[part]
                        else:
                            name_list_tmp += part
                    else:
                        name_list_tmp += part
                else:
                    name_list_tmp += part
            
            name_list_out.append(name_list_tmp)
        
        return name_list_out
    
    def load_act(self, file_idx):
        
        with open(st.path_act + 'act_{:0=3}'.format(file_idx) + '.yaml') as file:
            s = yaml.load(file, Loader=yaml.FullLoader)
        
        with open(st.path_act + 'object_flexibility.yaml') as file:
            s2 = yaml.load(file, Loader=yaml.FullLoader)
        
        target_obj = self.modify_target_obj(s['objects'], s2['object flexibility'], list(self.obj_state['obj_list'].keys()))
        s['P1']['high-level task'] = self.replace_obj_name([s['P1']['high-level task']], target_obj)[0]
        s['P1']['action'] = self.replace_obj_name(s['P1']['action'], target_obj)
        s['P2']['action'] = self.replace_obj_name(s['P2']['action'], target_obj)
        
        self.person_state['P1']['internal state']['high-level task'] = copy.deepcopy(s['P1']['high-level task'])
        self.conv_pt2 = self.get_conv_pt2(s['conversation'], self.person_state['P1']['persona']['name'], self.person_state['P2']['persona']['name'])
        
        for p in st.list_person_use:
            self.person_state[p]['internal state']['action plan short'] = s[p]['action']
            self.person_state[p]['internal state']['low-level task'] = s[p]['action'][0]
            self.person_state[p]['observable state']['action history short'] = []
            
            for i in range(len(s[p]['start'])):
                s[p]['start'][i] = target_obj[s[p]['start'][i]]
            for i in range(len(s[p]['goal'])):
                s[p]['goal'][i] = target_obj[s[p]['goal'][i]]
            
            self.person_state[p]['internal state']['start obj'] = s[p]['start']
            self.person_state[p]['internal state']['goal obj'] = s[p]['goal']
        
    def update_pt(self, mode='prediction', keys_remove=[]):
        
        #convert the scene states to prompts
        self.obj_pt = pt.get_obj_pt(self.obj_state)
        
        person_state_tmp = copy.deepcopy(self.person_state)
        
        if mode=='prediction':
            if flag_history==0:
                keys_remove.append(['observable state','action history short'])
            
            keys_remove.append(['internal state','action plan short'])
            keys_remove.append(['internal state','high-level task'])
                
        if len(keys_remove) > 0:
            for key in keys_remove:
                if key[0] in person_state_tmp['P1']:
                    if key[1] in person_state_tmp['P1'][key[0]]:
                        del person_state_tmp['P1'][key[0]][key[1]]
                        del person_state_tmp['P2'][key[0]][key[1]]
        
        self.person_pt, self.person_obs_pt = pt.get_person_pt(person_state_tmp)
            
        self.scene_pt = pt.get_scene_pt(self.scene_state)
    
    def get_pt(self, p=[], obs_type=0):
        
        assistant_pt = self.scene_pt
           
        for p2 in st.list_person_use:
            
            if obs_type==0:
                if p2==p:
                    assistant_pt += [self.person_pt[p2]]
                else:
                    assistant_pt += [self.person_obs_pt[p2]]
            
            if obs_type==1:
                assistant_pt += [self.person_pt[p2]]
            
            if obs_type==2:
                assistant_pt += [self.person_obs_pt[p2]]
        
        return assistant_pt
    
    def update_scenario(self):
        
        while(1):
            if self.t_act > 0 or self.n_act > 0:
                self.person_state['P1']['observable state']['action history short'].append(self.person_state['P1']['internal state']['low-level task'])
                del self.person_state['P1']['internal state']['action plan short'][0]
                self.person_state['P1']['internal state']['low-level task'] = copy.deepcopy(self.person_state['P1']['internal state']['action plan short'][0])
                del self.person_state['P1']['internal state']['start obj'][0]
                del self.person_state['P1']['internal state']['goal obj'][0]
                del self.conv_pt2[0]
                self.n_act += 1
            
            self.t_act = 1
            
            if len(self.person_state['P1']['internal state']['action plan short'])==1:
                return 0
            
            if isinstance(self.person_state['P1']['internal state']['start obj'][0], list):
                return 0
            
            if isinstance(self.person_state['P1']['internal state']['goal obj'][0], list):
                return 0
            
            if self.person_state['P1']['internal state']['start obj'][0]==self.person_state['P1']['internal state']['goal obj'][0]:
                continue
            
            if not self.get_location():
                continue
            
            if not self.get_traj() and flag_traj==1:
                continue
            
            if flag_llm==1:
                if flag_conversation==1:
                    self.get_conversation_llm()
                
                if not self.pred_obj_llm():
                    continue
                
            return 1
            
if __name__ == "__main__":
    
    flag_conversation = st.flag_conversation
    flag_llm = st.flag_llm
    flag_traj = st.flag_traj
    flag_history = st.flag_history
    flag_multi = st.flag_multi
    flag_etc = st.flag_etc
    
    output_type = ""
    args = sys.argv
    if len(args)>1:
        output_type = args[1]
        flag_traj = int(args[2])
        flag_llm = int(args[3])
        flag_conversation = int(args[4])
        flag_history = int(args[5])
    
    output_dir = './Result' + output_type + '/'
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir + 'TRAJ/'):
        os.mkdir(output_dir + 'TRAJ/')
    if not os.path.exists(output_dir + 'LLM/'):
        os.mkdir(output_dir + 'LLM/')

    for sc_id in st.scene_id:
        for a_id in st.act_id:
                
            act = ACT()
            act.scene_id = sc_id
            act.act_id = a_id
                
            act.init_obj_state()
            act.load_situation(file_idx=a_id)
                
            act.load_act(file_idx=act.act_id)
            act.load_map()
                
            act.t=0
            res_traj = []
            res_llm = []
            header = []
                
            while 1:
                    
                if act.t_act > act.trajectory.shape[0] or act.t==0 or flag_traj==0:
                        
                    if res_traj != []:
                        res_traj = np.array(res_traj)
                        res_traj = np.vstack([np.array(header).reshape(1,-1),res_traj]).astype(str)
                        np.savetxt(output_dir + "TRAJ/result_traj_{:0=3}".format(act.scene_id) + "_{:0=3}".format(act.act_id) + "_{:0=3}".format(act.n_act) + ".csv", res_traj, delimiter=',', fmt='%s')
                        np.savetxt(output_dir + "TRAJ/traj_{:0=3}".format(act.scene_id) + "_{:0=3}".format(act.act_id) + "_{:0=3}".format(act.n_act) + ".csv", np.concatenate([act.trajectory,np.repeat(act.goal.reshape(1,-1),act.trajectory.shape[0],axis=0)],axis=1), delimiter=',', fmt='%s')
                        res_traj = []
                        
                    if len(act.assistant_pt) != 0 and flag_llm != 0:
                        header = list(act.obj_state['obj_list'].keys())
                        res_llm = np.array(list(act.obj_prob_llm.values()))
                        res_llm = np.vstack([np.array(header).reshape(1,-1),res_llm]).astype(str)
                            
                        np.savetxt(output_dir + "LLM/result_llm_{:0=3}".format(act.scene_id) + "_{:0=3}".format(act.act_id) + "_{:0=3}".format(act.n_act) + ".csv", res_llm, delimiter=',', fmt='%s')
                        res_llm = []
                            
                        np.savetxt(output_dir + "LLM//result_llm_input_{:0=3}".format(act.scene_id) + "_{:0=3}".format(act.act_id) + "_{:0=3}".format(act.n_act) + ".txt", act.assistant_pt, fmt='%s')
                            
                    if act.update_scenario()==0:
                        break
                    
                if flag_traj==1:
                    act.get_goal_area()
                    prob = act.pred_obj_traj()
                        
                    header = []
                    res_traj_tmp = []
                    for obj in list(prob.keys()):
                        for i in range(prob[obj].shape[0]):
                            header.append(obj)
                            res_traj_tmp.append(prob[obj][i])
                    res_traj.append(res_traj_tmp)
                    
                act.t_act += 1
                act.t += 1
                

            
