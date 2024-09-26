import pickle
import cv2
import numpy as np
import setting as st
import csv
import os
import torch
import torch.nn.functional as F
import math
import copy
from model_vrlocomotion import PRED_GOAL, PRED_GOAL2, PRED_GM, PRED_TRAJ_ALONG_PATH
import torchvision.transforms as T

def disp_prob_obj(map_in, obj_candidate, obj_prob, obj_list, dir_name, file_name, prob_scale, flag_save=1):

    map_out = copy.deepcopy(map_in)
    prob_sum = 0
    for obj in obj_candidate:
        for i in range(obj_prob[obj].shape[0]):
            prob_sum += obj_prob[obj][i]
    
    for obj in obj_candidate:
        for i in range(obj_prob[obj].shape[0]):
            probability = obj_prob[obj][i]/prob_sum*100
            #print([obj, probability])
                
            x = obj_list[obj][i]
                
            b = 255-int(probability*prob_scale)
            g = 0
            r = int(probability*prob_scale)
                
            if b<0:
                b = 0
            if r>255:
                r = 255
            map_out[x[:,1],x[:,0],:]=np.array([b,g,r])
            #print(prob)
    
    if flag_save==1:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
                
        cv2.imwrite(dir_name + file_name, map_out)
    
    return map_out
        
def rotate_img(img, angle):
    
    img_out = torch.zeros(angle.shape[0],angle.shape[1],1,img.shape[1],img.shape[2])
    
    for i in range(angle.shape[0]):
        
        rotation_matrix = torch.zeros(angle.shape[1], 2, 3)
        rotation_matrix[:, 0, 0] = torch.cos(angle[i])
        rotation_matrix[:, 1, 1] = rotation_matrix[:, 0, 0]
        rotation_matrix[:, 0, 1] = -torch.sin(angle[i])  # +/- sin(angle)
        rotation_matrix[:, 1, 0] = -rotation_matrix[:, 0, 1]

        rotation_grids = F.affine_grid(rotation_matrix, (angle.shape[1], 1, img.shape[1], img.shape[2]))
        
        img_out[i] = F.grid_sample(img.repeat(angle.shape[1],1,1,1), rotation_grids)
    
    return torch.swapaxes(img_out,3,4)

def gaussian_distribution(x, mu, sigma):
    # Compute the prefactor 1/(sqrt(2*pi*sigma^2))
    prefactor = 1 / (math.sqrt(2 * math.pi * sigma**2))
    # Compute the exponent factor of the Gaussian formula
    exponent = math.exp(-((x - mu) ** 2) / (2 * sigma**2))
    
    return prefactor * exponent

def get_temp_img(distribution, img_size):
    
    img_out = np.zeros([img_size*2, img_size*2])
    
    for x in range(-img_size, img_size):
        for y in range(img_size):
            
            theta = np.arctan2(y,x)
            dist = np.sqrt(x**2+y**2)/img_size*10
            img_out[img_size+y,img_size+x] = gaussian_distribution(theta, torch.pi/2, distribution)/(5+dist)
    
    return img_out

def conv_heading2img2(pos, heading, template, img_size):
    
    img_r = rotate_img(template, heading)
    
    img_out = torch.zeros(pos.shape[0],pos.shape[1],img_size,img_size)
    
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            
            x = img_size - int(pos[i,j,0])
            y = img_size - int(pos[i,j,1])
            
            if x<0:
                x = 0
            if y<0:
                y = 0
            if x+img_size>img_r.shape[3]:
                x = img_r.shape[3] - img_size
            if y+img_size>img_r.shape[4]:
                y = img_r.shape[4] - img_size
                
            img_out[i,j] = img_r[i,j,0,y:y+img_size,x:x+img_size]
            img_out[i,j] = img_out[i,j]/torch.max(img_out[i,j])
    
    return img_out

def conv_traj2heading(traj):
    
    vel = np.concatenate([traj[1:2]-traj[0:1],traj[1:]-traj[:-1]],axis=0)
    heading = np.arctan2(vel[:,1],vel[:,0])
    
    return torch.tensor(heading.reshape(1,-1))
    
def viz_input(scene_image, observed_map, observed_map2, goal, pred_map):
    
	save_dir = './viz_input_debug/'
                        
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)
    
	fname = os.listdir(save_dir)
        
	s_img = scene_image.detach().cpu().numpy()*255
	obs = observed_map.detach().cpu().numpy()*255
	obs2 = observed_map2.detach().cpu().numpy()*255
	g = goal.detach().cpu().numpy()*255
    
	for i in range(obs.shape[0]):
		for j in range(obs.shape[1]):
            
			img_out = cv2.UMat(np.swapaxes(np.swapaxes(np.concatenate([s_img[i], obs[i,j:j+1]+obs2[i,j:j+1], g[i,j:j+1]+pred_map.reshape(1,128,128)*2000], axis=0),0,2),0,1))
			#cv2.circle(img_out, (int(goal[i,0,0].detach().cpu().numpy()), int(goal[i,0,1].detach().cpu().numpy())), 2, (0, 255, 255), thickness=-1)
			#cv2.circle(img_out, (int(goal_traj[i,0,0].detach().cpu().numpy()), int(goal_traj[i,0,1].detach().cpu().numpy())), 2, (0, 155, 155), thickness=-1)
			
			cv2.imwrite(save_dir + '{:0=6}'.format(len(fname)) + '_{:0=3}'.format(i) + '_{:0=3}'.format(j) +'.png', img_out)

def disp_pred_goal(scene_image, observed, goal_map, dir_name, file_name):

    save_dir1 = './'
                        
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    
    s_img = cv2.resize(scene_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    goal_map = cv2.resize(goal_map, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    observed = observed*2

    p_img = goal_map.reshape(256,256,1)
    #img3 = torch.swapaxes(torch.swapaxes(pred_map[0],0,2),0,1).cpu().numpy()
    p_img = np.clip(p_img,0,1)
    p_img[0,0,0] += 0.00000000001
    p_img = p_img/np.max(p_img)*200
        
    p_img = np.concatenate([p_img*0,p_img,p_img],axis=2)
    
    img_out = s_img + p_img
                            
    for k in range(observed.shape[0]-1):
        x1 = int(observed[k,0])
        y1 = int(observed[k,1])
        x2 = int(observed[k+1,0])
        y2 = int(observed[k+1,1])
        cv2.line(img_out, (x1, y1), (x2, y2), (0,255,0), thickness=2)
        
    cv2.circle(img_out, (int(observed[0,0]), int(observed[0,1])), 5, (0, 0, 255), thickness=-1)
    cv2.circle(img_out, (int(observed[-1,0]), int(observed[-1,1])), 5, (0, 255, 0), thickness=-1)
        
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
            
    cv2.imwrite(dir_name + file_name, img_out)
    
    return img_out
        
def conv_points2img(traj, distribution, img_size):
    
    traj = torch.tensor(traj.reshape(1,-1,2))
    x = torch.linspace(1, img_size, img_size)-0.5
    y = torch.linspace(1, img_size, img_size)-0.5
    x, y = torch.meshgrid(x, y)
        
    return 1 / (2*math.pi*distribution[0]*distribution[1])*torch.exp(-((y.repeat(traj.shape[0],traj.shape[1],1,1) - torch.swapaxes(torch.swapaxes(traj[:,:,0].repeat(img_size,img_size,1,1),0,2),1,3))**2 / (2*distribution[0]**2) + (x.repeat(traj.shape[0],traj.shape[1],1,1) - torch.swapaxes(torch.swapaxes(traj[:,:,1].repeat(img_size,img_size,1,1),0,2),1,3))**2 / (2*distribution[1]**2)))

def get_goal_area(map_binary, traj_map, map_size):
    
    transform = T.Resize(size = (map_size,map_size))
    map_binary = transform(torch.tensor(map_binary).reshape(1,1,map_binary.shape[0],map_binary.shape[1]))

    feature_input = torch.cat([map_binary, traj_map], dim=1).type(torch.float32)
    
    model = PRED_GOAL(obs_len=15,
			   pred_len=90,
			   map_channel=1,
			   encoder_channels=[256,256,512,512,512],
			   decoder_channels=[512,512,512,256,256]
                           )
    
    model.load_state_dict(torch.load('./vrlocomotion_models_000/model_pred_goal_6epoch.pt')['model_state_dict'])
    
    pred = model(feature_input)
    
    return torch.sigmoid(pred[0,0]).detach().numpy()

def get_goal_area2(map_binary, traj_map, heading_map, map_size):
    
    transform = T.Resize(size = (map_size,map_size))
    map_binary = transform(torch.tensor(map_binary).reshape(1,1,map_binary.shape[0],map_binary.shape[1]))

    feature_input = torch.cat([map_binary, traj_map, heading_map], dim=1).type(torch.float32)
    
    model = PRED_GOAL2(obs_len=15,
			   pred_len=90,
			   map_channel=1,
			   encoder_channels=[256,256,512,512,512],
			   decoder_channels=[512,512,512,256,256]
                           )
    
    model.load_state_dict(torch.load('./vrlocomotion_models_000/model_1.pt')['model_state_dict'])
    
    pred = model(feature_input)
    
    return torch.sigmoid(pred[0,0]).detach().numpy()
    
def save_data_csv(data, path):
    
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write all rows at once
        writer.writerows(data)
    
def load_col2sem(scene_id):
    
    f = open(st.path_col2sem + '{:0=3}'.format(scene_id),"rb")
    col2sem = pickle.load(f)
    
    return col2sem

def get_sem_pos(scene_id):
    
    col2sem = load_col2sem(scene_id)
    map_sem = cv2.imread(st.path_semantic_map + '/{:0=3}'.format(scene_id) + '.png')
    
    sem_list = {}
    
    for i in range(len(col2sem[0])):
        
        col = col2sem[0][i]
        sem = col2sem[1][i]
        
        idx = np.where(np.sum(((map_sem-col)**2),axis=2)==0)
        idx = np.transpose(np.array(idx))
        idx = np.concatenate([idx[:,1].reshape(-1,1),idx[:,0].reshape(-1,1)],axis=1) #interchange x and y axis.
        
        if idx.shape[0]>0:

            if sem in sem_list:
                sem_list[sem].append(idx)
            
            else:
                sem_list[sem] = [idx]
        
    """
    map_debug = cv2.imread('./texture_map/444.png')
    for c in sem_list['pillow']:
        x = np.array(c)
        map_debug[x[:,1],x[:,0],:]=np.array([0,0,255])
    
    cv2.imwrite('./map_debug.png', map_debug)
    """    
    return sem_list
    
def get_objpos(idx_list):
    
    objpos_list = []
    
    for idx in idx_list:
        objpos_list.append(np.around(np.mean(idx,axis=0)/1024*10,1))
    
    return objpos_list
    
def get_dist_from_obj(person_pos, obj_pos):
    
    dist = np.around(np.linalg.norm(np.array(obj_pos)-np.array(person_pos),axis=1),1)
    
    return dist


def get_waypoint_map(scene_id):
    
    scene_map = cv2.imread(st.path_binary_map + '/{:0=3}'.format(scene_id) + '.png')
    
    waypoints = []
    
    N = 25
    r = 5    #area range that is used for detecting walkable area [pixels] 
    step = int(np.floor(1024/N))
    
    for x in range(N):
        for y in range(N):
            
            if x-r<=0 or x+r>=1024 or y-r<=0 or y+r>=1024:
                continue
            
            if int(np.mean(scene_map[x*step-r:x*step+r,y*step-r:y*step+r,0]))==255:
                waypoints.append(np.array([y*step/1024*10, 10 - x*step/1024*10]))
    
    return waypoints

def get_item_from_actdic_by_key(dic, key1, key2=[]):
    
    dic_out = []
    for v in dic:
        
        if key2!=[]:
            if v[key1]==key2:
                dic_out.append(v)
        else:
            dic_out.append({})
            for key in key1:

                dic_out[-1][key]=v[key]
    
    return dic_out

def save_pickle(value, path):
    
    f = open(path, 'wb')
    pickle.dump(value, f)
    f.close()
    
def save_scenario(act, n, t):
    
    path_save = './Result'
    if os.path.exists(path_save)==False:
        os.makedirs(path_save)
    
    path_save2 = path_save + '/scenario_id=' + str(f"{n:05}")
    if os.path.exists(path_save2)==False:
        os.makedirs(path_save2)
    
    if st.scenario_id != []:
        fname = path_save2 + '/s_' + str(f"{t:05}") + '.pkl'
        
    save_pickle(act, fname)

def load_scenario(n, t):
    
    fname = './Result/scenario_id=' + str(f"{n:05}") + '/s_' + str(f"{t:05}") + '.pkl'
    f = open(fname,'rb')
    data = pickle.load(f)
    f.close()
    
    return data
