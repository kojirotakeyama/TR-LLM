import cv2
import numpy as np

import torch
import copy
from model_vrlocomotion import PRED_GM
import torchvision.transforms as T
import pyastar2d

def create_gaussian_heatmap_template(size, kernlen=81, nsig=4, normalize=True):
	""" Create a big gaussian heatmap template to later get patches out """
	template = np.zeros([size, size])
	kernel = gkern(kernlen=kernlen, nsig=nsig)
	m = kernel.shape[0]
	x_low = template.shape[1] // 2 - int(np.floor(m / 2))
	x_up = template.shape[1] // 2 + int(np.ceil(m / 2))
	y_low = template.shape[0] // 2 - int(np.floor(m / 2))
	y_up = template.shape[0] // 2 + int(np.ceil(m / 2))
	template[y_low:y_up, x_low:x_up] = kernel
	if normalize:
		template = template / template.max()
	return template

def get_patch(template, traj, H, W):
	x = np.round(traj[:,0]).astype('int')
	y = np.round(traj[:,1]).astype('int')

	x_low = template.shape[1] // 2 - x
	x_up = template.shape[1] // 2 + W - x
	y_low = template.shape[0] // 2 - y
	y_up = template.shape[0] // 2 + H - y

	patch = [template[y_l:y_u, x_l:x_u] for x_l, x_u, y_l, y_u in zip(x_low, x_up, y_low, y_up)]

	return patch

def gkern(kernlen=31, nsig=4):
	"""	creates gaussian kernel with side length l and a sigma of sig """
	ax = np.linspace(-(kernlen - 1) / 2., (kernlen - 1) / 2., kernlen)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))
	return kernel / np.sum(kernel)

def padd_ones(map, pos):
        
    map[pos[0],pos[1]]=1
    map[pos[0]+1,pos[1]]=1
    map[pos[0],pos[1]+1]=1
    map[pos[0]-1,pos[1]]=1
    map[pos[0],pos[1]-1]=1
        
    return map
    
def get_global_path(gmap, smap, start, goal):
    
    costmap_org = copy.deepcopy(gmap)
    cost_map = cv2.GaussianBlur(costmap_org, (5, 5), 0)
    cost_map[costmap_org<np.percentile(costmap_org,85)] = np.inf
    cost_map[costmap_org>=np.percentile(costmap_org,85)] = 30
    cost_map[costmap_org>=np.percentile(costmap_org,90)] = 25
    cost_map[costmap_org>=np.percentile(costmap_org,95)] = 20
    cost_map[costmap_org>=np.percentile(costmap_org,96)] = 15
    cost_map[costmap_org>=np.percentile(costmap_org,97)] = 10
    cost_map[costmap_org>=np.percentile(costmap_org,98)] = 6
    cost_map[costmap_org>=np.percentile(costmap_org,99)] = 3
    cost_map[costmap_org>=np.percentile(costmap_org,99.5)] = 1

    cost_map[smap==0] = np.inf
    cost_map = padd_ones(cost_map,start)
    cost_map = padd_ones(cost_map,goal)
    
    path = pyastar2d.astar_path(cost_map, (start[1],start[0]), (goal[1],goal[0]), allow_diagonal=False)
    
    return path

def get_trajectory(map_binary, obs, goal):
    
    map_size = map_binary.shape[0]
    
    transform = T.Resize(size = (map_size,map_size))
    map_binary = transform(torch.tensor(map_binary).reshape(1,1,map_binary.shape[0],map_binary.shape[1]))

    gt_template = create_gaussian_heatmap_template(size=map_size*2, kernlen=31, nsig=4, normalize=False)
    gt_template = torch.Tensor(gt_template)
    
    obs_map = get_patch(gt_template, obs, map_size, map_size)
    obs_map = torch.stack(obs_map).reshape([-1, obs.shape[0], map_size, map_size])
    
    goal_map = get_patch(gt_template, goal.reshape(1,-1), map_size, map_size)
    goal_map = torch.stack(goal_map).reshape([-1, 1, map_size, map_size])
    
    feature_input = torch.cat([map_binary, obs_map, obs_map*0, goal_map], dim=1).type(torch.float32)
    
    model = PRED_GM(obs_len=15,
			   pred_len=135,
			   map_channel=1,
			   encoder_channels=[64,64,128,128,128],
			   decoder_channels=[128,128,128,64,64]
                           )
    
    checkpoint = torch.load('./vrlocomotion_models_000/model_pred_gm_8epoch.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    pred = model(feature_input)
    
    global_path_map = torch.sigmoid(pred[0,0]).detach().cpu().numpy()
    
    #viz_input(map_binary, obs_map, obs_map, goal_map, global_path_map)
    
    global_path = get_global_path(global_path_map, map_binary.detach().cpu().numpy()[0,0], obs[-1].astype(np.int32), goal[0,0].astype(np.int32))
