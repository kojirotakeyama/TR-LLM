import cv2
import numpy as np
from typing import Optional
import torch
import copy
import os
from model_vrlocomotion import PRED_GM, PRED_TRAJ_ALONG_PATH, PRED_TRAJ_POS
import torchvision.transforms as T
import pyastar2d

def disp_traj(scene_image, pred_map1, observed, observed2, goal, pred_pos1=[], pred_pos2=[]):

	save_dir1 = './disp_traj/'
                        
	if not os.path.exists(save_dir1):
	    os.makedirs(save_dir1)
    
	pred_len = int(pred_map1.shape[0])
    
	for i in range(pred_len):
        
		img = scene_image[0,0,:].reshape(256,256,1)*255
            
		img2 = np.swapaxes(np.swapaxes(pred_map1[i:i+1,:,:],0,2),0,1)
		img2 = np.clip(img2,0,1)
		img2[0,0,0] += 0.00000000001
		img2 = img2/np.max(img2)*1000
            
		img3 = np.swapaxes(np.swapaxes(pred_map1[i:i+1,:,:],0,2),0,1)
		img3 = np.clip(img3,0,1)
		img3[0,0,0] += 0.00000000001
		img3 = img3/np.max(img3)*1000*0

		img_out = np.concatenate([img,img2,img3],axis=2)
                            
		for k in range(0,observed.shape[1]-1):
		    x1 = int(observed[k,0])
		    y1 = int(observed[k,1])
		    x2 = int(observed[k+1,0])
		    y2 = int(observed[k+1,1])
		    cv2.line(img_out, (x1, y1), (x2, y2), (0,155,0), thickness=1)
                
		for k in range(0,observed2.shape[1]-1):
		    x1 = int(observed2[k,0])
		    y1 = int(observed2[k,1])
		    x2 = int(observed2[k+1,0])
		    y2 = int(observed2[k+1,1])
		    cv2.line(img_out, (x1, y1), (x2, y2), (0,0,155), thickness=1)
            
		for k in range(0,i):
		    x1 = int(pred_pos1[k,0])
		    y1 = int(pred_pos1[k,1])
		    x2 = int(pred_pos1[k+1,0])
		    y2 = int(pred_pos1[k+1,1])
		    cv2.line(img_out, (x1, y1), (x2, y2), (155,155,0), thickness=1)

		cv2.circle(img_out, (int(goal[0]), int(goal[1])), 5, (255, 255, 255), thickness=-1)
		cv2.circle(img_out, (int(pred_pos1[i,0]), int(pred_pos1[i,1])), 5, (155, 155, 0), thickness=-1)
		#cv2.circle(img_out, (int(pred_pos2[i,0]), int(pred_pos2[i,1])), 5, (0, 255, 255), thickness=-1)
		cv2.imwrite(save_dir1 + '{:0=3}'.format(i)+'.png', img_out)
        
def create_meshgrid(
        x: torch.Tensor,
        normalized_coordinates: Optional[bool]) -> torch.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs)  # pos_y, pos_x

def conv_points2img(grid, traj, distribution, img_size):
    
    return 1 / (2*np.pi*distribution[0]*distribution[1])*torch.exp(-((grid[0].repeat(traj.shape[0],traj.shape[1],1,1) - torch.swapaxes(torch.swapaxes(traj[:,:,0].repeat(img_size,img_size,1,1),0,2),1,3))**2 / (2*distribution[0]**2) + (grid[1].repeat(traj.shape[0],traj.shape[1],1,1) - torch.swapaxes(torch.swapaxes(traj[:,:,1].repeat(img_size,img_size,1,1),0,2),1,3))**2 / (2*distribution[1]**2)))

def softargmax(x):
	""" Softargmax: As input a batched image where softmax is already performed (not logits) """
		
	x2 = x**2
	x2 = x2/torch.sum(x2)

	pos_y, pos_x = create_meshgrid((x2.unsqueeze(0)).unsqueeze(0), normalized_coordinates=False)
	pos_x = pos_x.reshape(-1)
	pos_y = pos_y.reshape(-1)
	x2 = x2.flatten(0)

	estimated_x = pos_x * x2
	estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
	estimated_y = pos_y * x2
	estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
	softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
	
	return softargmax_coords

def get_pos(last_obs, pred, img_size):
    
    prev_pos = last_obs
    traj_pos = []
    
    x = torch.linspace(0, img_size, img_size)
    y = torch.linspace(0, img_size, img_size)
    x, y = torch.meshgrid(x, y)
    
    for i in range(pred.shape[0]):
        
        prev_map = conv_points2img([y,x], traj=torch.tensor(prev_pos.reshape(1,1,-1)), distribution=[30.0, 30.0], img_size=img_size).detach().numpy()
        prev_map = get_patch_cut(prev_map, prev_pos, size=50)
        cur_map = prev_map[0]*pred[i,:]
        
        if np.sum(cur_map)!=0:
            cur_map = cur_map/np.sum(cur_map)
        else:
            cur_map = pred[i:i+1,:]
            
        cur_pos = np.array(softargmax(torch.tensor(cur_map[0])))
        
        if np.isnan(cur_pos[0]) or np.isnan(cur_pos[1]):
            cur_pos = prev_pos
        else:
            prev_pos = cur_pos
        
        traj_pos.append(cur_pos)
    
    return np.array(traj_pos)

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

def get_patch_cut(x, pos, size=50):
     
    x1 = int(pos[1])-int(size/2)
    y1 = int(pos[0])-int(size/2)
    x2 = int(pos[1])+int(size/2)
    y2 = int(pos[0])+int(size/2)
    
    if x1<0:
        x1 = 0
    if x2>x.shape[3]-1:
        x2 = x.shape[3]-1
    if y1<0:
        y1 = 0
    if y2>x.shape[3]-1:
        y2 = x.shape[3]-1
    
    x_out = np.zeros(x.shape)
    x_out[:,:,x1:x2,y1:y2] = copy.deepcopy(x[:,:,x1:x2,y1:y2])
    
    return x_out

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

def interpolate_points(coords_batch_, M):
    """
    Converts a batch of trajectories to M evenly spaced points for each trajectory.
    
    :param coords_batch: B×N×2 array of trajectory points
    :param M: Number of equally spaced points to return for each trajectory
    :return: B×M×2 array of interpolated points
    """
    coords_batch = np.expand_dims(coords_batch_, axis=0)
    
    B, N, _ = coords_batch.shape
    
    # Initialize the output array
    interpolated_batch = np.zeros((B, M, 2))
    
    # Process each trajectory in the batch
    for b in range(B):
        coords = coords_batch[b]
        
        # Compute the cumulative distance along the trajectory
        distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
        
        # Total length of the trajectory
        total_length = cumulative_distances[-1]
        
        # Desired distances for M points
        desired_distances = np.linspace(0, total_length, M)
        
        # Interpolating points along the trajectory
        for i in range(M):
            # Find the segment of the trajectory that contains the desired distance
            segment_index = np.searchsorted(cumulative_distances, desired_distances[i])
            segment_start = segment_index - 1
            segment_end = segment_index
            
            # Compute the position within the segment
            segment_length = cumulative_distances[segment_end] - cumulative_distances[segment_start]
            if segment_length == 0:
                interpolated_batch[b, i] = coords[segment_start]
            else:
                segment_fraction = (desired_distances[i] - cumulative_distances[segment_start]) / segment_length
                start_point = coords[segment_start]
                end_point = coords[segment_end]
                interpolated_batch[b, i] = start_point + segment_fraction * (end_point - start_point)
    
    return np.squeeze(interpolated_batch.astype(np.float32), axis=0)

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
    
    modify_cost = 50
    modify_range = 0
    while(path is None):
        cost_map[costmap_org<np.percentile(costmap_org,85-modify_range)] = modify_cost
        cost_map[smap==0] = np.inf
        cost_map = padd_ones(cost_map,start)
        cost_map = padd_ones(cost_map,goal)
        
        path = pyastar2d.astar_path(cost_map, (start[1],start[0]), (goal[1],goal[0]), allow_diagonal=False)
        modify_cost += 10
        modify_range += 5
    
    path_out = copy.deepcopy(path)
    path_out[:,0] = copy.deepcopy(path[:,1])
    path_out[:,1] = copy.deepcopy(path[:,0])  
                                  
    return path_out

def get_global_path_map(map_binary, obs, obs2, goal, map_size=128):
    
    gt_template = create_gaussian_heatmap_template(size=map_size*2, kernlen=31, nsig=4, normalize=False)
    gt_template = torch.Tensor(gt_template)
    
    obs_map = get_patch(gt_template, obs, map_size, map_size)
    obs_map = torch.stack(obs_map).reshape([-1, obs.shape[0], map_size, map_size])
    
    obs_map2 = get_patch(gt_template, obs2, map_size, map_size)
    obs_map2 = torch.stack(obs_map2).reshape([-1, obs2.shape[0], map_size, map_size])
    
    goal_map = get_patch(gt_template, goal.reshape(1,-1), map_size, map_size)
    goal_map = torch.stack(goal_map).reshape([-1, 1, map_size, map_size])
    
    feature_input = torch.cat([map_binary, obs_map, obs_map2, goal_map], dim=1).type(torch.float32)
    
    model = PRED_GM(obs_len=15,
			   pred_len=135,
			   map_channel=1,
			   encoder_channels=[64,64,128,128,128],
			   decoder_channels=[128,128,128,64,64]
                           )
    
    checkpoint = torch.load('./vrlocomotion_models_000/model_pred_gm_8epoch.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    pred = model(feature_input)
    
    return torch.sigmoid(pred[0,0]).detach().cpu().numpy()

def get_dynamic_path(static_path, obs, goal, map_size=256):
    
    static_path2 = torch.tensor(interpolate_points(static_path, 64))
    feature_input = torch.cat([torch.tensor(static_path2).unsqueeze(0), torch.tensor(obs).unsqueeze(0), torch.tensor(goal).reshape(1,1,-1)], dim=1).type(torch.float32)/map_size
    
    model = PRED_TRAJ_POS(obs_len=15,
			   pred_len=135,
			   map_channel=1,
			   encoder_channels=[128,128,256,256,256],
			   decoder_channels=[256,256,256,128,128]
                           )
    
    checkpoint = torch.load('./vrlocomotion_models_000/model_pred_global_traj_1976epoch.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    pred = model(feature_input)
    
    pred = torch.sigmoid(pred).reshape(-1,135,2).detach().cpu().numpy()[0]
    #pred_out = copy.deepcopy(pred)
    #pred_out[:,0] = copy.deepcopy(pred[:,1])
    #pred_out[:,1] = copy.deepcopy(pred[:,0])  
    
    return pred

def get_traj_along_path(map_binary, obs, obs2, global_path_map, goal, map_size=256):
    
    map_binary = torch.Tensor(map_binary).reshape(1,1,map_size,map_size)
    global_path_map = torch.Tensor(global_path_map).reshape(1,1,map_size,map_size)
    
    gt_template = create_gaussian_heatmap_template(size=map_size*2, kernlen=31, nsig=4, normalize=False)
    gt_template = torch.Tensor(gt_template)
    
    obs_map = get_patch(gt_template, obs, map_size, map_size)
    obs_map = torch.stack(obs_map).reshape([-1, obs.shape[0], map_size, map_size])
    
    obs_map2 = get_patch(gt_template, obs2, map_size, map_size)
    obs_map2 = torch.stack(obs_map2).reshape([-1, obs2.shape[0], map_size, map_size])
    
    goal_map = get_patch(gt_template, goal.reshape(1,-1), map_size, map_size)
    goal_map = torch.stack(goal_map).reshape([-1, 1, map_size, map_size])
    
    feature_input = torch.cat([map_binary, obs_map, obs_map2, global_path_map, goal_map], dim=1).type(torch.float32)
    
    model = PRED_TRAJ_ALONG_PATH(obs_len=15,
			   pred_len=135,
			   map_channel=1,
			   encoder_channels=[64,64,128,128,128],
			   decoder_channels=[128,128,128,64,64]
                           )
    
    checkpoint = torch.load('./vrlocomotion_models_000/model_pred_traj_wog_14epoch.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    pred = model(feature_input)
    
    return torch.sigmoid(pred[0]).detach().cpu().numpy()

def conv_traj2img(path, map_size):
    
    path_map = np.zeros([map_size,map_size])
        
    for x in path:
        path_map[x[0],x[1]] = 1
            
    return path_map
    
def get_trajectory(map_binary, obs, obs2, goal, map_size=128):
    
    transform = T.Resize(size = (map_size,map_size))
    map_binary = transform(torch.tensor(map_binary).reshape(1,1,map_binary.shape[0],map_binary.shape[1]))

    global_path_prob_map = get_global_path_map(map_binary, obs, obs2, goal, map_size=map_size)
    
    #resize scale to 256*256
    map_size2 = 256
    transform = T.Resize(size = (map_size2,map_size2))
    map_binary = transform(map_binary).reshape(-1,1,map_size2,map_size2)

    global_path_prob_map = cv2.resize(global_path_prob_map, (map_size2,map_size2))
    obs = obs*map_size2/map_size
    obs2 = obs2*map_size2/map_size
    goal = goal*map_size2/map_size
    
    #viz_input(map_binary, obs_map, obs_map, goal_map, global_path_map)
    
    global_path = get_global_path(global_path_prob_map, map_binary.detach().cpu().numpy()[0,0], obs[-1].astype(np.int32), goal.astype(np.int32))
    global_path_map = conv_traj2img(global_path, map_size2)
    
    traj_along_path = get_traj_along_path(map_binary.detach().cpu().numpy()[0,0], obs, obs2, global_path_map, goal, map_size=256)
    
    """
    traj_pos = []
    for i in range(traj_along_path.shape[0]):
        traj_pos.append(softargmax(torch.Tensor(traj_along_path[i,:])))
    traj_pos = np.array(torch.stack(traj_pos))
    """
        
    traj_pos = get_pos(obs[-1],traj_along_path,map_size2)
    disp_traj(map_binary, traj_along_path, obs, obs2, goal, pred_pos1=traj_pos)
    
    return traj_pos

def disp_debug_img(bmap, path1, path2, debug):
    
    bmap = np.expand_dims(bmap, axis=2)
    bmap = np.concatenate([bmap,bmap,bmap],axis=2)

    for i in range(path1.shape[0]):
        cv2.circle(bmap, (int(path1[i,0]), int(path1[i,1])), 2, (255,150,150), thickness=-1)
    for i in range(path2.shape[0]):
        cv2.circle(bmap, (int(path2[i,0]), int(path2[i,1])), 1, (255,0,0), thickness=-1)
    
    cv2.imwrite(debug, bmap)

def get_trajectory2(map_binary, obs, obs2, goal, map_size=128, flag_fix=0, debug=[]):
    
    margin = 0.7
    map_binary2 = cv2.resize(map_binary, dsize=(map_size, map_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    
    if flag_fix==0:
        cv2.circle(map_binary2, (int(obs[0,0]), int(obs[0,1])), int(margin*map_size/10), (1), thickness=-1)
        cv2.circle(map_binary2, (int(goal[0]), int(goal[1])), int(margin*map_size/10), (1), thickness=-1)
    
    map_binary2 = torch.tensor(map_binary2).unsqueeze(0).unsqueeze(0)

    global_path_prob_map = get_global_path_map(map_binary2, obs, obs2, goal, map_size=map_size)
    
    #resize scale to 256*256
    map_size2 = 256

    global_path_prob_map = cv2.resize(global_path_prob_map, (map_size2,map_size2))
    obs = obs*map_size2/map_size
    obs2 = obs2*map_size2/map_size
    goal = goal*map_size2/map_size
    
    map_binary2 = cv2.resize(map_binary, dsize=(map_size2, map_size2), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    if flag_fix==0:
        cv2.circle(map_binary2, (int(obs[0,0]), int(obs[0,1])), int(margin*map_size2/10), (1), thickness=-1)
        cv2.circle(map_binary2, (int(goal[0]), int(goal[1])), int(margin*map_size2/10), (1), thickness=-1)
    
    #viz_input(map_binary, obs_map, obs_map, goal_map, global_path_map)
    
    global_path_static = get_global_path(global_path_prob_map, map_binary2, obs[-1].astype(np.int32), goal.astype(np.int32))
    
    global_path_dynamic = get_dynamic_path(global_path_static, obs, goal, map_size=map_size2)*map_size2
    
    if flag_fix==1:
        v = np.sqrt(np.sum((global_path_dynamic[:-1]-global_path_dynamic[1:])**2,axis=1))/256*10*15
        v_smooth = (v[:-2] + v[1:-1] + v[2:])/3
        if np.min(v_smooth[15:])<0.1:
            end_idx = np.min(np.where(v_smooth[15:]<0.1))+15
            global_path_dynamic = global_path_dynamic[:end_idx]
    
    if flag_fix==0:
        cv2.circle(map_binary2, (int(obs[0,0]), int(obs[0,1])), int(margin*map_size2/10), (0), thickness=-1)
        cv2.circle(map_binary2, (int(goal[0]), int(goal[1])), int(margin*map_size2/10), (0), thickness=-1)
        global_path2 = []
        global_path_dynamic = global_path_dynamic[::3]
        #map_debug = copy.deepcopy(map_binary2)*255
        for i in range(global_path_dynamic.shape[0]):
            #map_debug[global_path[i,0],global_path[i,1]]=100
            if map_binary2[int(global_path_dynamic[i,1]),int(global_path_dynamic[i,0])]==1:
                global_path2.append(global_path_dynamic[i])
        #cv2.imwrite('./debug.png', map_debug)
        global_path2 = np.array(global_path2)

    else:
        global_path2 = global_path_dynamic[::3]
    
    #if global_path2.shape[0]>35:
    #    global_path2 = global_path2[:35]
    if debug!=[]:
        disp_debug_img(map_binary2*255, global_path_dynamic, global_path2, debug)
    
    if global_path2.shape[0]<2:
        return np.zeros([10,2])
    
    return global_path2/2