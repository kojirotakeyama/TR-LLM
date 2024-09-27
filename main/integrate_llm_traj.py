import numpy as np
import os
import glob

def integrate_llm_traj():
    
    base_dir = './Result/'
    folder_path_llm = base_dir + '/LLM/'
    folder_path_traj = base_dir + '/TRAJ/'
    
    f_llm = os.path.join(folder_path_llm, f"*{'llm'}*")
    f_llm2 = os.path.join(folder_path_llm, f"*{'llm_input'}*")
    f_traj = os.path.join(folder_path_traj, f"*{'traj'}*")
    
    # List all files matching the pattern
    fname_llm_ = sorted(glob.glob(f_llm))
    fname_llm2 = sorted(glob.glob(f_llm2))
    fname_llm = [item for item in fname_llm_ if item not in fname_llm2]
    
    
    folder_path_out = base_dir + '/TR-LLM/'
    if not os.path.exists(folder_path_out):
        os.mkdir(folder_path_out)
    
    for fp_llm in fname_llm:
        
        fp_traj = glob.glob(folder_path_traj+'/result_traj' + fp_llm[-16:])
        fp_traj2 = glob.glob(folder_path_traj+'/traj' + fp_llm[-16:])
        
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

        data_ours = np.zeros(data_traj.shape)
        for i in range(data_traj.shape[0]):
            obj_idx = header.index(header2[i])
            for j in range(data_traj.shape[1]):
                data_ours[i,j] = data_traj[i,j]*data_llm[obj_idx]
        
        res_ours = np.concatenate([np.array(header2).reshape(-1,1),data_ours], axis=1).astype(str).transpose()
        np.savetxt(folder_path_out + "result_tr-llm_" + fp_llm[-15:], res_ours, delimiter=',', fmt='%s')

if __name__ == "__main__":
    
    integrate_llm_traj()