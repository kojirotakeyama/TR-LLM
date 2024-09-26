#model_chatgpt = 'gpt-3.5-turbo-1106'
#model_chatgpt = 'gpt-4-1106-preview'
model_chatgpt = 'gpt-4o-mini-2024-07-18'
seed = 0
temperature = 1.0

API_KEY = '**********************'
ORGANIZATION = '************************'

path_input = '../Eval_data/'
path_texture_map = path_input + 'texture_map/'
path_binary_map = path_input + 'binary_map/'
path_semantic_map = path_input + 'semantic_map_raw/'
path_col2sem = path_input + 'col2sem/'
path_situation = path_input + 'situation/'
path_act = path_input + 'act/'

list_obj_use = []
list_obj_exclude = ['floor','wall','unknown','window frame','rug','carpet']
list_person_use = ['P1','P2']

scene_id = [444,813,814,824,829,839,853,873,876]
act_id = [0,1,2,3,4,5,6,7,8,9]

flag_traj = 1
flag_llm = 1
flag_conversation = 1
flag_history = 1
flag_seq = 1
