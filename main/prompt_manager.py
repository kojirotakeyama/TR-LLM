from openai import OpenAI
import numpy as np
import json
import setting as st
import utility as util
import copy
import random
from scipy.spatial import distance

def conv_time_float2pt(time):
    
    seconds = np.fmod(time,60)
    minutes = int(np.fmod(time-seconds,3600)/60)
    hours = np.fmod(int((time-minutes*60-seconds)/3600),24)
            
    ap = 'am'
    if hours>12:
        hours = hours - 12
        ap = 'pm'
    
    return str(hours) + ":" + "{:0=2}".format(minutes) + ap + " " + str(seconds)+ "sec"

def conv_time_str2float(time):
    
    hours = int(time[:time.find(':')])
    minutes = int(time[(time.find(':')+1):])
    
    seconds = hours*3600 + minutes*60
    
    return seconds

def get_obj_pt(obj_state):
    
    obj_pt = []
    
    text = '[OBJECT LIST]\n'
    for obj in list(obj_state['obj_list'].keys()):
        if (obj in st.list_obj_use)==False and st.list_obj_use!=[]:
            continue
        
        if (obj in st.list_obj_exclude) and st.list_obj_exclude!=[]:
            continue
        
        text += '- ' + obj +'\n'
        #text += ' at (x,y)=(' + str(obj_state[1][i][0]) + 'm,' + str(obj_state[1][i][1]) + 'm)\n'
    
    obj_pt.append(text)
        
    return obj_pt

def get_scene_pt(scene_state, keys=[]):
    
    if keys==[]:
        keys = list(scene_state.keys())
        
    pt = "[scene state]\n"
            
    if 'month' in scene_state.keys() and 'month' in keys:
        pt += " - Today is " + scene_state['month']
        
    if 'day' in scene_state.keys() and 'day' in keys:
        pt += "/" + str(scene_state['day'])
        
    if 'year' in scene_state.keys() and 'year' in keys:
        pt += "/" + str(scene_state['year'])
        
    if 'day of week' in scene_state.keys() and 'day of week' in keys:
        pt += " (" + str(scene_state['day of week']) + ")\n"
                
    if 'time' in scene_state.keys() and 'time' in keys:
        time = conv_time_float2pt(scene_state['time'])
        pt += " - Current time is " + time
                 
    if 'area' in scene_state.keys() and 'area' in keys:
        pt += " in " + scene_state['area'] + "\n"
        
    if 'room temperature' in scene_state.keys() and 'room temperature' in keys:
        pt += " - The room temperature is " + scene_state['room temperature'] + "\n"
        
    if 'outside temperature' in scene_state.keys() and 'outside temperature' in keys:
        pt += " - The temperature outside the room is " + scene_state['outside temperature'] + "\n"
        
    if 'weather' in scene_state.keys() and 'weather' in keys:
        pt += " - The weather is " + scene_state['weather'] + "\n"
    
    pt += "\n"
    
    return [pt]

def get_person_pt(person_state, keys_remove=[]):
    
    keys = {}
    keys_tmp = list(person_state[st.list_person_use[0]].keys())
    for k in keys_tmp:
        keys[k] = list(set(list(person_state[st.list_person_use[0]][k].keys())) - set(keys_remove))
    
    person_pt = {}
    person_obs_pt = {}
    
    for p in st.list_person_use:
        
        name = person_state[p]['persona']['name']
        pt = "\n"
        pt += "[" + name + "'s' state] {\n"
        
        #Adding persona
        pt += name + "'s persona is defined as follows. \n"
        
        if 'name' in keys['persona'] and 'name' in person_state[p]['persona']:
            pt += 'name' + ': ' + str(person_state[p]['persona']['name']) + '\n'
        
        if 'age' in keys['persona'] and 'age' in person_state[p]['persona']:
            pt += 'age' + ': ' + str(person_state[p]['persona']['age']) + '\n'
        
        if 'gender' in keys['persona'] and 'gender' in person_state[p]['persona']:
            pt += 'gender' + ': ' + str(person_state[p]['persona']['gender']) + '\n'
        
        if 'nationality' in keys['persona'] and 'nationality' in person_state[p]['persona']:
            pt += 'nationality' + ': ' + str(person_state[p]['persona']['nationality']) + '\n'
        
        if 'agent type' in keys['persona'] and 'agent type' in person_state[p]['persona']:
            pt += 'agent type' + ': ' + str(person_state[p]['persona']['agent type']) + '\n'
            
        if 'relationship to others' in keys['persona'] and 'relationship to others' in person_state[p]['persona']:
            if len(person_state[p]['persona']['relationship to others'])>0:
                pt += "relationship to others:\n"
                for v in person_state[p]['persona']['relationship to others']:
                    pt += ' - ' + list(v.keys())[0] + ': ' + list(v.values())[0] + '\n'
                    
        if 'personality' in keys['persona'] and 'personality' in person_state[p]['persona']:
            if len(person_state[p]['persona']['personality'])>0:
                pt += "personality:\n"
                for v in person_state[p]['persona']['personality']:
                    pt += ' - ' + v + '\n'

        if 'preference' in keys['persona'] and 'preference' in person_state[p]['persona']:
            if len(person_state[p]['persona']['preference'])>0:
                pt += "preference:\n"
                for v in person_state[p]['persona']['preference']:
                    pt += ' - ' + v + '\n'
                        
        if 'daily routine' in keys['persona'] and 'daily routine' in person_state[p]['persona']:
            if len(person_state[p]['persona']['daily routine'])>0:
                pt += "daily routine:\n"
                pt += " week days:\n"
                for v in person_state[p]['persona']['daily routine']['weekdays']:
                    pt += '  - ' + v['task'] + ': ' + v['time'] + '\n'
                pt += " week end:\n"
                for v in person_state[p]['persona']['daily routine']['weekend']:
                    pt += '  - ' + v['task'] + ': ' + v['time'] + '\n'

        #pt += "\n" + name + "'s observable state is described as follows. \n"
        
        #Adding observable state
        if 'holding object' in keys['observable state'] and 'holding object' in person_state[p]['observable state']:
            if person_state[p]['observable state']['holding object']['right hand'] != 'nan':
                pt += name + ' is holding ' + str(person_state[p]['observable state']['holding object']['right hand']) + ' with the right hand.\n'
                
            if person_state[p]['observable state']['holding object']['left hand'] != 'nan':
                pt += name + ' is holding ' + str(person_state[p]['observable state']['holding object']['left hand']) + ' with the left hand.\n'

        if 'action history long' in keys['observable state'] and 'action history long' in person_state[p]['observable state']:
            if len(person_state[p]['observable state']['action history long'])>0:
                pt += "action history long:\n"
                for v in person_state[p]['observable state']['action history long']:
                    pt += ' - ' + v['action'] + '\n'
                        
        if 'action history short' in keys['observable state'] and 'action history short' in person_state[p]['observable state']:
            if len(person_state[p]['observable state']['action history short'])>0:
                pt += "past actions:\n"
                for i in range(len(person_state[p]['observable state']['action history short'])):
                    if i<len(person_state[p]['observable state']['action history short'])-30:
                        continue
                    v = person_state[p]['observable state']['action history short'][i]
                    pt += " - " + v + ".\n" 

        if 'clothes' in keys['observable state'] and 'clothes' in person_state[p]['observable state']:
            if len(person_state[p]['observable state']['clothes'])>0:
                pt += "clothes:\n"
                for v in person_state[p]['observable state']['clothes']:
                    pt += ' - ' + v + '\n'

        if 'action plan short' in keys['internal state'] and 'action plan short' in person_state[p]['internal state']:
            if len(person_state[p]['internal state']['action plan short'])>0:
                pt += "future action plan:\n"
                for v in person_state[p]['internal state']['action plan short']:
                    pt += ' - ' + v + '\n'
                    
        if len(person_state[p]['internal state']['start obj'])>0:
                pt += "current location:\n"
                pt += " - " + str(person_state[p]['internal state']['start obj'][0]) + "\n"
        """
        if 'velocity' in keys['observable state'] and 'velocity' in person_state[p]['observable state']:
            pt += "velocity:\n"
            pt += " - vx: " + str(person_state[p]['observable state']['position']['vx']) + "[m/s]\n"
            pt += " - vy: " + str(person_state[p]['observable state']['position']['vy']) + '[m/s]\n'
        """
        person_obs_pt[p] = pt + "}"
        
        #Adding internal (unobservable) state
        if 'high-level task' in keys['internal state'] and 'high-level task' in person_state[p]['internal state']:
            if person_state[p]['internal state']['high-level task']!="nan":
                pt += name + "'s high-level task:\n"
                pt += " - " + person_state[p]['internal state']['high-level task']
                pt += "\n"

        person_pt[p] = pt + "}"
        
    return person_pt, person_obs_pt

def proc_chatgpt(assistant_pt, order_pt, hidden_pt, res_type, past_pt=[], seed=st.seed):
    
    client = OpenAI(api_key=st.API_KEY, organization=st.ORGANIZATION)
    
    prompt = []
    prompt.append({'role': 'system', 'content': "You are a helpful assistant."})
    
    #Adding prompt of surrounding scene infromation
    pt_tmp = ""
    for pt in assistant_pt:
        pt_tmp += pt
    prompt.append({'role': 'assistant', 'content': pt_tmp})
    
    #Adding the order prompt
    prompt.append({'role': 'user', 'content': order_pt})
    
    #Additional hidden prompt
    prompt.append({'role': 'user', 'content': hidden_pt})
    
    if res_type=="json_object":
        prompt.append({'role': 'user', 'content': "Make sure the output format is in the JSON format defined above."})
    
    #if st.type_obj_pos==1:
    #    prompt.append({'role': 'user', 'content': "Note that coordinate (x,y) represents 2d horizontal location in the room."})
                   
    response = client.chat.completions.create(
        model = st.model_chatgpt,
        response_format={ "type": res_type },
        messages = prompt,
        seed=seed,
        temperature=st.temperature,
        max_tokens = 4096
        )
    
    ans = response.choices[0].message.content
    
    if res_type=="json_object":
        
        while(1):
            count=0
            try:
                ans = json.loads(ans)
            except ValueError:
                seed = random.randint(0, 10000)
                
                response = client.chat.completions.create(
                    model = st.model_chatgpt,
                    response_format={ "type": res_type },
                    messages = prompt + ["Ensure that the answer is with a json format."],
                    seed=seed,
                    temperature=st.temperature,
                    max_tokens = 4096
                )
                ans = response.choices[0].message.content
                count+=1
            if count>=5 or count==0:
                break
    
    prompt.append({'role': 'assistant', 'content': ans})
    
    return ans, prompt

def get_order_pred_act(name):
    
    order_pt = ""
    
    order_pt += (name + " is about to leave the current location and is planning to do the next action.\n")
    order_pt += ("Which object will " + name + " interact with and what action will " + name + " take?\n")
    order_pt += ("Please list all possible target objects and associated actions, ensuring that the target object is selected only from the [OBJECT LIST].\n")
    
    return order_pt

def pred_act(assistant_pt, past_pt, name, res_type="text"):
    
    order_pt = get_order_pred_act(name)
    
    hidden_pt = "Make the answer considering surrounding scene information such as following factors.\
                   \n - spatial relationship between objects and persons\
                   \n - person state: e.g. grabbing objects, clothes, position, etc.\
                   \n - object state: size, temperature, open or close, dirty or clean, etc.\
                   \n - relationship between persons\
                   \n - time\
                   \n - season, weather, temperature, etc.\
                   "
    ans, past_pt = proc_chatgpt(assistant_pt, order_pt, hidden_pt, res_type, past_pt=past_pt)
    
    return ans, past_pt

def get_order_pred_act_based_on_obj(name, obj):
    
    order_pt = ""
    
    order_pt += (name + " is about to leave the current location and is planning to go to " + obj + " to do the next action.\n")
    order_pt += ("What action will " + name + " take at " + obj + "?\n")
    order_pt += ("Please answer the action " + name + " will take, considering scene context.\n\n")
    
    order_pt += ("The answer should be with a json format. Here are some examples.\n")
    order_pt += ("Example1: in case the next target object is couch\n")
    order_pt += ("{\"content\":{\"action\":\"relax on the couch\"}}\n\n")
    order_pt += ("Example1: in case the next target object is fridge\n")
    order_pt += ("{\"content\":{\"action\":\"get something to drink\"}}\n\n")
    
    order_pt += ("\"action\" should be a purpose that a person interact with the object.\n")
    order_pt += ("\"action\" should be described in five words or less.\n")
    order_pt += ("\"action\" should not contain any person name.\n")
    
    return order_pt

def pred_act_based_on_obj(assistant_pt, gt_obj, name):
    
    order_pt = get_order_pred_act_based_on_obj(name, gt_obj)
    
    hidden_pt = "Make the answer considering surrounding scene information such as following factors.\
                   \n - spatial relationship between objects and persons\
                   \n - person state: e.g. grabbing objects, clothes, position, etc.\
                   \n - object state: size, temperature, open or close, dirty or clean, etc.\
                   \n - relationship between persons\
                   \n - time\
                   \n - season, weather, temperature, etc.\
                   "
    seed = random.randint(0, 10000)
    ans, _ = proc_chatgpt(assistant_pt, order_pt, hidden_pt, res_type="json_object", seed=seed)
    
    return ans

def get_order_get_target_object2(name, obj_pt, pred_pt):

    order_pt = ""
    
    order_pt += (name + " is in a living room. " + name + " is now beginning to leave the current location to visit and interact with another object in the room from the [OBJECT LIST]. We want to rank these objects based on the likelihood that " + name + " will interact with them. Please rank the objects in the list following the example provided below. When creating the rank, consider factors such as " + name + "'s action history, relationships with others, time of day, day of the week, personality, and other relevant factors that might influence his next action.\n\n")
    order_pt += ("Please refer [Predicted actions] to rank the objects.\n\n")
    order_pt += ("[Predicted actions]\n" + pred_pt + "\n\n")

    order_pt += ("[Definition of the rank]\n")
    order_pt += ("Rank is represented as A,B,C,D. \n")
    order_pt += ("A: Highly relevant with " + name + "'s next action\n")
    order_pt += ("B: Relevant with " + name + "'s next action\n")
    order_pt += ("C: Poorly relevant with " + name + "'s next action\n")
    order_pt += ("D: Not relevant with " + name + "'s next action\n\n")
    
    #order_pt += ("If you think there is no plausible object to interact with in the [OBJECT LIST], answer \"none\".\n")

    order_pt += ("Example 1:\n")
    order_pt += ("[OBJECT LIST].\n")
    order_pt += ("- fridge\n- kettle\n- cupboard\n- sink\n- paint\n- book\n- appliance\n- tv\n- dining table\n- chair\n- kitchen counter\n")
    order_pt += ("Answer:\n")
    order_pt += ("{\"content\":{\"fridge\":\"B\",\"kettle\":\"B\",\"cupboard\":\"C\",\"sink\":\"B\",\"paint\":\"D\",\"book\":\"C\",\"appliance\":\"D\",\"tv\":\"B\",\"dining table\":\"A\",\"chair\":\"A\",\"kitchen counter\":\"A\"}}\n\n")
    
    order_pt += ("Example 2:\n")
    order_pt += ("[OBJECT LIST].\n")
    order_pt += ("- couch\n- trash bin\n- cabinet\n- glass\n- fireplace\n- carpet\n- window\n- coffee table\n- pillow\n- sink\n- microwave\n- lamp\n- coffee machine\n")
    order_pt += ("Answer:\n")
    order_pt += ("{\"content\":{\"couch\":\"A\",\"trash bin\":\"C\",\"cabinet\":\"C\",\"glass\":\"B\",\"fireplace\":\"D\",\"carpet\":\"D\",\"window\":\"D\",\"coffee table\":\"B\",\"pillow\":\"D\",\"sink\":\"B\",\"microwave\":\"C\",\"lamp\":\"C\",\"coffee machine\":\"B\"}}\n\n")
    
    order_pt += ("Inference:\n")
    order_pt += obj_pt + "\n"
    order_pt += ("Answer:\n ?\n\n")
    #order_pt += ("{\"target object\":\"none\"}\n\n")

    order_pt += ("- DO NOT rank objects that is NOT in the [OBJECT LIST].\n")
    order_pt += ("- Consider " + name + "'s action history to evaluate the relevancy.\n")
    order_pt += ("- The answer should be represented with a json format.\n")
    
    #order_pt += ("- Make sure \"target object\" is included in the content.\n\n")
      
    return order_pt

def get_target_object(assistant_pt, obj_pt, pred_pt, name, res_type="text"):
    
    order_pt = get_order_get_target_object2(name, obj_pt, pred_pt)
    
    hidden_pt = "Make the answer considering surrounding scene information such as following factors.\
                   \n - spatial relationship between objects and persons\
                   \n - person state: e.g. grabbing objects, clothes, position, etc.\
                   \n - object state: size, temperature, open or close, dirty or clean, etc.\
                   \n - relationship between persons\
                   \n - time\
                   \n - season, weather, temperature, etc.\
                   "
    ans, _ = proc_chatgpt(assistant_pt, order_pt, hidden_pt, res_type)
    
    return ans

def get_order_conversation(name1, name2):

    order_pt = ""
    order_pt += (name1 + " and " + name2 + " is having a conversation. Please generate a conversation that is plausible in the situation.\n\n")
    order_pt += ("- Assume that the conversation happens during the time that is after the past actions and before the future actions.\n")
    order_pt += ("- The conversation is should be in 5 phrases.\n")
    #order_pt += ("- The conversation is not necessary to be related to their action.\n")
    
    #order_pt += ("- Make sure \"target object\" is included in the content.\n\n")
      
    return order_pt

def get_conversation(assistant_pt, name1, name2, res_type="text"):
    
    order_pt = get_order_conversation(name1, name2)

    hidden_pt = ""
    ans, _ = proc_chatgpt(assistant_pt, order_pt, hidden_pt, res_type)
    
    return "[" + name1 + " and " + name2 + "'s conversation]\n" + ans

def get_order_eval_score(act_pred, gt_act):
    
    order_pt = ("A and B describe actions.\n")
    order_pt += ("A: " + gt_act + "\n")
    order_pt += ("B: " + act_pred + "\n\n")
    order_pt += ("We would like to judge A and B are similar or not, based on the following criteria.\n\n")
    order_pt += ("Output 1 when one of the following conditions are met. If not all conditions are satisfied, output 0.\n")
    order_pt += ("- A and B possibly refer to the same action.\n")
    order_pt += ("- A and B could be actions that have a sequential relationship with each other..\n")
    order_pt += ("- A is a broader concept of B.\n")
    order_pt += ("- A is possibly a part of actions to achieve B.\n")
    order_pt += ("- B is a broader concept of A.\n")
    order_pt += ("- B is possibly a part of actions to achieve A.\n\n")

    order_pt += ("Please answer in a json format. Here are two examples of the answer.\n")
    order_pt += ("Example1:\n")
    order_pt += ("{\"content\":{\"score\":\"1\"}}\n\n")
    order_pt += ("Example2:\n")
    order_pt += ("{\"content\":{\"score\":\"0\"}}\n\n")
    return order_pt
"""
def get_order_eval_score(act_pred, gt_act):

    order_pt += ("A and B describes actions.\n")
    order_pt += ("A: " + gt_act + "\n")
    order_pt += ("B: " + act_pred + "\n\n")
    order_pt += ("We would like to judge A and B are similar or not, based on the following criteria.\n\n")
    order_pt += ("Output 1 when one of the following conditions are met. If not all conditions are satisfied, output 0.\n")
    order_pt += ("- A and B generally refer to the same action.\n")
    order_pt += ("- A is a broader concept of B.\n")
    order_pt += ("- A is possibly a part of actions to achieve B.\n")
    order_pt += ("- B is a broader concept of A.\n")
    order_pt += ("- B is possibly a part of actions to achieve A.\n\n")

    order_pt += ("Please answer in a json format. Here are two examples of the answer.\n")
    order_pt += ("Example1:\n")
    order_pt += ("{\"content\":{\"score\":\"1\"}}\n\n")
    order_pt += ("Example2:\n")
    order_pt += ("{\"content\":{\"score\":\"0\"}}\n\n")
    return order_pt
"""
def eval_score(act_pred, gt_act):
    
    order_pt = get_order_eval_score(act_pred, gt_act)
    ans, _ = proc_chatgpt(assistant_pt="", order_pt=order_pt, hidden_pt="", res_type="json_object")
    
    return ans
