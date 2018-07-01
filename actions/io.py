import numpy as np
import actions,utils

class ActionReader(object):
    def __init__(self):
        self.get_action_desc=cp_dataset
#        self.img_seq=img_seq
        self.get_action_paths=utils.bottom_files  	
    
    def __call__(self,action_dir):
        action_paths=self.get_action_paths(action_dir)
        actions=[self.parse_action(action_path_i) 
                   for action_path_i in action_paths]
        if(not actions):
            raise Exception("No actions found at " + str(action_dir))
        return actions

    def parse_action(self,action_path):
        name,cat,person=self.get_action_desc(action_path)       
        img_seq=read_text_action(action_path)
        return actions.Action(img_seq,name,cat,person)

def read_text_action(action_path):
    return list(np.genfromtxt(action_path, delimiter=','))

def cp_dataset(action_path):
    action_name=action_path.split('/')[-1]
    raw=action_name.split('_')
    if(len(raw)==4):
        cat=raw[0].replace('a','')
        person=int(raw[1].replace('s',''))
        return action_name,cat,person
    raise Exception("Wrong dataset format " + name +" " + str(len(names)))

