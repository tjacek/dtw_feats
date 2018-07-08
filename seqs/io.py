import numpy as np
import seqs,utils
import cv2,os

class ActionReader(object):
    def __init__(self,read_dirs,read_seq,as_dict=False):
        self.as_dict=as_dict
        self.get_action_desc=cp_dataset
        self.get_action_paths=read_dirs
        self.read_seq=read_seq  	
    
    def __call__(self,action_dir):
        action_paths=self.get_action_paths(action_dir)
        actions=[self.parse_action(action_path_i) 
                   for action_path_i in action_paths]
        if(not actions):
            raise Exception("No actions found at " + str(action_dir))
        if(self.as_dict):
            actions=as_action_dict(actions)
        return actions

    def parse_action(self,action_path):
        name,cat,person=self.get_action_desc(action_path)       
        img_seq= self.read_seq(action_path) #read_text_action(action_path)
        return seqs.Action(img_seq,name,cat,person)

class ActionWriter(object):
    def __init__(self,img_seq=False):
        if(img_seq):
            self.save_action=as_imgs
        else:
            self.save_action=as_text
        
    def __call__(self,actions,out_path):
        if(type(actions)==dict):
            actions=actions.values()
        utils.make_dir(out_path)
        for action_i in actions:
            action_path=out_path+'/'+action_i.name
            self.save_action(action_i,action_path)

def build_action_reader(img_seq,as_dict=True):
    if(img_seq):
        read_dirs=utils.bottom_dirs
        read_seq=read_img_action
    else:
        read_dirs=utils.bottom_files
        read_seq=read_text_action
    return ActionReader(read_dirs,read_seq,as_dict)

def as_action_dict(actions):
    return { action_i.name:action_i for action_i in actions} 

def read_text_action(action_path):
    return list(np.genfromtxt(action_path, delimiter=','))

def read_img_action(action_path):
    img_names=os.listdir(action_path)
    img_names.sort(key=utils.natural_keys)
    img_paths=[ action_path+'/'+name_i for name_i in img_names]
    return [cv2.imread(img_path_i,0) 
                for img_path_i in img_paths]

def as_text(action_i,out_path):
    def line_helper(frame):
        line=[ str(cord_i) for cord_i in list(frame)]
        return ",".join(line) 
    lines=[line_helper(frame_i) 
            for frame_i in action_i.img_seq]
    text="\n".join(lines)
    utils.save_string(out_path,text)

def as_imgs(action_i,action_path):
    print(action_path)
    utils.make_dir(action_path)
    for j,img_j in enumerate(action_i.img_seq):
        path_ij=action_path+'/img'+str(j)+'.png'
        cv2.imwrite(path_ij,img_j)

def cp_dataset(action_path):
    action_name=action_path.split('/')[-1]
    raw=action_name.split('_')
    if(len(raw)==4):
        cat=raw[0].replace('a','')
        person=int(raw[1].replace('s',''))
        return action_name,cat,person
    raise Exception("Wrong dataset format " + action_name +" " + str(len(raw)))