import numpy as np 
import utils
import seqs.select,seqs.io

class Action(object):
    def __init__(self,img_seq,name,cat,person):
        self.img_seq=img_seq
        self.name=name
        self.cat=cat
        self.person=person
    
    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.img_seq)
    
    def __call__(self,fun,whole_seq=True):
        print(self.name)
        if(whole_seq):
            new_seq=fun(self.img_seq)
        else:
            new_seq=[ fun(img_i) for img_i in self.img_seq]
        return Action(new_seq,self.name,self.cat,self.person)	
    
    def clone(self,img_seq):
        return Action(img_seq,self.name,self.cat,self.person)

    def dim(self):
        frame=self.img_seq[0]
        if(type(frame)==list):
            return len(frame)
        return frame.shape[0]

    def as_array(self):
        return np.array(self.img_seq)

    def as_features(self):
        action_array=self.as_array().T
        return [ feature_i for feature_i in action_array]

    def as_pairs(self,norm=255.0):
        norm_imgs=[ (img_i/norm) 
                    for img_i in self.img_seq]
        return [ (self.cat,img_i) for img_i in norm_imgs]

def norm_actions(out_path,in_path):
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_actions(in_path)
    feats=[]
    for action_i in actions:
        feats+=action_i.img_seq
    feats=np.array(feats)
    mean_feats=np.mean(feats,axis=0)
    std_feats=np.std(feats,axis=0)
    for action_i in actions:
        img_seq_i=action_i.as_array()
        img_seq_i-=mean_feats
        img_seq_i/=std_feats
        action_i.img_seq=list(img_seq_i)
    save_actions=seqs.io.ActionWriter(img_seq=False)
    save_actions(actions,out_path)
    return actions

def split(actions,selector=None):
    train,test=[],[]
    if(not selector):
        selector=seqs.select.ModuloSelector(1)
    for action_i in actions:
        if(selector(action_i)):
            train.append(action_i)
        else:
            test.append(action_i)
    return train,test

def by_cat(actions):
    cats=[action_i.cat for action_i in actions]
    actions_by_cat={ cat_i:[] for cat_i in np.unique(cats)}
    for action_i in actions:
        actions_by_cat[action_i.cat].append(action_i)
    return actions_by_cat

def person_rep(actions):
    reps={}
    for action_i in actions:
        action_id=str(action_i.cat)+str(action_i.person)
        if(not action_id in reps):
            reps[action_id]=action_i
    return reps.values()   