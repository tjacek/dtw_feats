import numpy as np 
import utils

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
        if(whole_seq):
            new_seq=fun(self.img_seq)
        else:
            new_seq=[ fun(img_i) for img_i in self.img_seq]
        return Action(new_seq,self.name,self.cat,self.person)	
    
    def dim(self):
        frame=self.img_seq[0]
        if(type(frame)==list):
            return len(frame)
        return frame.shape

    def as_array(self):
        return np.array(self.img_seq)

    def as_feature(self):
        action_array=self.as_array().T
        return [ feature_i for feature_i in action_array]