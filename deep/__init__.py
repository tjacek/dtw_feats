import numpy as np
import seqs.io 

class ImgPreproc(object):
    def __init__(self,dim):
        self.dim=dim

    def __call__(self,imgset):
        x,y=get_dims(imgset)
        new_height=self.get_new_height(x)
        def reshape_helper(img_i):
            frames=np.vsplit(img_i,self.dim) 
            splited_img=np.stack(frames)
            print(np.max(splited_img))
            return splited_img
        return [ reshape_helper(img_i) for img_i in imgset]

    def get_new_height(self,x):
        if(x % self.dim != 0):
            raise Exception("incorrect img size")
        return x/self.dim

def get_dims(imgset):
    first=imgset[0]
    return first.shape[0],first.shape[1]

def read_dataset(in_path):
    read_actions=seqs.io.build_action_reader(img_seq=True,as_dict=False)
    actions=read_actions(in_path)
    pairs=[]
    for action_i in actions:
        pairs+=action_i.as_pairs()
    y=[pair_i[0] for pair_i in pairs]
    x=[pair_i[1] for pair_i in pairs]
    return np.array(x),y 