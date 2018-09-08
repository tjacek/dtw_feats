import numpy as np
import seqs.io
import deep.reader

class ImgPreproc(object):
    def __init__(self,dim):
        self.dim=dim

    def __call__(self,img_i):
        if(type(img_i)==list):
            return [ self.apply(img_j) for img_j in img_i]
        else:
            return self.apply(img_i)

    def apply(self,img_i):
        height=img_i.shape[0]
        new_height=self.get_new_height(height)
        frames=np.vsplit(img_i,self.dim) 
        splited_img=np.stack(frames)
        return splited_img

    def get_new_height(self,x):
        if(x % self.dim != 0):
            raise Exception("incorrect img size")
        return x/self.dim

    def postproc(self,imgs):
        return [np.concatenate(img_i,axis=0)
                    for img_i in imgs]

def get_dataset(in_path,preproc):
    X,y=read_dataset(in_path)
    y=np.array(y)
    y-= np.min(y)
    n_cats=count_cats(y)
    X=np.array([preproc(x_i) for x_i in X])
    return X,y,n_cats

def binarize(y,cat_j):
    return [ 0 if(y_i==cat_j) else 1
               for y_i in y]

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

def count_cats(y):
    cats=np.unique(y)
    return cats.shape[0]

def deep_seqs(in_path,nn_path,out_path,n_frames=4):
    preproc=ImgPreproc(n_frames)
    deep_reader= deep.reader.NNReader(preproc)
    nn_transform=deep_reader(nn_path)
    seqs.io.transform_actions(in_path,out_path,transform=nn_transform,
                      img_in=True,img_out=False,whole_seq=False)

def check_nn(nn_path,dataset_path,n_frames):
    preproc=ImgPreproc(n_frames)
    deep_reader= deep.reader.NNReader(preproc)
    convnet=deep_reader(nn_path)
    X,true_y=read_dataset(in_path)
    pred_y=[convnet.get_category(x_i) for x_i in X]
    hist=np.zeros((count_cats(true_y),count_cats(pred_y)))
    for true_i,pred_i in zip(true_y,pred_y):
        hist[true_i][pred_i]+=1
    return hist