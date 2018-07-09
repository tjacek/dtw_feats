import numpy as np
import seqs.io 
import lasagne
import pickle

class NeuralNetwork(object):
    def __init__(self,hyperparams,out_layer):
        self.hyperparams=hyperparams
        self.out_layer=out_layer

    def get_model(self):
        type_name=type(self).__name__
        data = lasagne.layers.get_all_param_values(self.out_layer)
        return Model(self.hyperparams,data,type_name)
    
    def set_model(self,model):
        lasagne.layers.set_all_param_values(self.out_layer,model.params)

    def __str__(self):
        return str(self.hyperparams)

class Model(object):
    def __init__(self,hyperparams,params,type_name='Convet'):
        self.params=params
        self.hyperparams=hyperparams
        self.type_name=type_name

    def save(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def set_determistic(self):
        if('p' in self.hyperparams):
            self.hyperparams['p']=0.0

    def add_empty_params(self,param_shape):
        print(type(param_shape))
        print(param_shape)
        self.params.append(np.ones(param_shape))


class ImgPreproc(object):
    def __init__(self,dim):
        self.dim=dim

    def __call__(self,img_i):
        height=img_i.shape[0]
        new_height=self.get_new_height(height)
        frames=np.vsplit(img_i,self.dim) 
        splited_img=np.stack(frames)
        return splited_img

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

def count_cats(y):
    cats=np.unique(y)
    return cats.shape[0] 