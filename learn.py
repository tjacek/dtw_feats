import numpy as np
import deep,deep.convnet,deep.train
import utils
import theano.gpuarray

def get_dataset(in_path,preproc):
    X,y=deep.read_dataset(in_path)
    y=np.array(y)
    y-= np.min(y)
    n_cats=deep.count_cats(y)
    X=np.array([preproc(x_i) for x_i in X])
    return X,y,n_cats

def simple_exp(dataset_path='data/MSR',nn_path='data/nn',compile=False):
    preproc=deep.ImgPreproc(4)
    X,y,n_cats=get_dataset(dataset_path,preproc)
    model_path=None if(compile) else nn_path
    model=deep.convnet.get_model(n_cats,preproc,nn_path=model_path)
    deep.train.train_super_model(X,y,model,num_iter=25)
    model.get_model().save(nn_path)

def ensemble_exp(dataset_path='data/MSR',nn_path='data/nn',compile=False):
    preproc=deep.ImgPreproc(4)
    X,y,n_cats=get_dataset(dataset_path,preproc)
    utils.make_dir(nn_path)
    for j in range(n_cats):
        nn_path_j=nn_path+'/nn'+str(j)
        print(nn_path_j)
        model_path=None if(compile) else nn_path_j        
        model_j=deep.convnet.get_model(2,preproc,nn_path=model_path)
        y_j=binarize(y,j)
        deep.train.train_super_model(X,y_j,model_j,num_iter=150)
        model_j.get_model().save(nn_path_j)

def binarize(y,cat_j):
    return [ 0 if(y_i==cat_j) else 1
               for y_i in y]

#simple_exp(dataset_path='data/MSR',nn_path='data/nn',compile=False)
ensemble_exp(dataset_path='data/train',nn_path='data/models',compile=False)