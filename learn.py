import numpy as np
import deep,deep.convnet,deep.train
#import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu")
import theano.gpuarray
#theano.gpuarray.use("gpu")
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

simple_exp(dataset_path='data/MSR',nn_path='data/nn',compile=False)