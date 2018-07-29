import numpy as np
import utils,deep,deep.convnet,deep.train
import ensemble
import theano.gpuarray

def simple_exp(dataset_path='data/MSR',nn_path='data/nn',
                compile=False,n_frames=4):
    preproc=deep.ImgPreproc(n_frames)
    X,y,n_cats=deep.convnet.get_dataset(dataset_path,preproc)
    model_path=None if(compile) else nn_path
    model=deep.convnet.get_model(n_cats,preproc,nn_path=model_path)
    deep.train.train_super_model(X,y,model,num_iter=25)
    model.get_model().save(nn_path)

def ensemble_exp(dataset_path='data/MSR',nn_path='data/nn',
                    compile=False,n_frames=4):
    preproc=deep.ImgPreproc(n_frames)
    X,y,n_cats=deep.convnet.get_dataset(dataset_path,preproc)
    utils.make_dir(nn_path)
    for j in range(n_cats):
        nn_path_j=nn_path+'/nn'+str(j)
        print(nn_path_j)
        model_path=None if(compile) else nn_path_j        
        model_j=deep.convnet.get_model(2,preproc,nn_path=model_path)
        y_j=binarize(y,j)
        deep.train.train_super_model(X,y_j,model_j,num_iter=150)
        model_j.get_model().save(nn_path_j)

#simple_exp(dataset_path='data/MSR',nn_path='data/nn',compile=False)
#ensemble_exp(dataset_path='data/train',nn_path='data/models',compile=False)
#ensemble.extract_deep(data_path='data/MSR',nn_path="data/models",out_path="data/feats")
#ensemble.global_feats("mhad/feats","mhad/datasets")
