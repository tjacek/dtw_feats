import numpy as np
import utils,deep,deep.convnet,deep.train
import ensemble,pairs,utils
import theano.gpuarray

def simple_exp(dataset_path='data/MSR',nn_path='data/nn',
                compile=False,n_frames=4):
    preproc=deep.ImgPreproc(n_frames)
    X,y,n_cats=deep.convnet.get_dataset(dataset_path,preproc)
    model_path=None if(compile) else nn_path
    model=deep.convnet.get_model(n_cats,preproc,nn_path=model_path)
    deep.train.train_super_model(X,y,model,num_iter=25)
    model.get_model().save(nn_path)

def ensemble_exp(compile=False,n_frames=4,n_iters=150):
    preproc=deep.ImgPreproc(n_frames)
    X,y,n_cats=deep.convnet.get_dataset(dataset_path,preproc)
    def in_ensemble(in_path):
        model_path=None if(compile) else nn_path_j        
        model_j=deep.convnet.get_model(2,preproc,nn_path=model_path)
        y_j=binarize(y,j)
        deep.train.train_super_model(X,y_j,model_j,num_iter=n_iters)
        return model_j
    def out_ensemble(nn_path_j,result_i):
        model_j.get_model().save(nn_path_j)
    return ensemble.EnsembleFun(in_ensemble,out_ensemble,gen_paths=n_cats)

def ensemble_pairs(in_path='mhad/feats',out_path='mhad/deep_pairs'):
    deep_paths=utils.bottom_dirs(in_path)
    deep_paths=deep_paths[2:]
    print(deep_paths)
    utils.make_dir(out_path)
    for in_i in deep_paths:
        out_i=ensemble.get_out_path(in_i,out_path)
        print(out_i)
        pairs.compute_pairs(in_i,out_i)

ensemble_pairs(in_path='mhad/feats')
#simple_exp(dataset_path='data/MSR',nn_path='data/nn',compile=False)
#ensemble_exp(dataset_path='data/train',nn_path='data/models',compile=False)
#ensemble.extract_deep(data_path='data/MSR',nn_path="data/models",out_path="data/feats")
#ensemble.global_feats("mhad/feats","mhad/datasets")
