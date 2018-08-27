import numpy as np
import utils,deep,deep.convnet
import deep.train,deep.autoconv
import ensemble,pairs,utils
import theano.gpuarray

def ensemble_exp(compile=False,n_frames=4,n_iters=150):
    preproc=deep.ImgPreproc(n_frames)
    X,y,n_cats=deep.convnet.get_dataset(dataset_path,preproc)
    def in_fun(in_path):
        model_path=None if(compile) else nn_path_j        
        model_j=deep.get_model(2,preproc,nn_path=model_path)
        y_j=binarize(y,j)
        deep.train.train_super_model(X,y_j,model_j,num_iter=n_iters)
        return model_j
    def out_fun(nn_path_j,result_i):
        model_j.get_model().save(nn_path_j)
    return ensemble.EnsembleFun(in_fun,out_fun,gen_paths=n_cats)

def ensemble_pairs(in_path='mhad/feats',out_path='mhad/deep_pairs'):
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=True)
    def in_fun(in_path_i):
        actions=read_actions(in_path)
        t0=time.time()
        result_i=make_pairwise_distance(actions)
        print("pairs computation %d" % (time.time()-t0))    
        return result_i
    def out_fun(out_path_i,result_i):
        lines_i=as_txt(result_i)
        utils.save_string(out_path_i,lines_i)
    return ensemble.EnsembleFun(in_fun,out_fun)

def train_autoencoder(dataset_path, n_frames=2,n_iters=5):
    preproc=deep.ImgPreproc(n_frames)
    X,y,n_cats=deep.get_dataset(dataset_path,preproc)
    ae_model=deep.autoconv.get_model(preproc=preproc)
    deep.train.train_unsuper_model(X,ae_model,num_iter=n_iters)

train_autoencoder(dataset_path='mhad/test')
#ensemble_pairs(in_path='mhad/feats')