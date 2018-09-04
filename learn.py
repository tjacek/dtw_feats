import numpy as np
import utils
import deep,deep.convnet
import deep.train,deep.autoconv
import deep.tools
import ensemble,pairs
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

def train_autoencoder(dataset_path,nn_path,n_frames=2,n_iters=5,read=True):
    preproc=deep.ImgPreproc(n_frames)
    X,y,n_cats=deep.get_dataset(dataset_path,preproc)
    read_path=  nn_path if(read) else None
    ae_model=deep.autoconv.get_model(preproc=preproc,nn_path=read_path)
    deep.train.train_unsuper_model(X,ae_model,num_iter=n_iters)
    ae_model.get_model().save(nn_path)

def train_convnet(out_path,dataset_path,
                        compile=False,n_frames=4,n_iters=1000,n_binarize=None):
    model_path=None if(compile) else out_path
    preproc=deep.ImgPreproc(n_frames)
    X,y,n_cats=deep.tools.get_dataset(dataset_path,preproc)
    if(n_binarize):
        y=binarize(y,n_binarize)
    model_j=deep.convnet.get_model(n_cats,preproc,nn_path=model_path)
    deep.train.train_super_model(X,y,model_j,num_iter=n_iters)
    model_j.get_model().save(out_path)

deep.tools.deep_seqs(in_path='../mhad/four/full',nn_path='../mhad/four/conv',
                out_path='../mhad/four/seqs',n_frames=4)
#deep.autoconv.reconstruct_actions(in_path='mhad/test',
#                                nn_path='mhad/ae',out_path='mhad/rec')
#train_autoencoder('mhad/test','mhad/ae')