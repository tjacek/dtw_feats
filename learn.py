import numpy as np
import utils,time
import deep,deep.convnet,deep.lstm
import deep.train,deep.autoconv
import deep.tools,pairs
import ensemble,pairs,seqs.io,seqs
import theano.gpuarray

def ensemble_exp(dataset_path,compile=False,n_frames=4,n_iters=150):
    preproc=deep.tools.ImgPreproc(n_frames)
    X,y,n_cats=deep.tools.get_dataset(dataset_path,preproc)
    def in_fun(in_path):
        model_path=None if(compile) else nn_path_j        
        model_j=deep.convnet.get_model(n_frames,preproc,nn_path=model_path)
        name_j=utils.get_name(in_path)
        j=utils.extract_numbers(name_j)
        y_j=deep.tools.binarize(y,j)
        deep.train.train_super_model(X,y_j,model_j,num_iter=n_iters)
        return model_j
    def out_fun(nn_path_j,model_j):
        model_j.get_model().save(nn_path_j)
    return ensemble.EnsembleFun(in_fun,out_fun,n_paths=n_cats)

def extract_deep(dataset_path,n_frames=4):
    deep_reader= deep.reader.NNReader(n_frames)
    def deep_helper(path_i):
        return deep_reader(path_i)
    def out_fun(out_i,deep_nn):    
        seqs.io.transform_actions(dataset_path,out_i,transform=deep_nn,
                      img_in=True,img_out=False,whole_seq=False)
    return ensemble.EnsembleFun(deep_helper,out_fun)

def ensemble_pairs():
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=True)
    def in_fun(in_path_i):
        print(in_path_i)
        actions=read_actions(in_path_i)
        t0=time.time()
        result_i=pairs.make_pairwise_distance(actions)
        print("pairs computation %d" % (time.time()-t0))    
        return result_i
    def out_fun(out_path_i,result_i):
        result_i.save(out_path_i)
    return ensemble.EnsembleFun(in_fun,out_fun,'dirs')

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
        y=deep.tools.binarize(y,n_binarize)
    model_j=deep.convnet.get_model(n_cats,preproc,nn_path=model_path)
    deep.train.train_super_model(X,y,model_j,num_iter=n_iters)
    model_j.get_model().save(out_path)

def train_lstm(seq_path,p=0.25):
    train,test=deep.tools.lstm_dataset(seq_path)
    print(train['params'])
    hyper_params=deep.lstm.get_hyper_params(train)
    hyper_params['p']=p
#    hyper_params['n_hidden']= train[0].dim()
    model=deep.lstm.compile_lstm(hyper_params)
    deep.train.train_seq(model,train,epochs=300)
    deep.tools.check_lstm(model,test)

train_lstm("../LSTM/all")
#ens=extract_deep(dataset_path='../../mhad/data/full')
#ens('../../mhad/models','../../mhad/seqs')
#ens=ensemble_pairs()
#ens('../../mhad/seqs','../../mhad/pairs')
#ens=ensemble_exp(dataset_path='../../mhad/data/train',compile=True,n_frames=4,n_iters=1000)
#ens(in_path='../../mhad/data/train',out_path='../../mhad/models')
#deep.tools.deep_seqs(in_path='../mhad/four/full',nn_path='../mhad/four/conv',
#                out_path='../mhad/four/seqs',n_frames=4)
#deep.autoconv.reconstruct_actions(in_path='mhad/test',
#                                nn_path='mhad/ae',out_path='mhad/rec')
#train_autoencoder('mhad/test','mhad/ae')