import numpy as np
import seqs,seqs.io
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

def make_params(x,y):
    max_seq = max([x_i.shape[0] for x_i in x])
    n_batch = len(x)
    n_cats=count_cats(y)
    seq_dim=x[0].shape[1]
    params={'n_batch':n_batch,'max_seq':max_seq,'seq_dim':seq_dim,'n_cats':n_cats}
    return params

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

def lstm_dataset(seq_path):
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_actions(seq_path)
    train,test=seqs.split(actions)
    train,test=make_dataset(train,masked=True),make_dataset(test,masked=True)
    return train,test

def make_dataset(actions,masked=False):
    x=[np.array(action_i.img_seq)
        for action_i in actions]
    x=np.array(x)
    y=[int(action_i.cat) for action_i in actions]
    names=[action_i.name for action_i in actions]
    persons=[action_i.person for action_i in actions]
#    y=[ e(y_i) for y_i in y]
    basic_dataset={'x':x ,'y':y,'names':names,'persons':persons}
    if(masked):
        return masked_dataset(basic_dataset)
    else:
        return basic_dataset

def masked_dataset(dataset):
    x=dataset['x']
    y=dataset['y']
    names=dataset['names']
    params= make_params(x,y)
    mask=make_mask(x,params['n_batch'],params['max_seq'])
    x_masked=make_masked_seq(x,params['max_seq'],params['seq_dim'])
    new_dataset={'x':x_masked,'y':dataset['y'],'mask':mask,
                 'persons':dataset['persons'],'params':params,
                 'names':names}
    return new_dataset#SeqDataset(new_dataset)

def make_mask(x,n_batch,max_seq):
    mask = np.zeros((n_batch, max_seq),dtype=float)
    for i,seq_i in enumerate(x):
        seq_i_len=seq_i.shape[0]
        mask[i][:seq_i_len]=1.0
    return mask

def make_masked_seq(x,max_seq,seq_dim):
    
    def masked_seq(seq_i):
        if(seq_i.shape[0]>max_seq):
            return np.ones( (seq_i.shape[0],seq_dim))
        seq_i_len= seq_i.shape[0]#utils.data.seq_len(seq_i)
        new_seq_i=np.zeros((max_seq,seq_dim))
        new_seq_i[:seq_i_len]=seq_i[:seq_i_len]
        return new_seq_i
    return [masked_seq(seq_i) for seq_i in x]   