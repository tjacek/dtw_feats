import dataset,dataset.instances,utils,ensemble.single_cls
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import Counter

class VotingEnsemble(object):
    def __init__(self,norm=True,basic_feats=250,deep_feats=100):
        self.build_dataset=EarlyPreproc(norm,basic_feats,deep_feats)

    def __call__(self,basic_paths,deep_paths):
        if(deep_paths):
            y_true,all_pred=self.all_predictions(basic_paths,deep_paths)
            y_pred=self.vote(all_pred)
        else:
            self.build_dataset.init(basic_paths)
            y_true,y_pred=ensemble.single_cls.simple_exp(self.build_dataset.basic_dataset)
        print(basic_paths)
        print(deep_paths)
        print(str(self.build_dataset))
        ensemble.single_cls.show_result(y_true,y_pred)

    def all_predictions(self,basic_paths,deep_path):
        self.build_dataset.init(basic_paths)
        deep_paths=utils.bottom_files(deep_path)
        all_pred=[]
        y_true=None
        for feat_path_i in deep_paths:
            dataset_i=self.build_dataset.get_dataset(feat_path_i) 
            y_i,y_pred=train_model(dataset_i)
            if(y_true is None):
                y_true=y_i
            all_pred.append(y_pred)
        return y_true,all_pred

    def vote(self,all_votes):
        all_votes=np.array(all_votes)
        y_pred=[]
        for vote_i in all_votes.T:
            count =Counter(vote_i)
            cat_i=count.most_common()[0][0]
            y_pred.append(cat_i)
        return y_pred 

class LatePreproc(object):
    def __init__(self,norm=True,n_feats=100):   
        self.norm=norm
        self.n_feats=n_feats
        self.basic_dataset=None

    def __str__(self):
        n_feats= self.n_feats  if(self.n_feats) else 0
        return "norm:%i n_feats:%i" % (self.norm,n_feats)

    def init(self,basic_paths):
        if(len(basic_paths)==0):
            print("No basic dataset")
            self.basic_feats=None
        datasets=[dataset.read_dataset(basic_i) 
                for basic_i in basic_paths]
        self.basic_dataset=dataset.unify_datasets(datasets)

    def get_dataset(self,feat_path_i):
        print(feat_path_i)
        adapt_dataset=dataset.read_dataset(feat_path_i)
        full_dataset=dataset.unify_datasets([adapt_dataset,self.basic_dataset])
        preproc(full_dataset,self.norm,self.n_feats)
        print(full_dataset.X.shape)
        return full_dataset

class EarlyPreproc(object):
    def __init__(self,norm=True,basic_feats=150,deep_feats=100):   
        self.norm=norm
        self.basic_feats=basic_feats
        self.deep_feats=deep_feats
        self.basic_dataset=None
    
    def __str__(self):
        basic_feats= self.basic_feats if(self.basic_feats) else 0
        deep_feats= self.deep_feats if(self.deep_feats) else 0
        return "norm:%i basic_feat:%i deep_feats:%i" % (self.norm,basic_feats,deep_feats)

    def init(self,basic_paths):
        if(len(basic_paths)==0):
            self.basic_feats=None
        datasets=[dataset.read_dataset(basic_i) 
                for basic_i in basic_paths]
        self.basic_dataset=dataset.unify_datasets(datasets)
        preproc(self.basic_dataset,self.norm,self.basic_feats)

    def get_dataset(self,feat_path_i):
        adapt_dataset=dataset.read_dataset(feat_path_i)
        preproc(adapt_dataset,self.norm,self.deep_feats)
        full_dataset=dataset.unify_datasets([adapt_dataset,self.basic_dataset])
        return full_dataset    

def preproc(dataset_i,norm,select):
    if(norm):
        dataset_i.norm()
    if(not select is None ):
        dataset_i.select(select)