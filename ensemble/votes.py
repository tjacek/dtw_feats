import dataset,dataset.instances,utils,ensemble.single_cls
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import Counter
from sets import Set

class VotingEnsemble(object):
    def __init__(self,norm=True,basic_feats=250,deep_feats=100,restr=None):
        self.build_dataset=EarlyPreproc(norm,basic_feats,deep_feats,restr)

    def __call__(self,basic_paths,deep_paths):
        if(deep_paths):
            y_true,all_pred=self.build_dataset.all_predictions(basic_paths,deep_paths)
            y_pred=vote(all_pred)
        else:
            self.build_dataset.init(basic_paths)
            y_true,y_pred=ensemble.single_cls.simple_exp(self.build_dataset.basic_dataset)
        print(basic_paths)
        print(deep_paths)
        print(str(self.build_dataset))
        names=self.build_dataset.test_names()
        if(names):
            print(ensemble.single_cls.show_errors(y_true,y_pred,names))
        ensemble.single_cls.show_result(y_true,y_pred)

def vote(all_votes):
    all_votes=np.array(all_votes)
    y_pred=[]
    for vote_i in all_votes.T:
        count =Counter(vote_i)
        cat_i=count.most_common()[0][0]
        y_pred.append(cat_i)
    return y_pred 

class BuildDataset(object):
    def __init__(self, norm,restr=None):
        self.norm = norm
        self.basic_dataset=None
        self.restr=Set(restr)

    def all_predictions(self,basic_paths,deep_path):
        self.init(basic_paths)
        deep_paths=utils.bottom_files(deep_path)
        if(not deep_paths):
            raise Exception("No datasets at " + deep_paths)
        if(self.restr):
            deep_paths=[path_i for i,path_i in enumerate(deep_paths)
                            if(i in self.restr)]
        def pred_helper(feat_path_i):
            print(feat_path_i)
            dataset_i=self.get_dataset(feat_path_i) 
            return ensemble.single_cls.train_model(dataset_i)
        result=[ pred_helper(feat_path_i) 
                    for feat_path_i in deep_paths]
        y_true=result[0][0]
        all_preds=[result_i[1] for result_i in result]
        return y_true,all_preds

    def test_names(self):
        if(self.basic_dataset):
            train,test=self.basic_dataset.split()
            return test.names
        return None

class LatePreproc(BuildDataset):
    def __init__(self,norm=True,n_feats=100,restr=None):   
        super(LatePreproc, self).__init__(norm,restr)
        self.n_feats=n_feats

    def __str__(self):
        n_feats= self.n_feats  if(self.n_feats) else 0
        return "norm:%i n_feats:%i" % (self.norm,n_feats)

    def init(self,basic_paths):
        if(not basic_paths):
            return
        self.basic_dataset=dataset.read_dataset(basic_paths)

    def get_dataset(self,feat_path_i):
        adapt_dataset=dataset.read_dataset(feat_path_i)
        full_dataset=dataset.unify_datasets([adapt_dataset,self.basic_dataset])
        preproc(full_dataset,self.norm,self.n_feats)
        print(full_dataset.X.shape)
        return full_dataset

class EarlyPreproc(BuildDataset):
    def __init__(self,norm=True,basic_feats=150,deep_feats=100,restr=None):   
        super(EarlyPreproc, self).__init__(norm,restr)
        self.basic_feats=basic_feats
        self.deep_feats=deep_feats
    
    def __str__(self):
        basic_feats= self.basic_feats if(self.basic_feats) else 0
        deep_feats= self.deep_feats if(self.deep_feats) else 0
        return "norm:%i basic_feat:%i deep_feats:%i" % (self.norm,basic_feats,deep_feats)

    def init(self,basic_paths):
        if(not basic_paths):
            return
        print(basic_paths)
        self.basic_dataset=dataset.read_dataset(basic_paths)
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