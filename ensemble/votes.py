import dataset,dataset.instances,utils
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter

class VotingEnsemble(object):
    def __init__(self,norm=True,select=None):
        self.norm=norm
        self.select=select
        self.basic_feats=None

    def __call__(self,basic_paths,deep_path):
        y_true,all_pred=self.all_predictions(basic_paths,deep_path)
        y_pred=self.vote(all_pred)
        indep_votes=indepen_measure(y_true,all_pred)
        indep=np.sum(indep_votes,axis=0).astype(float)
        indep/=len(y_true)
        print(list(indep))  
        print(classification_report(y_true, y_pred,digits=4))
    
    def all_predictions(self,basic_paths,deep_path):
        self.basic_feats=get_basic_dataset(basic_paths)
        deep_paths=utils.bottom_files(deep_path)

        all_pred=[]
        y_true=None
        for feat_path_i in deep_paths:
            dataset_i=self.get_dataset(feat_path_i) 
            y_i,y_pred=self.get_model(dataset_i)
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
    
    def get_dataset(self,feat_path_i):
        adapt_dataset=dataset.read_dataset(feat_path_i)
        full_dataset=dataset.unify_datasets([adapt_dataset,self.basic_feats])
        if(self.norm):
            full_dataset.norm()
        if(not self.select is None ):
            full_dataset.select(self.select)
        return full_dataset

    def get_model(self,dataset_i):
        train,test=dataset_i.split()
        clf=LogisticRegression()
        clf = clf.fit(train.X, train.y)
        y_pred = clf.predict(test.X)
        return test.y,y_pred

def get_basic_dataset(basic_paths):
    if(len(basic_paths)==0):
        return None
    datasets=[dataset.read_dataset(basic_i) 
                for basic_i in basic_paths]
    return dataset.unify_datasets(datasets)

def indepen_measure(y_true,all_pred):
    all_pred=np.array( all_pred).T
    def indepen_helper(i,pred_i,pred_ij):
        if(y_true[i]==pred_ij):
            print(i)
            return 0
        bool_array=(pred_i==pred_ij)
        indepen_ij=sum(bool_array.astype(int))
        return indepen_ij-1

    indep=[ [indepen_helper(i,pred_i,pred_ij)
                for pred_ij in pred_i]
                    for i,pred_i in enumerate(all_pred)]
    indep=np.matrix(indep) 
    print(indep.shape)
#    for i,y_i in enumerate(y_true):
#        indep[i,y_i-1]=
    return indep