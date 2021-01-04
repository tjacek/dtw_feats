import numpy as np
import feats
from sklearn.metrics import accuracy_score
import clf,files,feats

class Result(object):
    def __init__(self,y_true,y_pred,names):
        self.y_true=y_true
        self.y_pred=y_pred
        self.names=names

    def as_numpy(self):
        if(self.y_pred.ndim==2):
            return self.y_pred
        else:           
            print(len(self.y_pred))
            n_cats=np.amax(self.y_true)+1
            votes=np.zeros((len(self.y_true),n_cats))
            for  i,vote_i in enumerate(self.y_pred):
                votes[i,vote_i]=1
            return votes
    
    def as_labels(self):
        if(self.y_pred.ndim==2):
            pred=np.argmax(self.y_pred,axis=1)
        else:
            pred=self.y_pred
        return self.y_true,pred

    def get_acc(self):
        y_true,y_pred=self.as_labels()
        return accuracy_score(y_true,y_pred)

    def report(self):
        y_true,y_pred=self.as_labels()
        print(classification_report(y_true, y_pred,digits=4))

def train_model(data,binary=False,clf_type="LR",acc_only=False):
    if(type(data)==str):	
        data=feats.read(data)[0]
#    data.norm()
    print(data.dim())
    print(len(data))
    train,test=data.split()
    model= clf.get_cls(clf_type)
    X_train,y_train= train.get_X(),train.get_labels()
    model.fit(X_train,y_train)
    X_test,y_true=test.get_X(),test.get_labels()
    if(binary):
        y_pred=model.predict(X_test)
    else:
        y_pred=model.predict_proba(X_test)
    return Result(y_true,y_pred,test.names())
#    if(acc_only):
#        return accuracy_score(y_true,y_pred)
#    else:
#        return y_true,y_pred,test.info

def voting(results,binary):
    votes=get_prob(results)
    votes=np.sum(votes,axis=0)
    y_pred=[np.argmax(vote_i) for vote_i in votes]
    return y_pred

def get_prob(results):
    return np.array([result_i[1] for result_i in results])

def get_acc(paths,clf="LR"):
    datasets=tools.read_datasets(paths)
    return [train_model(data_i,True,clf,True) 
                for data_i in datasets]