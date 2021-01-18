import numpy as np
import feats
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report,accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
import clf,files,feats

class Result(object):
    def __init__(self,y_true,y_pred,names):
        if(type(y_pred)==list):
            y_pred=np.array(y_pred)
        self.y_true=y_true
        self.y_pred=y_pred
        self.names=names

    def n_cats(self):
        votes=self.as_numpy()
        return votes.shape[1]

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

    def as_hard_votes(self):
        hard_pred=[]
        n_cats=self.n_cats()
        for y_i in self.y_pred:
            hard_i=np.zeros((n_cats,))
            hard_i[np.argmax(y_i)]=1
            hard_pred.append(hard_i)
        return np.array(hard_pred)
   
    def get_cf(self,out_path=None):
        y_true,y_pred=self.as_labels()
        cf_matrix=confusion_matrix(y_true,y_pred)
        if(out_path):
            np.savetxt(out_path,cf_matrix,delimiter=",",fmt='%.2e')
        return cf_matrix

    def get_acc(self):
        y_true,y_pred=self.as_labels()
        return accuracy_score(y_true,y_pred)

    def report(self):
        y_true,y_pred=self.as_labels()
        print(classification_report(y_true, y_pred,digits=4))

    def metrics(self):
        y_true,y_pred=self.as_labels()
        return precision_recall_fscore_support(y_true,y_pred,average='weighted')

    def get_errors(self):
        errors=[]
        y_true,y_pred=self.as_labels()
        for i,y_i in enumerate(y_true):
            if(y_i!=y_pred[i]):
                errors.append( (y_i,y_pred[i],self.names[i]))
        return errors

    def cat_acc(self,cat_i):
        y_true,y_pred=self.as_labels()
        cat_pred=[]
        for j,y_j in enumerate(y_true): 
            if(y_j==cat_i):
                cat_pred.append(int(y_j==y_pred[j]))
        return np.mean(cat_pred)

    def save(self,out_path):
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)

def read(in_path):
    with open(in_path, 'rb') as handle:
        return pickle.load(handle)

def train_ens(datasets,clf="LR"):
    return [train_model(data_i,clf_type=clf) for data_i in datasets]

def train_model(data,binary=False,clf_type="LR"):
    if(type(data)==str):	
        data=feats.read(data)[0]
    data.norm()
    print(data.dim())
    print(len(data))
    train,test=data.split()
    model=make_model(train,clf_type)
    X_test,y_true=test.get_X(),test.get_labels()
#    X_train=np.nan_to_num(X_train)
    if(binary):
        y_pred=model.predict(X_test)
    else:
        y_pred=model.predict_proba(X_test)
    return Result(y_true,y_pred,test.names())

def make_model(train,clf_type):
    model= clf.get_cls(clf_type)
    X_train,y_train= train.get_X(),train.get_labels()
    model.fit(X_train,y_train)
    return model

def voting(results,binary):
    votes=get_prob(results)
    votes=np.sum(votes,axis=0)
    y_pred=[np.argmax(vote_i) for vote_i in votes]
    return y_pred

def get_prob(results):
    return np.array([result_i[1] for result_i in results])

#def get_acc(paths,clf="LR"):
#    datasets=tools.read_datasets(paths)
#    return [train_model(data_i,True,clf,True) 
#                for data_i in datasets]