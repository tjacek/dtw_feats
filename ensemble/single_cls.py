import dataset
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def simple_exp(in_path):
    if(type(in_path)==str):
        full_dataset=dataset.read_dataset(in_path)
    else:
        full_dataset=in_path
    print("dataset dim %i" % full_dataset.dim())
    full_dataset.norm()
    y_true,y_pred=train_model(full_dataset)
    show_result(y_true,y_pred)

def show_result(y_true,y_pred):
    print(classification_report(y_true, y_pred,digits=4))
    print("Accuracy %f " % accuracy_score(y_true,y_pred))
    cf=confusion_matrix(y_true, y_pred)
    cf_matrix=pd.DataFrame(cf,index=range(cf.shape[0]))
    print(cf_matrix)

def train_model(dataset_i):
    train,test=dataset_i.split()
    clf=LogisticRegression()
    clf = clf.fit(train.X, train.y)
    y_pred = clf.predict(test.X)
    return test.y,y_pred