import dataset
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def simple_exp(in_path):
    full_dataset=dataset.read_dataset(in_path)
    full_dataset.norm()
    train,test=full_dataset.split()
    clf=LogisticRegression()
    clf = clf.fit(train.X, train.y)
    y_true, y_pred = test.y, clf.predict(test.X)
    print(classification_report(y_true, y_pred,digits=4))

def show_result(y_true,y_pred):
    print(classification_report(y_true, y_pred,digits=4))
    print("Accuracy %f " % accuracy_score(y_true,y_pred))
    cf=confusion_matrix(y_true, y_pred)
    cf_matrix=pd.DataFrame(cf,index=range(cf.shape[0]))
    print(cf_matrix)

