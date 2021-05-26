import os
import numpy as np

def read_data(in_path,sep=" "):
    paths=["%s/%s" % (in_path,name_i) 
            for name_i in os.listdir(in_path)]
    return [read_dataset(path_i,sep) for path_i in paths]

def read_dataset(path_i,sep=" "):
    file_i = open(path_i, 'r')
    X_train,y_train,X_test,y_test=[],[],[],[]
    for line_i in file_i.readlines():
        raw_x,raw_y=line_i.split("#")
        raw_x=raw_x.replace('[','').replace(']','')
        raw_x=raw_x.replace("'",'')
        x_i=np.array(raw_x.split(sep)).astype(np.float64)
        y_i=int(raw_y.split('_')[0])
        person_i=int(raw_y.split('_')[1])
        if( (person_i % 2)==1):
            X_train.append(x_i)
            y_train.append(y_i)
        else:
            X_test.append(x_i)
            y_test.append(y_i)
    return X_train,y_train,X_test,y_test 

vote_path="3DHOI_data/votes"
votes=read_data(vote_path,' ')
features_path="3DHOI_data/features"
features=read_data(features_path,',')