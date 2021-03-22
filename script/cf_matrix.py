import numpy as np
import os,os.path
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def show_all(in_path,out_path):
    if(not os.path.isdir(out_path)):
        os.mkdir(out_path)
    if(not os.path.isdir(in_path)):
        title=in_path.split("/")[-1]
        show_confusion(in_path,title=title,out_path=out_path)
    all_paths=os.listdir(in_path)
    for path_i in all_paths:
        in_i=in_path+'/'+path_i
        out_i=out_path+'/'+path_i
        print(out_i)
        show_confusion(in_path=in_i,title=path_i,out_path=out_i)

def show_confusion(in_path,labels=None,title=None,out_path=None):       	
    plt.clf()
    cf_matrix=np.genfromtxt(in_path,delimiter=',')
    dim=cf_matrix.shape
    if(not labels):
        labels=range(dim[0])
    df_cm = pd.DataFrame(cf_matrix, labels,labels)
#    sn.set(font_scale=1.0)
    sn.heatmap(df_cm,cmap="YlGnBu",#linewidths=0.5,
    	annot=True,annot_kws={"size": 9}, fmt='g')
    if(title):
        plt.title(title)
    b, t = plt.ylim()
    b += 0.5 
    t -= 0.5 
    plt.ylim(b, t)
    if(all( type(i) == int for i in labels)):
        plt.xlabel("predicted labels")
        plt.ylabel("true labels")
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90)
    if(out_path):
        plt.savefig(out_path)
    else:
        plt.show()


cats_msr=['high arm wave','horizontal arm wave','hammer','hand catch','forward punch',
          'high throw','draw x','draw tick','draw circle','hand clap',
          'two hand wave','side-boxing','bend','forward kick','side kick',
          'jogging','tennis swing','tennis serve','golfswing','pick up & throw']

cats_mhad=['right arm swipe to the left', 'right arm swipe to the right', 'right hand wave','two hand front clap','right arm throw',
 'cross arms in the chest','basketball shoot','right hand draw x', 'draw circle (clockwise)','draw circle (counter clockwise)', 
 'draw triangle','bowling', 'front boxing', 'baseball swing from right', 'tennis right hand forehand swing','arm curl', 
 'tennis serve', 'two hand push', 'right hand knock on door', 'right hand catch an object', 'right hand pick up and throw', 'jogging in place', 
 'walking in place', 'sit to stand', 'stand to sit','forward lunge','squat']

#show_confusion("cf/pairs",None,"pairs")
show_all("raw","plots")
