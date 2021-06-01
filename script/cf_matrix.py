import numpy as np
import os,os.path
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def show_all(in_path,out_path,labels=None):
    if(not os.path.isdir(out_path)):
        os.mkdir(out_path)
    if(not os.path.isdir(in_path)):
        title=in_path.split("/")[-1]
        show_confusion(in_path,title=title,out_path=out_path,
            colors="grey")
    all_paths=os.listdir(in_path)
    for i,path_i in enumerate(all_paths):
        in_i="%s/%s"  % (in_path,path_i)
        out_i="%s/%s" % (out_path,path_i)
        print(out_i)
        labels_i=labels[i] if(labels) else None
        show_confusion(in_path=in_i,title=path_i,out_path=out_i,
            labels=labels_i,colors="grey")

def show_confusion(in_path,labels=None,title=None,out_path=None,
                    colors="base"):       	
    cmap_dict={"grey":'Greys','base':"YlGnBu"}
    plt.clf()
    cf_matrix=np.genfromtxt(in_path,delimiter=',')
    dim=cf_matrix.shape
    if(not labels):
        labels=range(dim[0])
    df_cm = pd.DataFrame(cf_matrix, labels,labels)
    sn.set(font_scale=0.8)

    fig, ax = plt.subplots(figsize=(6,6))
    
    sn.heatmap(df_cm,cmap=cmap_dict[colors],#linewidths=0.5,
    	annot=True,annot_kws={"size": 6}, fmt='g',#ax=ax)
        xticklabels=True, yticklabels=True)
    if(title):
        plt.title(title)
    ax.figure.subplots_adjust(left = 0.3, bottom=0.3)

    if(all( type(i) == int for i in labels)):
        plt.xlabel("predicted labels")
        plt.ylabel("true labels")
    
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90)
    if(out_path):
        plt.savefig(out_path)
    else:
        plt.show()


cats_MSR=['high arm wave','horizontal arm wave','hammer','hand catch','forward punch',
          'high throw','draw x','draw tick','draw circle','hand clap',
          'two hand wave','side-boxing','bend','forward kick','side kick',
          'jogging','tennis swing','tennis serve','golfswing','pick up & throw']

cats_MHAD=['right arm swipe to the left', 'right arm swipe to the right', 'right hand wave','two hand front clap','right arm throw',
 'cross arms in the chest','basketball shoot','right hand draw x', 'draw circle (clockwise)','draw circle (counter clockwise)', 
 'draw triangle','bowling', 'front boxing', 'baseball swing from right', 'tennis right hand forehand swing','arm curl', 
 'tennis serve', 'two hand push', 'right hand knock on door', 'right hand catch an object', 'right hand pick up and throw', 'jogging in place', 
 'walking in place', 'sit to stand', 'stand to sit','forward lunge','squat']

cats_MHAD=['arm swipe to the left','arm swipe to the right','right hand wave','two hand front clap','right arm throw','cross arms in the chest','basketball shoot',
 'draw x','circle (clockwise)','circle (counter clock)','draw triangle','bowling', 'front boxing','baseball swing','tennis forehand swing',
 'arm curl','tennis serve','two hand push','knock on door','catch an object','pick up and throw','jogging in place','walking in place', 
 'sit to stand', 'stand to sit','forward lunge','squat']

cats_3DHOI=['drinking','pouring','calling phone','playing phone',
'wearing backpacks','packing backpacks','sitting chair','moving chair',
'taking out wallet','taking from wallet','mopping','sweeping']

in_path="../../ICSS/wyniki/confusion_matrix/raw/"
#show_confusion("cf/pairs",None,"pairs")
show_all(in_path,"plots",labels=[cats_MHAD,cats_3DHOI])
