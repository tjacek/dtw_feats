import re,os,shutil
import numpy as np

def count_cats(in_path,out_path=None):
    paths=get_paths(in_path)
    paths=[get_paths(get_paths(path_i)[0]) for path_i in paths]
    cats=np.zeros((12,))
    for person_i in paths:
        for path_i in  person_i:	
            cats[get_cat(path_i)-1]+=1
    print(cats)

def get_paths(in_path):
    paths=os.listdir(in_path)
    return ["%s/%s" % (in_path,path_i)  
                for path_i in paths]

def get_cat(path_i):
    name_i=path_i.split("/")[-1]
    return int(re.findall(r'\d+', name_i)[0])


path="../../raw_3DHOI/SYSU"

count_cats(path)