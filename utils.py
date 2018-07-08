import os,re,pickle

def bottom_files(path):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            paths=[ root+'/'+filename_i 
                for filename_i in filenames]
            all_paths+=paths
    return all_paths

def bottom_dirs(path):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            all_paths.append(root)
    print(all_paths)
    return all_paths

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def save_object(nn,path):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def read_object(path):
    file_object = open(path,'rb')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj

def save_string(path,string):
    file_str = open(str(path),'w')
    file_str.write(string)
    file_str.close()