import os,os.path,re,pickle

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

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
    return all_paths

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def extract_numbers(text):
    str_numb=re.findall(r'\d+',text)
    return [int(n) for n in str_numb]

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
    if(type(string)==list):
        string="\n".join(string)
    file_str = open(str(path),'w')
    file_str.write(string)
    file_str.close()