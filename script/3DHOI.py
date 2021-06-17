import re,os,shutil

def format_names(in_path,out_path=None):
    paths=os.listdir(in_path)
    persons=list(set(paths))
    print(persons)

path="../../raw_3DHOI/SYSU"

format_names(path)