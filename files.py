import os,re,itertools

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def clean(self):
        digits=[ str(int(digit_i)) 
                for digit_i in re.findall(r'\d+',self)]
        return Name("_".join(digits))

    def get_cat(self):
        return int(self.split('_')[0])-1

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def all_files(path):
    dicts=[]
    for root, directories, filenames in os.walk(path):
        dicts+=top_files(root)
    return dicts

def all_dicts(path):
    dicts=[]
    for root, directories, filenames in os.walk(path):
        dicts.append(root)
    return dicts

def natural_sort(l):
    return sorted(l,key=natural_keys)

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def iter_product(args):
    return list(itertools.product(*args))

def flatten(args):
    return list(itertools.chain(*args))

def split(dict,selector=None):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in dict.keys():
        if(selector(name_i)):
            train.append((name_i,dict[name_i]))
        else:
            test.append((name_i,dict[name_i]))
    return train,test

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1

def to_csv(lines,out_path):
    csv="\n".join(lines)
    file1 = open(out_path,"w")   
    file1.write(csv) 
    file1.close()