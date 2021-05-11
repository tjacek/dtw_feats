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

    def get_person(self):
        return int(self.split('_')[1])

    def subname(self,k):
        subname_k="_".join(self.split("_")[:k])
        return Name(subname_k)

class SetSelector(object):
    def __init__(self,names):
        self.train=set(names)

    def __call__(self,name_i):
        return name_i in self.train

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

def save_txt(text,out_path):
    if(type(text)==list):
        text="\n".join(text)
    file1 = open(out_path,"w")   
    file1.write(text) 
    file1.close()

def read_txt(in_path):
    f = open(in_path, "r")
    return f.read()

def get_paths(prefix,names):
    return { name_i:"%s/%s" % (prefix,name_i) for name_i in names}

def by_cat(names):
    group={}
    for name_i in names:
        cat_i=name_i.get_cat()
        if(not cat_i in group):
            group[cat_i]=[]
        group[cat_i].append(name_i)
    return group

def get_paths(in_path,name="dtw"):
    return ["%s/%s" % (path_i,name) 
                for path_i in top_files(in_path)]