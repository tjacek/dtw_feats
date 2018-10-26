import seqs.io

class ValueSelector(object):
    def __init__(self,n,data_type=True):
        self.data_type = data_type
        self.n=n

    def __call__(self,action_i):
        i=get_data(action_i,self.data_type)
        return i==self.n 

class ModuloSelector(object):
    def __init__(self,n,data_type=True,k=2):
        self.k=k
        self.n=n
        self.data_type=data_type

    def __call__(self,action_i):
        i=get_data(action_i,self.data_type)
        return (i % self.k)==self.n        

def select_actions(in_path,out_path,selector=1, img_seq=True):
    read_actions=seqs.io.build_action_reader(img_seq=img_seq,as_dict=False)
    actions=read_actions(in_path)
    new_actions=select(actions,selector)
    save_actions=seqs.io.ActionWriter(img_seq=img_seq)  
    save_actions(new_actions,out_path)

def select(actions,selector):
    if(type(selector)==int):
        selector=ModuloSelector(selector,unpack=unpack_person)
    return [ action_i
                for action_i in actions
                    if(selector(action_i))]

def get_data(action_i,data_type=True):
    return action_i.cat if(data_type) else action_i.person 