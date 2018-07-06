class ModuloSelector(object):
    def __init__(self,n,k=2,unpack=unpack_cat):
        self.k=k
        self.n=n
        self.unpack=unpack

    def __call__(self,i):
        i=self.unpack(i)
        return (i % self.k)==self.n

def select(actions,selector):
    if(type(selector)==int):
        selector=ModuloSelector(selector)
    return [ action_i
                for action_i in action_i
                    if(selector(action_i))]

def unpack_cat(action_i):
    return action_i.cat 

def unpack_person(action_i):
    return action_i.person	