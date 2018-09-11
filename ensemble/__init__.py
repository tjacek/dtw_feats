import seqs.io,utils,deep.reader 
import feats

class EnsembleFun(object):
    def __init__(self,in_fun,out_fun=None,n_paths=None):
        self.in_fun=in_fun
        self.out_fun=out_fun
        self.n_paths=n_paths

    def __call__(self,in_path,out_path=None):
        model_paths=self.get_paths(in_path)
        print(model_paths)
        if(out_path):
            utils.make_dir(out_path)
        all_result=[]
        for path_i in model_paths:
            print(path_i)
            result_i=self.in_fun(path_i)
            all_result.append(result_i)
            if(self.out_fun):
                out_path_i=get_out_path(path_i,out_path)
                self.out_fun(out_path_i,result_i)
        return all_result

    def get_paths(self,in_path):
        if(type(self.n_paths)==int):
            return [ in_path+'/nn'+str(i) for i in range(self.n_paths)]
        if(self.n_paths=='dirs'):
            return utils.bottom_dirs(in_path)
        return utils.bottom_files(in_path)

    def single_call(self,in_path,out_path=None):
        result=self.in_fun(in_path)
        if(self.out_fun):
            self.out_fun(out_path,result)

def global_feats(global_features):
    def in_fun(in_path_i): 
        return in_path_i
    def out_fun(out_i,in_i):
        global_features.apply(in_i,out_i)
    return EnsembleFun(in_fun,out_fun,'dirs')

def get_out_path(in_path,dir_path):
    name= in_path.split('/')[-1]
    return dir_path+'/'+name