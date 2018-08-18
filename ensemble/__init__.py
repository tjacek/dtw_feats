import seqs.io,utils,deep.reader 
import feats

class EnsembleFun(object):
    def __init__(self,in_fun,out_fun=None,gen_paths=None):
        self.in_fun=in_fun
        self.out_fun=out_fun
        self.get_paths=get_paths

    def __call__(self,in_path,out_path=None):
        model_paths=self.get_paths(in_path)
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
        if(self.gen_paths):
            return [ in_path+'/nn'+str(i) for i in range(self.get_paths)]
        return utils.bottom_files(in_path)

def extract_deep(nn_path):
    read_actions=seqs.io.build_action_reader(img_seq=True,as_dict=False)
    save_actions=seqs.io.ActionWriter(img_seq=False)
    actions=read_actions(data_path)
    nn_reader=deep.reader.NNReader(4)
    def deep_helper(path_i):
        model_i=nn_reader(path_i)
        return [ action_i(model_i,whole_seq=False)
                    for action_i in actions]
    out_fun=lambda out_path_i,result_i:save_actions(result_i,out_path_i)
    return EnsembleFun(deep_helper,out_fun)

def global_feats(in_path,out_path):
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    feat_extractor=feats.GlobalFeatures()
    def in_fun(in_path_i): 
        actions=read_actions(in_path_i)
        return [str(feat_extractor(action_i)) 
                    for action_i in actions]
    out_fun=utils.save_string(out_i,lines)
    return EnsembleFun(in_fun,out_fun)

def get_out_path(in_path,dir_path):
    name= in_path.split('/')[-1]
    return dir_path+'/'+name