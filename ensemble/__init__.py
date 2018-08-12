import seqs.io,utils,deep.reader 
import feats

class EnsembleFun(object):
    def __init__(self,fun):
        self.fun=fun

    def __call__(self,in_path):
        model_paths=utils.bottom_files(nn_path)
        utils.make_dir(out_path)
        return [self.fun(path_i) for path_i in model_paths]

def extract_deep(data_path,nn_path,out_path):
    read_actions=seqs.io.build_action_reader(img_seq=True,as_dict=False)
    save_actions=seqs.io.ActionWriter(img_seq=False)
    actions=read_actions(data_path)
    model_paths=utils.bottom_files(nn_path)
    nn_reader=deep.reader.NNReader(4)
    utils.make_dir(out_path)
    for path_i in model_paths:
        model_i=nn_reader(path_i)
        feat_actions_i=[ action_i(model_i,whole_seq=False)
                            for action_i in actions]
        out_i=get_out_path(path_i,out_path)
        print(out_i)
        save_actions(feat_actions_i,out_i)

def global_feats(in_path,out_path):
    feat_paths=utils.bottom_dirs(in_path)
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    utils.make_dir(out_path)
    feat_extractor=feats.GlobalFeatures()
    print(feat_paths)
    for in_path_i in feat_paths:
        print(in_path_i)
        actions=read_actions(in_path_i)
        lines=[str(feat_extractor(action_i)) for action_i in actions]
        out_i=get_out_path(in_path_i,out_path)
        utils.save_string(out_i,lines)

def get_out_path(in_path,dir_path):
    name= in_path.split('/')[-1]
    return dir_path+'/'+name