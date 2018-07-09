import seqs.io,utils,deep.reader 

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
        name_i=path_i.split('/')[-1]
        out_i=out_path+'/'+name_i
        print(out_i)
        save_actions(feat_actions_i,out_i)