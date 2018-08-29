import feats,feats.local,feats.global

def action_imgs(in_path,out_path,local_feats):
    if(type(local_feats)!=feats.LocalFeatures):
        local_feats=feats.LocalFeatures(local_feats)
    read_actions=seqs.io.build_action_reader(img_seq=True,as_dict=False)
    actions=read_actions(in_path)
    new_actions=[action_i(local_feats,whole_seq=False) 
                    for action_i in actions]
    utils.make_dir(out_path)
    for action_i in new_actions:
        out_i=out_path+'/'+action_i.name+".png"
        img_i=action_i.as_array()
        cv2.imwrite(out_i,img_i)