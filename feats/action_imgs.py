import numpy as np
import utils,cv2
import seqs
from skimage.transform import hough_line,hough_ellipse

class ActionImgs(object):        
    def __call__(self,in_path,out_path):
        read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=False)
        actions=read_actions(in_path)
        utils.make_dir(out_path)
        for action_i in actions:
            out_i=out_path+'/'+action_i.name+".png"
            cv2.imwrite(out_i,self.get_img(action_i)[0])

class TimeActionImgs(ActionImgs):
    def __init__(self,local_feats):
        if(type(local_feats)!=feats.LocalFeatures):
            local_feats=feats.LocalFeatures(local_feats)
        self.local_feats=local_feats

    def get_img(self,action_i):
        dummy_action=action_i(self.local_feats,whole_seq=False) 
        return dummy_action.as_array(),

class TimelessActionImgs(ActionImgs):
    def __init__(self,size=(65,65),indexes=(0,1),blur=3,hough=True):
        self.size=size
        self.index=indexes
        self.blur= (blur,blur) if(blur) else None
        self.hough=hough

    def get_img(self,action_i):
        features=action_i.as_features()
        x,y=features[self.index[0]],features[self.index[1]]
        time_len=len(x)
        action_img=np.zeros(self.size)
        for i in range(time_len):
            x_i,y_i=int(x[i]),int(y[i]) 
            action_img[x_i][y_i]=1
        if(self.blur):
            action_img=cv2.blur(action_img,self.blur)
        if(self.hough=="ellipse"):
            return ellipse_transform(action_img),float(time_len)
        if(self.hough):
            h, theta, d = hough_line(action_img)
            return h, float(time_len)
        action_img[action_img!=0]=100
        return action_img,float(time_len)

def ellipse_transform(action_img):
    result= hough_ellipse(action_img)
    acum=result[['accumulator']]
    acum.dtype=float
    #acum=acum.view(acum.dtype.fields or acum.dtype,np.ndarray)
    print(type(acum))
    return acum