from scipy import ndimage
from scipy import misc
import numpy as np

def pics2array(path, pic_names, im_size=(128, 128)):
    X = list()
    
    for pic_name in pic_names:
        pic = ndimage.imread(path + pic_name)
        pic = misc.imresize(pic, (im_size))
        pic = pic.transpose(2, 0, 1)
        
        X.append(pic)
        
    return X