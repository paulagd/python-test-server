import os, glob, sys
from keras.applications import vgg16
from IPython import embed
from tqdm import tqdm
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import cv2

def find_dimensions( ima, max_dim=340, r_ratio=False):
    """
    Find the dimensions for keeping the aspect ratio.
    It sets the larger dimension to 'max_dim'

    return tuple with the new dimensions
    """
    if len(ima.shape)==3:
        dim =np.array(map(float, ima.shape[:2]))
    else:
        dim =np.array(map(float, ima.shape))
    ratio = min( max_dim/dim )
    new_dim = tuple([int(ratio*d) for d in dim])
    if r_ratio:
        return new_dim, ratio
    else:
        return new_dim

def preprocess_image( path_image, dim=None, max_dim=None ):
    """
    Addapts image to size 'dim' outputs tensor for net
    args: path_image
    retu: X tensor
    """
    if 'str' in str(type(path_image)) or  'unicode' in str(type(path_image)):
        ima = cv2.imread( path_image )
    else:
        ima = path_image

    # check if it is the mask
    if len(ima.shape) == 2:
        if dim is None:
            dim = find_dimensions(ima, max_dim)
        ima = cv2.resize( ima, (dim[1], dim[0]) )
        ima[ima>0] = 1
        return ima
    else:
        if dim is None and max_dim is None:
            print "keeping crop"
            print ima.shape

            ima = ima[:,:,::-1].astype( dtype=np.float32 )
            ima = np.transpose( ima, (2,0,1) )
            ima = np.expand_dims( ima, axis=0 )
            return preprocess_input(ima)

        if dim is None:
            dim = find_dimensions(ima, max_dim)

        ima = cv2.resize( ima, (dim[1], dim[0]) )
        ima = ima[:,:,::-1].astype( dtype=np.float32 )
        ima = np.transpose( ima, (2,0,1) )
        ima = np.expand_dims( ima, axis=0 )
        return preprocess_input(ima)

##########################################################################
# Extract images for X dataset.
#
# Sets the larges dimension to 340. :)
###########################################################################

def init_model( layer='conv5_1' ):
    #init model
    model = vgg16.VGG16(include_top=False, weights='imagenet')

    if layer == 'pool1':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []


    if layer == 'pool2':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()


        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

    if layer == 'pool3':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []


    if layer == 'pool4':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

    if layer == 'conv5_1':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
    elif layer == 'conv5_2':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
    elif layer == 'conv5_3':
        # remove layers
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

    return model


class VGG_extractor():
    def __init__(self, layer='conv5_1', max_dim=340):
        self.model =  init_model( layer )
        self.max_dim = max_dim

    def extract_features( self,path ):
        x = preprocess_image( path, max_dim=self.max_dim )
        feats = self.model.predict(x)
        return feats.squeeze(axis=0)

def compute_features( keyframes, layer, max_dim, path_features ):
    """
    main loop to extract features.
    Uses VGG16conv resizing images to max_dim
    Stores numpy arrays in path_features folder
    """
    model = init_model( layer )

    i = 0
    for path in tqdm(keyframes):
        x = preprocess_image( path, max_dim=max_dim )
        feats = model.predict(x)
        np.save( "{}/{}".format(path_features,i), feats )
        i+=1


PATH_DATASET={
    'oxford':'/media/eva/Eva Data/Datasets/Oxford_Buildings',
    'paris':'/media/eva/Eva Data/Datasets/Paris_dataset',
    'instre':'/media/eva/Eva Data/Datasets/Instre',
    'trecvid':"/media/eva/TRECVID/ins16"
}
