import numpy as np
import os
from IPython import embed
import sys
import cv2
from tqdm import tqdm
import pickle
import matplotlib.pylab as plt
import cv2
sys.path.insert( 0, '/home/eva/2017/Experiments_weightening' )



def get_dim( path ):
    dim = cv2.imread( path ).shape[:2]
    return dim


def compute_ranks( targets, queries ):
    distances = targets.dot( queries.T )
    if 'sparse' in str(type(distances)):
        distances = np.array(distances.toarray())
    distances = distances.squeeze()

    N = queries.shape[0]

    if N == 1:
        rank = np.argsort( distances )[::-1]
        return rank, distances

    else:
        ranks = []
        for i in range(N):
            rank = np.argsort( distances[:,i] )[::-1]
            ranks.append(rank)
        return np.array(ranks).T, distances

class Dataset( ):
    def __init__(self, dataset ='oxford'):
        self.dataset = dataset
        self.set_data( dataset )
        if self.dataset == 'trecvid':
            ima_dim = get_dim(self.keyframes[0])
            self.keyframes_dim = np.ones( (self.keyframes.shape[0],2) )
            self.q_keyframes_dim = np.ones( (self.q_keyframes.shape[0],2) )

            self.keyframes_dim[:,0] = np.ones( self.keyframes.shape[0] )*ima_dim[0]
            self.keyframes_dim[:,1] = np.ones( self.keyframes.shape[0] )*ima_dim[1]

            self.q_keyframes_dim[:,0] = np.ones( self.q_keyframes.shape[0] )*ima_dim[0]
            self.q_keyframes_dim[:,1] = np.ones( self.q_keyframes.shape[0] )*ima_dim[1]
        else:
            self.get_dimension_images()

    def get_dimension_images(self):
        self.keyframes_dim = []
        self.q_keyframes_dim = []
        for path in tqdm(self.keyframes):
            self.keyframes_dim.append( get_dim(path) )

        for path in tqdm(self.q_keyframes):
            self.q_keyframes_dim.append( get_dim(path) )

    def set_data( self, dataset ):

        self.path_dataset = os.path.join( PATH_IMLIST, dataset )

        # set path query images
        if dataset == 'oxford' or dataset == 'paris':
            self.keyframes = np.loadtxt( os.path.join( self.path_dataset, 'imlist.txt' ), dtype='str' )
            # get keyframes
            self.q_keyframes = np.loadtxt( os.path.join( self.path_dataset, 'qimlist.txt' ), dtype='str' )[:,0]
            self.q_topics = np.loadtxt( os.path.join( self.path_dataset, 'qimlist.txt' ), dtype='str' )[:,1]

            # get bbx
            self.bbx =  np.loadtxt( os.path.join( self.path_dataset, 'qimlist.txt' ), dtype='str' )[:,2:].astype(float)
            # format oxford is x, y, x+w, y+h
            self.bbx[:,2] = self.bbx[:,2]-self.bbx[:,0]
            self.bbx[:,3] = self.bbx[:,3]-self.bbx[:,1]
            # full path
            self.keyframes = np.array( [os.path.join( self.path_dataset, 'images',k ) for k in self.keyframes] )
            self.q_keyframes = np.array( [os.path.join( self.path_dataset, 'images',k ) for k in self.q_keyframes] )


        elif dataset == 'instre':
            from instre.evaluation import get_gnd
            gnd = get_gnd()
            self.q_path_images = self.path_dataset
            # get keyframes
            self.q_keyframes = np.loadtxt( os.path.join( self.path_dataset, 'qimlist.txt' ), dtype='str' )
            self.keyframes = np.loadtxt( os.path.join( self.path_dataset, 'imlist.txt' ), dtype='str' )

            bbx = []
            for i in range(len(gnd)):
                bbx.append(gnd[i].bbx)
            self.bbx = np.array(bbx)
            self.bbx[:,2] = self.bbx[:,2]-self.bbx[:,0]
            self.bbx[:,3] = self.bbx[:,3]-self.bbx[:,1]
            # full path
            self.keyframes = np.array( [os.path.join( self.path_dataset, k ) for k in self.keyframes] )
            self.q_keyframes = np.array( [os.path.join( self.path_dataset, k ) for k in self.q_keyframes] )


    def get_mask_frame( self, q, ass_dim=None ):
        """ Get mask for a given query and resize to the feature map dimension
        desired """
        # frame ID without path and extension
        dim_ima = map(int, self.q_keyframes_dim[q])

        if ass_dim == None:
            ass_dim = dim_ima
        mask = np.zeros( dim_ima )

        dim_ima = map(float, self.q_keyframes_dim[q])

        # get bbx coordinates
        x,y,w,h = self.bbx[q,:].astype(int)
        if w == 0:
            w=1
        if h == 0:
            h=1

        mask[ y:y+h, x:x+w ]=1
        # init output mask

        #first resize
        mask = cv2.resize( mask, (ass_dim[1],ass_dim[0]) )
        mask[mask>=0.5]=1
        mask[mask<0.5]=0


        #print "debug mask ", mask.shape
        return mask


    def evaluate_ranks(self,ranks):
        if self.dataset == 'instre':
            from instre.evaluation import compute_map as instre_map
            mAP = instre_map(ranks)
            print np.mean(mAP)
        else:
            from oxford.evaluation import evaluate_old as oxf_par_map
            mAP = oxf_par_map(ranks, self.path_dataset).values()
        return mAP

    def evaluate( self, targets, queries ):
        mAP = self.evaluate_ranks( compute_ranks( targets, queries )[0] )
        return mAP

    def get_keyframes(self):
        return self.keyframes

    def get_qkeyframes(self):
        return self.q_keyframes




def load_dataset( dataset='oxford', path_to_load = None ):
    "  Load Dataset instance  "
    if path_to_load == None:
        with open("{}.pkl".format(dataset), "r") as fid:
            ds2 = pickle.load(  fid )
    else:
        with open(os.path.join(path_to_load,"{}.pkl".format(dataset)), "r") as fid:
            ds2 = pickle.load(  fid )

    return ds2


def create_bow(ima, n_clusters=25000):
    assignments = np.load( ima )

    # sparse encoding !
    rows = np.array([], dtype=np.int)
    cols = np.array([], dtype=np.int)
    vals = np.array([], dtype=np.float)
    n_docs = 0


    # get counts
    cnt = Counter(assignments)
    ids = np.array(cnt.keys())
    weights = np.array(cnt.values())

    #save index
    cols = np.append( cols, np.array(ids).astype(int) )
    rows = np.append( rows, np.ones( len(cnt.keys()), dtype=int )*n_docs )
    vals = np.append( vals, weights.astype(float) )


    bows = coo_matrix( ( vals, (rows, cols) ), shape=(n_docs,self.centroids.shape[0]) )
    bows = features.tocsr()
    bows = normalize(bows)

    return bows
