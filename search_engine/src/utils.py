from weights import get_weights, get_distanceTransform, gaussian_weights

import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse import hstack, vstack
from sklearn.preprocessing import normalize
from tqdm import tqdm
import os
import base64

from IPython import embed

from scipy.sparse import csr_matrix
from scipy.ndimage import zoom

def create_dir( path ):
    if not os.path.exists(path) and path!='':
        os.makedirs(path)

def decode_base64(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += b'='* (4 - missing_padding)
    return base64.decodestring(data)


def postprocess(feat, get_dim=False, interpolate=1, apply_pca=None):
    if interpolate > 1:
        z= interpolate
        feat = zoom(feat, (1,z,z), order=1)

    dim = feat.shape
    feat = np.reshape( feat, (feat.shape[0], -1) )
    feat = np.transpose( feat, (1,0) )
    feat = normalize(feat)
    if apply_pca is not None:
        feat = apply_pca.transform(feat)
        feat = normalize(feat)
    if get_dim:
        return feat.astype(np.float32), dim
    else:
        return feat.astype(np.float32)



def query_expansion( targets, queries, N=10 ):
    # get scores
    scores = targets.dot( queries.T )
    scores = scores.toarray().squeeze()
    NQ = queries.shape[0]

    new_queries = None
    for i in range(NQ):
        # get rankings
        if len( scores.shape )>1:
            idx = np.argsort( scores[:,i] )[::-1]
        else:
            idx = np.argsort( scores )[::-1]


        # get 10 features
        new = None
        for k in range(N+1):
            if new == None:
                new = queries[i,:].toarray()
            else:
                new = np.vstack( [new, targets[idx[k-1],:].toarray()] )
        new = new.mean(axis=0)

        if new_queries == None:
            new_queries = new
        else:
            new_queries = np.vstack( [new_queries, new] )

    new_queries = normalize( new_queries )
    return new_queries


def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])



def find_dimensions( ima_dims, max_dim=340, r_ratio=False):
    """
    Find the dimensions for keeping the aspect ratio.
    It sets the larger dimension to 'max_dim'

    return tuple with the new dimensions
    """
    ima_dims_max = max( np.array(ima_dims) )
    ratio = float(max_dim)/ima_dims_max
    new_dim = tuple([int(ratio*d) for d in ima_dims])

    if r_ratio:
        return new_dim, ratio
    else:
        return new_dim


def get_bow( assignments, weights=None, n=25000 ):
    # sparse encoding !
    rows = np.array([], dtype=np.int)
    cols = np.array([], dtype=np.int)
    vals = np.array([], dtype=np.float)
    n_docs = 0

    # get counts
    cnt = Counter(assignments.flatten())
    ids = np.array(cnt.keys())
    if weights == None:
        weights = np.array(cnt.values())
    else:
        weights = weights.flatten()
        weights = np.array([weights[np.where(assignments.flatten()==i)[0]].sum() for i in ids])

    #save index
    cols = np.append( cols, np.array(ids).astype(int) )
    rows = np.append( rows, np.ones( len(cnt.keys()), dtype=int )*n_docs )
    vals = np.append( vals, weights.astype(float) )
    n_docs +=1

    bow = coo_matrix( ( vals, (rows, cols) ), shape=(n_docs,n) )
    bow = bow.tocsr()
    bow = normalize(bow)
    return bow


def load_queries( ds, ass_queries, max_dim, N, mode='global'):
    for i in tqdm(range(N)):
        path = os.path.join( ass_queries, "{}.npy".format(i) )
        assignments = np.load(path)
        s_a =  assignments.shape

        if  mode=='crop': # only trec has computed ass!
            weights = ds.get_mask_frame( i, ass_dim=assignments.shape )
        elif mode=='gaussian_crop':
            mask = ds.get_mask_frame( i, ass_dim=assignments.shape )
            weights = gaussian_weights(s_a, center = None, sigma=None )
            weights[mask>0]=1
        elif mode=='d_weighting':
            weights = ds.get_mask_frame( i, ass_dim=assignments.shape )
            weights=get_distanceTransform(weights, i)
        else:
            weights=None
        # if weights is not None and i==0:
        #     import matplotlib.pylab as plt
        #     print assignments.shape, weights.shape
        #     plt.imshow(weights)
        #     plt.show()
        bow = get_bow( assignments, weights=weights)
        if i == 0:
            total_bow = bow
        else:
            total_bow = vstack( [total_bow, bow] )
    return normalize(total_bow)


def load_targets( dataset, ass_queries, max_dim, N, mask=None, network='vgg16' ):
    for i in tqdm(range(N)):
        path = os.path.join( ass_queries, "{}.npy".format(i) )
        assignments = np.load(path)

        weights = get_weights( dataset, i, assignments, spatial_w=mask, network=network )
        bow = get_bow( assignments, weights=weights)

        if i == 0:
            total_bow = bow
        else:
            total_bow = vstack( [total_bow, bow] )

    return normalize(total_bow)
