#from paths import *
import numpy as np
import cv2
import os
from sklearn.preprocessing import normalize
from IPython import embed
from skimage.measure import block_reduce
import matplotlib.pylab as plt
def get_distanceTransform(mask, k):
    img = (255*mask).astype(np.uint8)
    dist = cv2.distanceTransform(255-img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
    #mask_weights = np.exp(-0.05*dist)
    if np.max(dist)==0:
        return None
    dist = 1 - dist/np.max(dist)
    return dist

def apply_aggregation(X, S, C, pool='sum'):
    """
    Given a tensor of activations, compute the aggregate CroW feature, weighted
    spatially and channel-wise.
    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        CroW aggregated global image feature
    """
    X = X * S
    if pool=='sum':
        X = X.sum(axis=(1, 2))
    else:
        X = X.max(axis=(1,2))
    return X * C

def compute_crow_channel_weight(X):
    """
    Given a tensor of features, compute channel weights as the
    log of inverse channel sparsity.
    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        a channel weight vector
    """
    K, w, h = X.shape
    area = float(w * h)
    nonzeros = np.zeros(K, dtype=np.float32)
    for i, x in enumerate(X):
        nonzeros[i] = np.count_nonzero(x) / area

    nzsum = nonzeros.sum()
    for i, d in enumerate(nonzeros):
        nonzeros[i] = np.log(nzsum / d) if d > 0. else 0.

    return nonzeros

def compute_crow_spatial_weight(X, a=2, b=2):
    """
    Given a tensor of features, compute spatial weights as normalized total activation.
    Normalization parameters default to values determined experimentally to be most effective.
    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :param int a:
        the p-norm
    :param int b:
        power normalization
    :returns ndarray:
        a spatial weight matrix of size (height, width)
    """
    S = X.sum(axis=0)
    z = (S**a).sum()**(1./a)
    return (S / z)**(1./b) if b != 1 else (S / z)

def saliency_mask( path_saliency, name, size, ext='.png' ):
    """
    Get saliency for a particular image.
    """
    mask = cv2.imread( os.path.join(path_saliency, name+ext) )
    mask = cv2.resize( mask, (size[1], size[0]) )
    mask = mask[:,:,0].astype(np.float32)
    if mask.max() == 0:
        mask = 0.5
    else:
        return mask / mask.max()

def saliency_mask_blocks( path_saliency, name, size, ext='.png' ):
    """
    Get saliency for a particular image.
    """
    mask = cv2.imread( os.path.join(path_saliency, name+ext) )
    # reduce to size of image processing for feats
    mask = cv2.resize( mask, (size[1]*16, size[0]*16) )
    mask = block_reduce( mask[:,:,0], (16,16), np.max )
    mask = mask.astype(np.float32)
    return mask / mask.max()

def cams_mask( path_cams, name ):
    """
    Get cams for a particualr index

    cams are stored in numpy format.
    """
    mask = np.load( os.path.join(path_cams, name+'.npy') )
    mask = mask.astype(np.float32)
    mask -= mask.min()
    return mask / mask.max()

def l2_norm_maps( feat, path_feat=None, dim_r=None ):
    if path_feat is not None:
        feat = np.load( path_feat ).squeeze(axis=0)

    norm_m = np.sqrt(np.sum( feat**2, axis=0 ))
    norm_m /= norm_m.max()
    if dim_r is not None:
        norm_m = cv2.resize( norm_m, (dim_r[1], dim_r[0] ))
    return norm_m

def get_spatial_weights( feat ):
    S = np.sum( feat, axis=0 )
    S_norm = np.sqrt(np.sum( S**2 ))
    return np.sqrt(S/S_norm)

def gaussian_weights(shape, center = None, sigma=None ):
    r1 = shape[0] /2
    r2 = shape[1] /2

    ys = np.linspace(-r1, r1, shape[0])
    xs = np.linspace(-r2, r2, shape[1])
    YS, XS = np.meshgrid(xs, ys)

    if center is not None:
        YS -= ( center[1]-r1 )
        XS -= (center[0]-r2 )

    if sigma is None:
        sigma = min(shape[0], shape[1]) / 3.0
    g = np.exp(-0.5 * (XS**2 + YS**2) / (sigma**2))

    # normalize
    g -= np.min(g)
    g /= np.max(g)
    return g

def weighted_distances( dx=10, dy=10, c=None):
    '''
    Map with weighted distances to a point
	args: Dimension maps and point
    '''

    if c is None:
        c = (dx/2, dy/2)

    a = np.zeros((dx,dy))
    a[c]=1

    indr = np.indices(a.shape)[0,:]
    indc = np.indices(a.shape)[1,:]

    difr = indr-c[0]
    difc = indc-c[1]

    map_diff = np.sqrt((difr**2)+(difc**2))

    map_diff = 1.0 - (map_diff/ map_diff.flatten().max())

    # Return inverse distance map
    return map_diff


# R-MAC regions
def get_rmac_region_coordinates( H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)


########################### R-MAC ###################################################################
def get_rmac( feats, L, pca=None, pool='sum'):
    """
    L=0 normal pooling
    """
    ch, H, W = feats.shape
    regions = get_rmac_region_coordinates( H, W, L=L)

    # aggregated descriptor
    g_feat = np.zeros(ch)

    # extract regions
    N = len(regions)

    # loop on regions
    for k in range(N):
        x,y,w,h = map(int,regions[k])

        if pool == 'sum':
            r_feat = feats[:,y:y+h,x:x+w].sum(axis=(1,2))
        else:
            r_feat = feats[:,y:y+h,x:x+w].max(axis=(1,2))
        # post processing local
        r_feat = normalize(r_feat)
        if pca is not None:
            r_feat = np.expand_dims(r_feat, axis=0)
            r_feat = pca.transform(r_feat)
            r_feat = normalize(r_feat)
            r_feat = r_feat.squeeze()
        # sumpooling regions
        g_feat += r_feat

    return g_feat



def get_rmac_region_coordinates( H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions

        if L == 0:
            regions_xywh = []
            regions_xywh.append((0,0,H,W))
            return np.array(regions_xywh).astype(np.float32)
        else:
            ovr = 0.4
            # Possible regions for the long dimension
            steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
            w = np.minimum(H, W)

            b = (np.maximum(H, W) - w) / (steps - 1)
            # steps(idx) regions for long dimension. The +1 comes from Matlab
            # 1-indexing...
            idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

            # Region overplus per dimension
            Wd = 0
            Hd = 0
            if H < W:
                Wd = idx
            elif H > W:
                Hd = idx

            regions_xywh = []
            for l in range(1, L+1):
                wl = np.floor(2 * w / (l + 1))
                wl2 = np.floor(wl / 2 - 1)
                # Center coordinates
                if l + Wd - 1 > 0:
                    b = (W - wl) / (l + Wd - 1)
                else:
                    b = 0
                cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
                # Center coordinates
                if l + Hd - 1 > 0:
                    b = (H - wl) / (l + Hd - 1)
                else:
                    b = 0
                cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

                for i_ in cenH:
                    for j_ in cenW:
                        regions_xywh.append([j_, i_, wl, wl])

            # Round the regions. Careful with the borders!
            for i in range(len(regions_xywh)):
                for j in range(4):
                    regions_xywh[i][j] = int(round(regions_xywh[i][j]))
                if regions_xywh[i][0] + regions_xywh[i][2] > W:
                    regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
                if regions_xywh[i][1] + regions_xywh[i][3] > H:
                    regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
            return np.array(regions_xywh).astype(np.float32)


def get_rmac_spatial_weights(  H, W, L ):
    regions = get_rmac_region_coordinates( H, W, L )
    mask = np.zeros((H,W))
    for i in range(len(regions)):
        x,y,w,h = map(int, [k for k in regions[i]])
        mask[y:y+h,x:x+w]+=1

    mask /=np.max(mask)
    return mask

def get_rmac_spatial_weights_l2norms( feat, L ):
    ch, H, W = feat.shape
    regions = get_rmac_region_coordinates( H, W, L )
    mask = np.zeros((H,W)).astype(float32)
    for i in range(len(regions)):
        x,y,w,h = map(int, [k for k in regions[i]])

          # pool region descriptors
        f_r = np.sum( feat[:,y:y+h,x:x+w], axis=-1 ).sum(axis=-1)
        f_r_norm = np.sqrt( np.sum(f_r**2) )
        mask[y:y+h,x:x+w] += 1.0/f_r_norm
        # add to the global descriptor
    #mask /=np.max(mask)
    return mask


def get_weights( dataset, i, assignments, spatial_w='gaussian', network='vgg16' ):
    " select the kind of weight to apply to the assignments "
    " dim --  dim assignment map "
    x,y = assignments.shape
    path_feat = os.path.join( APATHS[dataset][network]['features_keyframes'], "{}.npy".format(i) )
    if i ==0:
        print spatial_w
    if spatial_w == "gaussian":
        S = gaussian_weights( (x,y) )
    elif spatial_w == "r-macw":
        S = get_rmac_spatial_weights(  x, y, L=3 )
    elif spatial_w == "distance":
        S = weighted_distances( dx=x, dy=y )
    elif spatial_w == "saliency":
        S = saliency_mask_blocks( APATHS[dataset]['saliency'], str(i), (x,y) )
    elif spatial_w == "saliency_mss":
        if dataset =='oxford' or dataset =='paris':
            keyframes = np.loadtxt( os.path.join( APATHS[dataset]['dataset'], 'imlist.txt'), dtype='str' )
            S = saliency_mask_blocks( APATHS[dataset]['saliency_mss'], keyframes[i].split('.')[0], (x,y), ext='.jpg' )
        else:
            S = saliency_mask_blocks( APATHS[dataset]['saliency_mss'], str(i), (x,y), ext='.jpg')
    elif spatial_w == "saliency_sam_vgg16":
        S = saliency_mask_blocks( APATHS[dataset]['saliency_sam_vgg16'], str(i), (x,y), ext='.png')
    elif spatial_w == "saliency_sam_resnet":
        S = saliency_mask_blocks( APATHS[dataset]['saliency_sam_resnet'], str(i), (x,y), ext='.png')
    elif spatial_w == 'saliency_salgan':
        S = saliency_mask_blocks( APATHS[dataset]['saliency_salgan'], str(i), (x,y), ext='.png')
    elif spatial_w == 'saliency_itty1998':
        S = saliency_mask_blocks( APATHS[dataset]['saliency_itty1998'], str(i), (x,y), ext='.png')
    elif spatial_w == "CAMS":
        S = cams_mask( APATHS[dataset]['CAMS'], str(i) )
    elif spatial_w == "l2norm":
        S = l2_norm_maps( None, path_feat=path_feat, dim_r=(x,y) )
    elif spatial_w == "crow_spatial":
        S = compute_crow_spatial_weight( feat_ )
    else:
        S = None
    return S
