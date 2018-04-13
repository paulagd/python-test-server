import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
import numpy as np
import os

class GPUCodebook(object):
    """
    GPU version of the codebook. Much faster on the Titan X: around 15000
    512D codewords per sec
    """
    def __init__(self, centers):
        culinalg.init()

        self.centers = centers.astype(np.float32)
        self.centers_gpu = gpuarray.to_gpu(self.centers)
        self.center_norms = cumisc.sum(self.centers_gpu**2, axis=1)

    def get_distances_to_centers(self, data):

        # make sure the array is c order
        data = np.asarray(data, dtype=np.float32, order='C')

        # ship to gpu
        data_gpu = gpuarray.to_gpu(data)

        # alloc space on gpu for distances
        dists_shape = (data.shape[0], self.centers.shape[0])
        dists_gpu = gpuarray.zeros(dists_shape, np.float32)

        # calc data norms on gpu
        data_norms = cumisc.sum(data_gpu**2, axis=1)

        # calc distance on gpu
        cumisc.add_matvec(dists_gpu, self.center_norms, 1, dists_gpu)
        cumisc.add_matvec(dists_gpu, data_norms, 0, dists_gpu)
        culinalg.add_dot(data_gpu, self.centers_gpu,
            dists_gpu, transb='T', alpha=-2.0)
        return dists_gpu

    def get_assignments(self, data):
        dists = self.get_distances_to_centers(data)
        return cumisc.argmin(dists, 1).get().astype(np.int32)

    @property
    def dimension(self):
        return self.centers.shape[1]


def get_models( dataset, max_dim, layer, model_paths=None, apply_pca=True, custom_c=None, custom_pca=None ):

    id_cent = "centroids"
    id_pca = "pca"
    if custom_c is not None:
        id_cent = custom_c
    if custom_pca is not None:
        id_pca = custom_pca

    if model_paths is None:
        model_paths = os.getcwd()

    path_model = os.path.join( model_paths, dataset, str(max_dim), layer )
    if apply_pca:
        centroids = np.load( os.path.join(path_model, "{}.npy".format(id_cent)) )
        pca_model = np.load( os.path.join(path_model, "{}.npy".format(id_pca)) ).tolist()
    else:
        centroids = np.load( os.path.join(path_model, "{}_l2.npy".format(id_cent)) )
        pca_model=None

    return pca_model, centroids
