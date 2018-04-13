from config import *
from dataset import *
from utils import *
from IPython import embed
from vgg import VGG_extractor
from codebook import *
from dataset import load_dataset
import urllib
import calendar;
import time;
from config import *
#from util_decoder import decode_image_data_url

def get_targets( params ):
    #ds = load_dataset( dataset=params[dataset], path_to_load = PATH_LOAD_DATASET )
    id_file = "{}/{}/{}".format(params['dataset'],
                                                params['network'],
                                                params['weighting'])
    file_out = os.path.join( INV_FILE, id_file)
    return load_sparse_csr( file_out+'/targets')

def get_queries( params ):
    #ds = load_dataset( dataset=params[dataset], path_to_load = PATH_LOAD_DATASET )
    id_file = "{}/{}/{}".format(params['dataset'],
                                                params['network'],
                                                params['weighting'])
    file_out = os.path.join( FIXED_QUERIES, params['dataset'], params['mode_query'])
    return load_sparse_csr( file_out+'/queries')


def map_file_query( id_name, ds, bow_queries ):
    q_name_files = np.array( [ os.path.basename(k).split('.')[0] for k in ds.q_keyframes ] )
    res = np.where( q_name_files==id_name )[0]
    if res.size > 0:
        return bow_queries[res[0],:]
    else:
        return None


def store_plot( id_ima, ranks, ds ):
    """
    create json rank for image id
    """
    #check sub
    sub = os.path.dirname(id_ima)
    create_dir(sub)
    create_dir(id_ima)
    for i,r in enumerate(ranks[:50]):
        ima = cv2.imread(ds.keyframes[r])
        cv2.imwrite( '{}/{}.jpg'.format(id_ima, i),ima)


def store_json_rank( path_ranks, id_ima, ranks, keyframes, top_n ):
    """
    create json rank for image id
    """
    path_ranks = os.path.join(path_ranks, id_ima)
    sub = os.path.dirname(path_ranks)
    name = os.path.basename(path_ranks).split(".")[0]
    create_dir(sub)
    json_file =  os.path.join(  sub,"{}.json".format(name) )
    #check if subfolder exists
    with open( json_file, 'wb' ) as f:
        ranks = ranks[:top_n]
        N = ranks.shape[0]
        for i, r in enumerate(ranks):
            line=''
            name = keyframes[r].split('.')[0]
            if i == 0:
                line+="[{\n\"IdSequence\" :\"%i\",\n" % ( i)
            else:
                line+="{\n\"IdSequence\" :\"%i\",\n" % ( i)
            line+="\"Image\":\"%s\"\n" % name
            if i == N-1:
                line+="}]"
            else:
                line+="},\n"
            # write
            f.write(line)
    return json_file

class Extractor():
    def __init__(self, layer=None, max_dim=None, dataset=None, cov_path=None):
        self.vgg = VGG_extractor(layer=params['layer'], max_dim=params['max_dim'])
        self.pca_model, centroids = get_models( params['dataset'], params['max_dim'],
                                           params['layer'],
                                           model_paths=CODEBOOK_FILES)
        self.codebook = GPUCodebook( centroids )

    def get_representation( self, ima_path ):
        """
        Get BoW representation for the image path
        """
        feats = self.vgg.extract_features( ima_path )
        feats = postprocess(feats, interpolate=2, apply_pca=self.pca_model)
        assignments = self.codebook.get_assignments(feats)
        bow_image = get_bow( assignments )
        return bow_image


class BLCF():
    def __init__(self, params):
        #store ranks
        create_dir( 'ranks' )
        self.path_ranks = os.path.join( 'ranks','dataset_{}_weighting_{}'.format( params['dataset'], params['weighting'] ))
        create_dir( self.path_ranks )
        self.dataset = params['dataset']
        self.ds = load_dataset(params['dataset'], path_to_load = PATH_LOAD_DATASET)
        self.top_n = params['top_n']

        if self.dataset == 'instre':
            self.id_queries = np.loadtxt( '{}/{}/qimlist.txt'.format(PATH_IMLIST, self.dataset), dtype='str', delimiter='\n' )
            self.id_keyframes = np.loadtxt( 'search_engine/imlists/{}/imlist.txt'.format(self.dataset), dtype='str', delimiter='\n' )
        else:
            self.id_queries = np.loadtxt( '{}/{}/qimlist.txt'.format(PATH_IMLIST, self.dataset), dtype='str' )[:,0]
            self.id_keyframes = np.array( [os.path.basename(k) for k in self.ds.keyframes] )

        # default model
        self.model = Extractor( params['layer'], params['max_dim'], params['dataset'], CODEBOOK_FILES )
        # fordward one image to warm the network
        #bow_query = self.model.get_representation( '/home/paula/Desktop/python-server/temp.jpg' )
        # select inverted file
        self.bow_targets = get_targets( params )
        self.bow_queries = get_queries( params )
        # init extractor

    def get_rank_image( self, id_ima, url, encoded_image ):
        """
        Get rank for an image id

        args:
        id_ima: name of jpg image file
        url: can be url to jpg or jpg abs path in server
        encoded_image: base64 encoded image
        [TODO] -- modify for url case!
        """
        # for image from outside the dataset

        if id_ima == 'unknown_id':
            print "case {}".format( id_ima )
            # check we have encoded url
            if encoded_image is not None and url is None:
                print "case 1 {} {}".format( encoded_image, url )
                # remove begining
                encoded_image = encoded_image.split(',')[1]
                # decode to ASCII
                str_data = decode_base64(encoded_image)
                # store temporal image
                with open( 'temp.jpg', 'wb') as f:
                    f.write(str_data)
                # read image in cv2 format
                ima_path = cv2.imread('temp.jpg')
                #ima_path = decode_image_data_url(encoded_image)
            # process url - in the case is an image within server
            elif os.path.exists(url):
                print "case 2  {} {}".format( encoded_image, url )
                ima_path = cv2.imread(url)
            # in the case it is an actual url download image and process
            else:
                # get url
                resp = urllib.urlopen(url)
                ima_path = np.asarray(bytearray(resp.read()), dtype="uint8")
                ima_path = cv2.imdecode(ima_path, cv2.IMREAD_COLOR)
                print "Shape image decoded from url {}".format( ima_path.shape )

            bow_query = self.model.get_representation( ima_path )

        else:
            print "case X  {} {}".format( encoded_image, url )
            embed()
            #process image from dataset
            id_ima = str(id_ima.split('.')[0])
            # check if json has been computed
            json_file = os.path.join(self.path_ranks, id_ima+'.json')

            if os.path.exists( json_file ):
                print "Query rank computed! {}".format( json_file )
                return json_file
            else:
                print "Computing", id_ima
                # load computed query
                if self.dataset == 'instre':
                    idx = np.where( id_ima+'.jpg'==self.id_queries )[0][0]
                    bow_query = self.bow_queries[idx]
                else:
                    bow_query = map_file_query( id_ima, self.ds, self.bow_queries )

                if bow_query is None:
                    # PROCESS IMAGE
                    print "Computing new features for {}".format( id_ima )
                    ima_path = os.path.join( PATH_IMAGES[self.dataset], id_ima+'.jpg' )
                    # if not computed...
                    bow_query = self.model.get_representation( ima_path )

        # compute ranks for query
        ranks, distances = compute_ranks( self.bow_targets, bow_query )
        #store_plot( id_ima, ranks, self.ds )
        # store txt
        json_file = store_json_rank( self.path_ranks, id_ima, ranks, self.id_keyframes, self.top_n)

        return json_file
