import urllib
import calendar;
import time;
from config import *
from dataset import *
from utils import *
from IPython import embed
from vgg import VGG_extractor
from CPUcodebook import *
from dataset import load_dataset
from config import *
from scipy.sparse import vstack

from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from search_engine.datasets.instre_utils import compute_map as eval_instre
from search_engine.datasets.oxford_utils import compute_map as eval_OXF_PAR

# for evaluation...
#from search_engine.datasets.datasets import compute_ranks
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


def map_file_query( id_name, ds, bow_queries, return_id=False ):
    q_name_files = np.array( [ os.path.basename(k).split('.')[0] for k in ds.q_keyframes ] )
    res = np.where( q_name_files==id_name )[0]
    if res.size > 0:
        if not return_id:
            return bow_queries[res[0],:]
        else:
            return bow_queries[res[0],:], res[0]
    else:
        return None


def map_file_OXFPAR( id_name, ds, bow_targets, return_id=False ):
    q_name_files = np.array( [ os.path.basename(k).split('.')[0] for k in ds.keyframes ] )

    #make sure there are not whitespaces
    id_name = id_name.replace(' ', '')
    res = np.where( q_name_files==str(id_name) )[0]

    if res.size > 0:
        if not return_id:
            return bow_targets[res[0],:]
        else:
            return bow_targets[res[0],:], res[0]
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


def store_json_rank( path_ranks, id_ima, ranks, keyframes, top_n, dataset):
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

            # IDEA: modified by paula
            if dataset == "instre":
                name = name.replace("/", "__")

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
        self.codebook = CPUCodebook( centroids )

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
            self.id_queries_topics = np.loadtxt( '{}/{}/qimlist.txt'.format(PATH_IMLIST, self.dataset), dtype='str' )[:,1]

            self.id_keyframes = np.array( [os.path.basename(k) for k in self.ds.keyframes] )

        # default model
        self.model = Extractor( params['layer'], params['max_dim'], params['dataset'], CODEBOOK_FILES )
        # fordward one image to warm the network
        #bow_query = self.model.get_representation( '/home/paula/Desktop/python-server/temp.jpg' )
        # select inverted file
        self.bow_targets = get_targets( params )
        self.bow_queries = get_queries( params )

        # initial evaluation...
        self.ranks_original, _ = compute_ranks( self.bow_targets, self.bow_queries )
        self.ranks = self.ranks_original

        if self.dataset == 'instre':
            self.aps = eval_instre( self.ranks_original )
            print "Original mAP={}".format(np.mean(self.aps))

        else:
            self.aps = eval_OXF_PAR( self.dataset, self.ranks_original, self.id_keyframes, self.id_queries_topics )
            print "Original mAP={}".format(np.mean(self.aps.values()))


        # init extractor

    def do_relevance_feedback( self, id_ima, similar_list ):

        p_annotations = similar_list['positive']
        n_annotations = similar_list['negative']

        # Get the descriptor of the main query
        bow_query, idx = map_file_query(id_ima.split('.')[0], self.ds, self.bow_queries, True )

        X = bow_query   # it starts with the main descriptor, we know it is positive
        count_p = 1     #we know there is already one positive query

        for id_image in p_annotations:
            bow_p, id_p = map_file_OXFPAR( id_image.split('.')[0], self.ds, self.bow_targets, True )
            X = vstack((X, bow_p))      # create a sparse matrix adding each new descriptor
            count_p+=1

        # we know that the counter of negative images starts with 0
        count_n = 0
        for id_image in n_annotations:
            bow_n, id_n = map_file_OXFPAR( id_image.split('.')[0], self.ds, self.bow_targets, True )
            X = vstack((X, bow_n))      # add to sparse matrix the negative descriptors
            count_n+=1

        # create the labels matrix with the first count_p rows to 1
        Y = np.zeros( (count_p+count_n) )
        Y[:count_p]=1

        # normalitze matrix X ??

        # entrenar linearSVM
        model = LinearSVC()
        model.fit(X, Y)

        # clasificar self.bow_targets i ordenar el resultat (np.argsort)
        ranks, distances = compute_ranks( self.bow_targets,  bow_query)
        # treure id_n de ranks
        # embed()

        targets_sorted  = self.bow_targets[ranks,...]
        id_sorted = self.id_keyframes[ranks]


        # targets = self.bow_targets
        prediction = model.predict(targets_sorted)
        scores_samples = model.decision_function(targets_sorted)

        # sort scores in descendent order (bigger first)
        new_rank = np.argsort( scores_samples )[::-1]
        # new_rank = sorted(scores_samples, key=float, reverse=True) # descending order
        # new_rank = np.array(new_rank)
        json_file = store_json_rank( self.path_ranks, id_ima, new_rank, id_sorted, self.top_n, self.dataset)

        return json_file

    def get_rank_image( self, id_ima, url, encoded_image, similar_list=[], evaluate=False ):
        """
        Get rank for an image id

        args:
        id_ima: name of jpg image file
        url: can be url to jpg or jpg abs path in server
        encoded_image: base64 encoded image
        [TODO] -- modify for url case!
        """
        # for image from outside the dataset

        ranks=None
        json_file = os.path.join(self.path_ranks, id_ima+'.json')

        if id_ima == 'unknown_id':
            print "case 0 -- {}".format(id_ima)
            # check we have encoded url
            if encoded_image is not None and url is None:
                # print "Encoded image -- {} {}".format(encoded_image, url)
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
                print "url as path -- {}".format(url)
                ima_path = cv2.imread(url)
            # in the case it is an actual url download image and process
            else:
                print "decoding url -- {}".format(url)
                # get url
                resp = urllib.urlopen(url)
                ima_path = np.asarray(bytearray(resp.read()), dtype="uint8")
                ima_path = cv2.imdecode(ima_path, cv2.IMREAD_COLOR)

            bow_query = self.model.get_representation( ima_path )

        else:
            #process image from dataset
            id_ima = str(id_ima.split('.')[0])
            # check if json has been computed
            count = 1

            # Check query id and load computed query
            if self.dataset == 'instre':
                try:
                    idx = np.where( id_ima+'.jpg'==self.id_queries )[0][0]
                    bow_query = self.bow_queries[idx]
                except:
                    idx = np.where( id_ima+'.jpg'==self.id_keyframes )[0][0]
                    bow_query = self.bow_targets[idx]

                # query expansion mode...
                id_for_QE = []
                descriptors_QE = []

                # for query expansion mode...
                for ima_id_qe in similar_list:
                    # get id within the dataset
                    id_QE =  np.where( ima_id_qe+'.jpg'==self.id_keyframes )[0][0]
                    bow_query+= self.bow_targets[idx]
                    count+=1

                if count >1:
                    bow_query /= count
                    bow_query = normalize(bow_query)
                    print "New query from {} annotated images".format( count )


            else:
                # case Oxford/Paris
                try:
                    bow_query, idx = map_file_query( id_ima, self.ds, self.bow_queries, True )
                except:
                    bow_query, idx = map_file_OXFPAR( id_ima.split('.')[0], self.ds, self.bow_targets, True )

                # query expansion mode...
                id_for_QE = []
                descriptors_QE = []
                # positive_examples = []

                # for query expansion mode...
                for ima_id_qe in similar_list:
                    # get is
                    bow_QE, id_QE = map_file_OXFPAR( ima_id_qe.split('.')[0], self.ds, self.bow_targets, True )

                    bow_query+= bow_QE
                    count+=1
                    # positive_examples.append(bow_QE)

                if count >1:
                    bow_query /= count
                    bow_query = normalize(bow_query)
                    print "New query from {} annotated images".format( count )


            if bow_query is None:
                # PROCESS IMAGE
                print "Computing new features for {}".format( id_ima )
                ima_path = os.path.join( PATH_IMAGES[self.dataset], id_ima+'.jpg' )
                # if not computed...
                bow_query = self.model.get_representation( ima_path )

        # do ranking and store results
        if os.path.exists( json_file ):
            print "Query rank computed! {}".format( json_file )
        else:

            # compute ranks for query
            ranks, distances = compute_ranks( self.bow_targets,  bow_query)

            #evaluate
            #store_plot( id_ima, ranks, self.ds )
            # store txt
            json_file = store_json_rank( self.path_ranks, id_ima, ranks, self.id_keyframes, self.top_n, self.dataset)


        # evaluate
        if evaluate:
            # get rank status
            ranks_all = self.ranks

            # compute ranks of query is not computed
            if ranks is None:
                ranks, distances  = compute_ranks( self.bow_targets,  bow_query)

            # update query
            ranks_all[...,idx] = ranks

            #evaluate
            if self.dataset == 'instre':
                aps = eval_instre( ranks_all )
                ap_query = aps[idx]

                print "Global mAP from {} to {}".format(np.mean( self.aps ), np.mean( aps ))
                print "Query-{} AP from AP={} to AP={}".format( idx, self.aps[idx], aps[idx] )

            else:
                print "evaluating {}".format(self.dataset)
                aps = eval_OXF_PAR(self.dataset, ranks_all, self.id_keyframes, self.id_queries_topics)

                # map to topic
                t = self.id_queries_topics[idx]
                ap_query = aps[t]

                print "Global mAP from {} to {}".format(np.mean( self.aps.values() ), np.mean( aps.values() ))
                print "Query-{} AP from AP={} to AP={}".format( idx, self.aps[t], aps[t] )

            result = {"json": json_file, "initial": self.aps[t], "final": aps[t]}
            # update ranks and aps
            self.ranks = ranks_all
            self.aps = aps

            return result
        else:
            return json_file
