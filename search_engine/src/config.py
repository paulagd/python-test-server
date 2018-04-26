import os
MAIN_PATH = '/Users/paulagomezduran/Desktop/TFG_NO_GUARDAT/python-test-server/search_engine'
#MAIN_PATH = '/home/eva/2017/paula_thesis/python-server/search_engine'

# download the 'data googledrive and locate the folder in search_engine/data'
INV_FILE = os.path.join(MAIN_PATH, 'data/inverted_files')
CODEBOOK_FILES = os.path.join(MAIN_PATH, 'data/BLCF_models')
FIXED_QUERIES = os.path.join(MAIN_PATH, 'data/processed_queries')
PATH_LOAD_DATASET = os.path.join(MAIN_PATH, 'data/datasets')


PATH_IMLIST=os.path.join(MAIN_PATH, 'imlists/')

# CHANGED
# # set the path to the images in your computer
# PATH_IMAGES = {
#     'oxford': '/media/eva/Eva Data/Datasets/Oxford_Buildings/images/',
#     'paris': '/media/eva/Eva Data/Datasets/Paris_dataset/images/',
#     'instre': '/media/eva/Eva Data/Datasets/Instre/'
# }

# set by default in search_engine/data
PATH_RANKINGS = os.path.join(MAIN_PATH, 'ranks')

# parameters of BLCF model
params = {
  'network': 'vgg16',
  'query_expansion':False,
  'max_dim': 340,
  'layer': 'conv5_1',
  'dataset':'oxford',
  'mask':'gaussian',
  'mode_query': 'crop',
  'weighting': None
}
