Search engine scripts
---------------------

--> Need to update config.py to point to the right files.
--> ``python rank_generator'' generates the ranks (json files) for a given dataset
--> Default config for function is defined in config.py file:

              params = {
                'network': 'vgg16',
                'query_expansion':False,
                'max_dim': 340,
                'layer': 'conv5_1',
                'dataset':'oxford',
                'mask':None,
                'mode_query': 'crop',
                'weighting': None
              }
