import zerorpc
import requests
import logging
import sys, json

class MainServer(object):

    def postFeedback_And_Update(self, id_img, url, encoded_image, dataset, path, similar_list, mode):
        # default action
        # if dataset.lower() == 'instre' and (url is None):
        #     id_img = path.split(".")[0]
        #
        # if 'unicode' in str(type(encoded_image)):
        #     url = None
        #
        # model_dic = {}
        # if dataset not in model_dic.keys():
        #     params = {
        #         'network': 'vgg16',
        #         'query_expansion':False,
        #         'max_dim': 340,
        #         'layer': 'conv5_1',
        #         'dataset':str(dataset).lower(),
        #         'weighting':None,
        #         'mode_query': 'crop',
        #         'top_n': 5012
        #     }
        #     model_dic[dataset] = BLCF(params)
        #
        # #
        # # similar_list --> arrray de querys de la forma [" keble_000214.jpg", " new_000283.jpg", etc]
        # # mode  = 'q' (for query query_expansion) or 'a' for annotation mode
        #
        # # mode query expansion
        # if mode == 'q':
        #     # ensure id_imq is one of the queries
        #     result = model_dic[dataset].get_rank_image( id_img, url, encoded_image, similar_list, True  )
        #     json_file = result['json']
        #
        #     with open(json_file) as data_file:
        #        data = json.load(data_file)
        #
        #     result['json'] = data
        #     return result
        #
        # elif mode == 'a':
        #     json_file = model_dic[dataset].do_relevance_feedback( id_img, similar_list )
        #     with open(json_file) as data_file:
        #        data = json.load(data_file)
        #
        #     result = {'json': {}, 'success': False }
        #     result['json'] = data
        #     result['success'] = True
        #     return result
        return {}

    def postServer(self, id_img, url, encoded_image, dataset, path):

        if dataset.lower() == 'groups':
            path_random_ranking = "groups/rand_qimList_groups.txt"
            ranking = []

            with open("./lists_test/"+path_random_ranking, 'r') as data:
                for i, item in enumerate(data):
                    ranking.append({"Image": item.strip().split('.')[0], "IdSequence": i})  #starting id in 1

            print('---------- RANKIN SENT ---------')
            return ranking
        else:
            raise ValueError('There is no qimList generated for this dataset.')
            return ValueError('There is no qimList generated for this dataset.')


logging.basicConfig()
s = zerorpc.Server(MainServer())
s.bind("tcp://0.0.0.0:4243")
print("Python server listening on: 4243")
s.run()
sys.stdout.flush()
