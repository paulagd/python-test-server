import zerorpc
import requests
import logging
import sys, json

# from search_engine.src.CPUrank_generator import *

class MainServer(object):

    """
    @api {post} /sendFeedback_receiveRanking/:id  postFeedback_And_Update()
    @apiName postFeedback_And_Update
    @apiGroup Main Functions

    @apiParam {String} id_img Id of the query
    @apiParam {String} [url] Url of the image introduced to the system.
    @apiParam {String} [encoded_image] Encoded image uploaded to the system.
    @apiParam {String} dataset Dataset given
    @apiParam {String} [path] Path given in the case of 'complicated' datasets as 'instre'.
    @apiParam {String} similar_list List of the queries selected for the query expansion.
    @apiParam {String} mode Mode selected to be in use.

    @apiDescription This function collects the annotations of the users. Depending on the
    mode selected by the user, it computes multiquery in order to improve the accuracy of
    the system (if in QE mode) or it trains an SVM (if Annotation mode). However,
    this is just our example and it can be modified.

    @apiExample {json} Request (QE mode)
          {
              "similar_list": [" paris_general_001620.jpg", " paris_general_002391.jpg", " paris_eiffel_000128.jpg"],
              "dataset":"paris",
              "mode": "q",
              "url": null,
              "encoded_image": null,
              "path": null
          }

     @apiExample {json} Request (Annotation mode)
          {
             "similar_list": {
                               "positive": ["paris_general_001620.jpg", "paris_eiffel_000128.jpg"],
                               "negative": ["paris_general_000144.jpg", "paris_general_002444.jpg"]
                },
              "dataset": "groups",
              "mode": "a",
              "url": null,
              "encoded_image": null,
              "path": null
          }

    """

    def postFeedback_And_Update(self, id_img, url, encoded_image, dataset, path, similar_list, mode):
        # default action

        if dataset.lower() == 'groups':
            path_random_ranking = "groups/rand_qimList_groups.txt"
            ranking = []

        with open("./lists_test/"+path_random_ranking, 'r') as data:
            for i, item in enumerate(data):
                ranking.append({"Image": item.strip().split('.')[0].decode("utf-8") , "IdSequence": i})  #starting id in 1

        # mode query expansion
        if mode == 'q':
            result = {'json': {}, 'initial': 0, 'final':0 }

            result['json'] = ranking
            result['initial'] = 0.8
            result['final'] = 0.9

            return result

        # mode annotations
        elif mode == 'a':

            result = {'json': {}, 'success': False }
            result['json'] = ranking
            result['success'] = True

            return result

    """
    @api {post} /getRankinById/:id  postServer()
    @apiName postServer
    @apiGroup Main Functions

    @apiParam {String} id_img Id of the query
    @apiParam {String} [url] Url of the image introduced to the system.
    @apiParam {String} [encoded_image] Encoded image uploaded to the system.
    @apiParam {String} dataset Dataset given
    @apiParam {String} [path] Path given in the case of 'complicated' datasets as 'instre'.

    @apiDescription This function generates the ranking of similarity given a query image.
    Depending on the dataset that is selected by the user, we compute one specific funtion.
    However, it is again just an example that can be modified. In the example, the datasets
    oxford, paris and instre are using the `search_engine` example in order to compute the rankings.
    On the other hand, the dataset groups is returning just a random list generated previously
    by the scripts given in the `node-server`. Feel free to use whichever suits you or either
    introduce your own system.


    @apiExample {json} Request
          {
              "dataset":"groups",
              "url": null,
              "encoded_image":null,
              "path": null
          }
    """
    def postServer(self, id_img, url, encoded_image, dataset, path):

        if dataset.lower() == 'groups':
            path_random_ranking = "groups/rand_qimList_groups.txt"
            ranking = []

            with open("./lists_test/"+path_random_ranking, 'r') as data:
                for i, item in enumerate(data):
                    ranking.append({"Image": item.strip().split('.')[0].decode("utf-8"), "IdSequence": i})  #starting id in 1

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
