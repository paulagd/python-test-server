## **Python test server**

This repository has been built just as an example to make it easy to plug in a new CBIR system.
All the functions are free to change and all the code can be adapted and reused in order to
plug in a new system.

> The full code of the CBIR used is available in https://github.com/imatge-upc/salbow.

### How needs the project to be structured?

The project provides a python server that allows to merge your code with the whole project.
To do that, a `main server` file is created and it will be the one which connects your code with
the `nodejs` server.

  1. Install the `zerorpc` library.

      ```
      pip install zerorpc
      ```

      > Install also the other dependencies if needed.

  2. There is 2 already existing methods that you can modify. You can also add other
  methods if required. A briefed explanation is given in this section and a full one
  is available in the `Main Functions` section as well.

      - `postServer` method

          This method receives all the parameters needed to compute the ranking
          of a query given.

          First of all, the method is defined like:

          ```py
          def postServer(self, id_img, url, encoded_image, dataset, path):
          ```

          Then, a code is done in order to compute the ranking. You can put here your
          own code but it should return a `json structure` with the keys `IdSequence` and `Image` as
          it can be seen in the following example.

          ```json
          [
            {
                "IdSequence": "0",
                "Image": "paris_general_002391"
            },
            {
                "IdSequence": "1",
                "Image": "paris_eiffel_000128"
            },
                 ...

            {
                "IdSequence": "5011",
                "Image": "paris_invalides_000541"
            }
          ]
          ```
          > Upper and lower case matters in this structures.

      - `postFeedback_And_Update` method

          This method receives all the parameters needed to get the feedback of the user.
          You should return the updated ranking after computing the desired experiments.

          First of all, a method is defined like:

          ```py
          def postFeedback_And_Update(self, id_img, url, encoded_image, dataset, path, similar_list, mode):

          ```

          Then, you can code whatever you need in order to return a json object with
          the following shape (it changes depending on the mode):

          * QE MODE:

          ```json
          {
              "json": [{},...,{}],
              "initial": 0.700725,
              "final": 0.639502
           }
          ```

           * Annotation mode
           ```json
           {
              "json": [{},...,{}],
              "success": true
           }
          ```

          > The `json` field will contain the ranking updated and it should have the same
          structure as in the method above.
