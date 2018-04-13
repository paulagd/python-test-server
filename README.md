# python-test-server
A python test server to incorporate to the "CBIR visualization tool" and its server.

### Prerequisits

* Create a virtual enviroment [how to create a virtual enviroment](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
```
 virtualenv ~/BLCF
 source ~/BLCF/bin/activate
```
* For python dependencies run:
```
 pip install --upgrade pip
 pip install -r requirements.txt
```
* Make sure you are using theano order for the images. Edit '~/.keras/keras.json'
```
{    "image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}
 ```

[NOTE] - In MAC_OS, matplotlib might cause problems using a python virtual enviroment. You might need to Create a file '~/.matplotlib/matplotlibrc' there and add the following code: 'backend: TkAgg'.

* If you want the system to be evaluated, follow the next steps:
      1. In the main project folder, create a file named `compute_ap.cpp` and copy/paste the code from [HERE](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp).
      2. In your shell, from the same path, run the following code:

        ```
          > gcc compute_ap.cpp -o test -lstdc++
          > ./test
          > rm compute_ap.cpp
          > mv test compute_ap
        ```

### Setting the paths and models
* Set the path to the main folder in 'src/config.py'
```
MAIN_PATH = '/path/to/python-server/search_engine'
```
* Download the [data](https://drive.google.com/open?id=1Too0gpYgqAk287YE1Fh5qjtr-9AyEHdt) folder and place within the 'search_engine' folder.
```
search_engine/data
```
### Execution

* To run the code run:
```
python main.py    
```

### Documentation

Install apidoc
```
npm install -g apidoc

```

Run documentation generation script:
```
apidoc -o documentation -e node_modules

```

> Open the file 'index.html' stored in the folder 'documentation' to see how to customize the system.
