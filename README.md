## ASC Net summary

### Introduction
ASC-Net is a framework which allows us to <strong>define a Reference Distribution Set</strong> and then take in <strong>any Input Image</strong> and <strong>compare with the Reference Distribution</strong> and <strong>throw out anomalies</strong> present in the Input Image. 

### Archive Link 

https://arxiv.org/pdf/2103.03664.pdf

### Highlights

1. Solves the difficulty in defining a class/set of things deterministically down to the nitty gritty details. The Reference Distribution can work on any combination of image set and abstract out the manifold encompassing them.
2. No need of perfect reconstruction. We care about the anomaly not the reconstruction unlike other existing algorithms.
3. We can potentially define any manifold using Reference Distribution and then compare any incoming input image to it.
4. Works on any image sizes. Simply adjust the size of the encoder/decoder sets to match your input size and hardware capacity.

### Network Architecture

![alt text](img/ASCnet.PNG)

### High level Summary [Short Video]


[![Click for a short vid](img/YT.PNG)](https://www.youtube.com/watch?v=oUeBNOYOheg)


### Code

## Built With/Things Needed to implement experiments

* [Python](https://www.python.org/downloads/) - Python-2 
* [Keras](http://www.keras.io) - Deep Learning Framework used
* [Numpy](http://www.numpy.org/) - Numpy
* [Sklearn](http://scikit-learn.org/stable/install.html) - Scipy/Sklearn/Scikit-learn
* [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) - CUDA-8
* [CUDNN](https://developer.nvidia.com/rdp/assets/cudnn_library-pdf-5prod) - CUDNN-5 You have to register to get access to CUDNN
* [Brats 2019](https://ipp.cbica.upenn.edu/) - Select Brats 2019
* [LiTS](https://competitions.codalab.org/competitions/17094) - LiTS Website
* [MS-SEG 2015](https://smart-stats-tools.org/lesion-challenge) - MS-SEG2015 website
* [12 gb TitanX]
