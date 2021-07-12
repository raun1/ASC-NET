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


## Code

### Dependencies/Environment used

* [CUDA](https://developer.nvidia.com/cuda-90-download-archive) - CUDA-9.0.176
* [CUDNN](https://developer.nvidia.com/cudnn-download-survey) - CUDNN- Major 6; Minor 0; PatchLevel 21 
* [Python](https://www.python.org/downloads/) - Version 2.7.12 
* [Tensorflow](https://www.tensorflow.org/install) - Version 1.10.0
* [Keras](http://www.keras.io) - Version 2.2.2
* [Numpy](http://www.numpy.org/) - Version 1.15.5
* [Nibabel](https://nipy.org/nibabel/) - Version 2.2.0
* [Open-CV](https://opencv.org/releases/) - Version 2.4.9.1
* [Brats 2019](https://ipp.cbica.upenn.edu/) - Select Brats 2019
* [LiTS](https://competitions.codalab.org/competitions/17094) - LiTS Website
* [MS-SEG 2015](https://smart-stats-tools.org/lesion-challenge) - MS-SEG2015 website
* [12 gb TitanX]
