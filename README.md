Network Architecture for the MICCAI_2021 paper : ASC-NET: Adversarial-based Selective Network for Unsupervised Anomaly Segmentation. To view the paper on Archive click the following https://arxiv.org/pdf/2103.03664.pdf

![alt text](img/ASCnet.PNG)
## ASC Net summary

### Introduction
ASC-Net is a framework which allows us to <strong>define a Reference Distribution Set</strong> and then take in <strong>any Input Image</strong> and <strong>compare with the Reference Distribution</strong> and <strong>throw out anomalies</strong> present in the Input Image. 

### Highlights

1. Solves the difficulty in defining a class/set of things deterministically down to the nitty gritty details. The Reference Distribution can work on any combination of image set and abstract out the manifold encompassing them.
2. No need of perfect reconstruction. We care about the anomaly not the reconstruction unlike other existing algorithms.
3. We can potentially define any manifold using Reference Distribution and then compare any incoming input image to it.
4. Works on any image sizes. Simply adjust the size of the encoder/decoder sets to match your input size and hardware capacity.

### High level Summary 

Click below for a short video explaining ASC-NET
[![Click for a short vid](https://github.com/raun1/ASC-NET/blob/42f3192248cc67ac111ab51c198a4c0dc87064c8/img/yt_logo_rgb_light.png)](https://www.youtube.com/watch?v=oUeBNOYOheg)

*ROI and CO branches - 
We take the downsampling branch of a U-Net as it is, however we split the upsampling branch into two halves, one to obtain the Region of Interest and the other for Complementary aka non region of interest. Losses here are negative dice for ROI and positive dice for Non-ROI region.*

*Reconstruction Branch - 
Next we merge these two ROI and non ROI outputs using "Summation" operation and then pass it into another U-Net, This U-Net is the reconstruction branch. The input is the summed image from previous step and the output is the "original" image that we start with. The loss of reconstruction branch is MSE.*

```
The code in this repository provides only the stand alone code for this architecture. You may implement it as is, or convert it into modular structure
if you so wish. The dataset of OASIS can obtained from the link above and the preprocessiong steps involved are mentioned in the paper. 
You have to provide the inputs.
```


email me - rd31879@uga.edu for any questions !! Am happy to discuss 

## Built With/Things Needed to implement experiments

* [Python](https://www.python.org/downloads/) - Python-2 
* [Keras](http://www.keras.io) - Deep Learning Framework used
* [Numpy](http://www.numpy.org/) - Numpy
* [Sklearn](http://scikit-learn.org/stable/install.html) - Scipy/Sklearn/Scikit-learn
* [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) - CUDA-8
* [CUDNN](https://developer.nvidia.com/rdp/assets/cudnn_library-pdf-5prod) - CUDNN-5 You have to register to get access to CUDNN
* [OASIS](https://www.oasis-brains.org/) - Oasis-dataset website
* [12 gb TitanX]- To implement this exact network

## Basic Idea


### Pre-requisites
This architecture can be understood after learning about the U-Net [https://arxiv.org/abs/1505.04597] {PLEASE READ U-NET before reading this paper} and W-Net [https://arxiv.org/abs/1711.08506] {Optional}.
* Please see line 1541 in comp_net_raw.py file in src for the main essence of complementary network - i.e. summing up the intermediate outputs of segmentation and complementary branches and then concatenating them for reconstruction layer.
* Hyper parameters to be set - 
* l2_Lambda - used for regularizing/penalizing parameters of the current layer
* Mainly used to prevent overfitting and is incorporated in the loss function
* Please see keras.io for more details
* DropP sets the % of dropout at the end of every dense block
* Kernel_size is the kernel size of the convolution filters
* Please see readme for additional resources.
* Lines 73 - 648 is the common encoder of the segmentation and complementary branches. 
* Layers such as xconv1a,xmerge1........ belong to the complementary upsampling branch branch of the architecture.
* The convolution layers's number indicates its level and so up6 and xup6 are at the same level
* and are parallel to each other
* Layers such as xxconv1a,xxmerge1 .... belong to the reconstruction branch. 
* For more details of the multi outputs please see my isbi repository here
https://github.com/raun1/ISBI2018-Diagnostic-Classification-Of-Lung-Nodules-Using-3D-Neural-Networks
* Basically to summarize, we have two branches one which has negative dice with ground truth brain mask 
 and is the segmentation branch
* We then have another branch with positive dice with ground truth masks
* The THEME of comp-net is to sum up the two sections, future works will provide a better way to do this and a generalized version :) 
* We do this theme of summing at every stage of the intermediate outputs i.e. the first intermediate output of segmentation branch 
 is summed with first intermediate output of the complementary branch.
* We obtain a final summary of the outputs of the segmentation branch and complementary branch and also sum these two new summaries
* Finally we concat all of these summations and send to the reconstruction branch
* reconstruction branch is a simple structure of dense multi-output U-Net and the ground truth is the input image and loss is MSE.


### Building your own Comp Net from whatever U-Net you have

* Copy the upsampling branch of your U-Net
* Duplicate it
* Use same loss functions as the original U-Net BUT change its sign {Warning - Make sure your loss function is defined for the opposite sign and try to think intuitively what it acheives. Example dice is simply overlap between two objects and optimizing negative dice gives us maximum possible overlap, but positive dice lowest value is 0 since you CANNOT quantify how much seperation is there between two objects using the DICE score but simply quantify if the two overlap or not and if they overlap how much }
* Add the two upsampling branch outputs pairwise for each channel using keras's model.add layer
* Feed that into the new reconstruction U-Net where the loss function is MSE with the Input image of the first U-Net i.e. the original  input


![alt text](https://github.com/raun1/Complementary_Segmentation_Network/blob/master/fig/sample_results.PNG)
Sample results

