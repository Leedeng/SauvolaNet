# SauvolaNet: Learning Adaptive Sauvola Network

<div align="left">
    <img src="https://www.um.edu.mo/wp-content/uploads/2020/09/UM-Logo_V-Black-1024x813.png" width="30%"><img src="https://viplab.cis.um.edu.mo/images/logo_5.JPG" width="30%"><img src="https://icdar2021.org/wp-content/uploads/icdar2021-logo.png" width="30%">     
</div>

***

This is the official repo for the SauvolaNet (ICDAR2021). For details of SauvolaNet, please refer to 

```

```

***

# Overview

SauvolaNet is an end-to-end document binarization solution. It is optimal for three hyper-parameters of the classic Sauvola algorithm. Compare with existing solutions, SauvolaNet has followed advantages:

- **SauvolaNet do not have any Pre/Post-processing**
- **SauvolaNet has comparable performance with SoTA**
- **SauvolaNet has a super lightweight network structure and faster than DNN-based SoTA**

<img src="https://github.com/Leedeng/SauvolaNet/blob/main/Image/FPS.png" width="50%">

More precisely, SauvolaNet consists of three modules, namely, Multi-window Sauvola (MWS), Pixelwise Window Attention (PWA), and Adaptive Sauolva Threshold (AST).

- **MWS generates multiple windows of different size Sauvola with trainable parameters**
- **PWA generates pixelwise attention of window size**
- **AST generates pixelwise threshold by fusing the result of MWS and PWA.**

<img src="https://github.com/Leedeng/SauvolaNet/blob/main/Image/Structure2.png" width="50%">

# Dependency

LineCounter is written in TensorFlow.
  
  - TensorFlow-GPU: 1.15.0
  - keras-gpu 2.2.4 
  
Other versions might also work but are not tested.


# Demo

Download the repo and create the virtual environment by following commands

```
conda create --name LineCounter --file spec-env.txt
conda activate Sauvola
pip install tensorflow-gpu==1.15.0
pip install opencv-python
pip install parse
```

Then play with the provided ipython notebook.

Alternatively, one may play with the inference code using this [google colab link](https://colab.research.google.com/drive/1aGYXVRuTf1dhoKSsOCPcB4vKULtplFSA?usp=sharing).

# Datasets
We do not own the copyright of the dataset used in this repo.

Below is a summary table of the datasets used in this work along with a link from which they can be downloaded:


| Dataset      | URL     |
| ------------ | ------- |
| DIBCO 2009   | http://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/   |
| DIBCO 2010   | http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/ |
| DIBCO 2011   | http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/        |
| DIBCO 2012   | http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/       |
| DIBCO 2013   | http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/        |
| DIBCO 2014   | http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/  |
| DIBCO 2016   | http://vc.ee.duth.gr/h-dibco2016/benchmark/                 |
| DIBCO 2017   | https://vc.ee.duth.gr/dibco2017/                 |
| DIBCO 2018   | https://vc.ee.duth.gr/h-dibco2018/                 |
| PHIDB | http://www.iapr-tc11.org/mediawiki/index.php/Persian_Heritage_Image_Binarization_Dataset_(PHIBD_2012) |
| Bickely-diary dataset | https://www.comp.nus.edu.sg/~brown/BinarizationShop/dataset.htm |
| Synchromedia Multispectral dataset | http://tc11.cvc.uab.es/datasets/SMADI_1　|
| Monk Cuper Set | https://www.ai.rug.nl/~sheng/ |


# Concat

For any paper-related questions, please feel free to contact leedengsh@gmail.com.
