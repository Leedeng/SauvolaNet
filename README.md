# SauvolaNet: Learning Adaptive Sauvola Network

<div align="left">
    <img src="https://www.um.edu.mo/wp-content/uploads/2020/09/UM-Logo_V-Black-1024x813.png" width="30%"><img src="https://viplab.cis.um.edu.mo/images/logo_5.JPG" width="30%"><img src="https://icdar2021.org/wp-content/uploads/icdar2021-logo.png" width="30%">     
</div>

***

This is the official repo for the SauvolaNet (ICDAR2021). For details of SauvolaNet, please refer to 

```
@article{li2021sauvolanet,
  title={SauvolaNet: Learning Adaptive Sauvola Network for Degraded Document Binarization},
  author={Li, Deng and Wu, Yue and Zhou, Yicong},
  journal={arXiv preprint arXiv:2105.05521},
  year={2021}
}

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


# Concat

For any paper-related questions, please feel free to contact leedengsh@gmail.com.
