# SauvolaNet: Learning Adaptive Sauvola Network

<div align="left">
    <img src="https://www.um.edu.mo/wp-content/uploads/2020/09/UM-Logo_V-Black-1024x813.png" width="30%"><img src="https://viplab.cis.um.edu.mo/images/logo_5.JPG" width="30%"><img src="https://icdar2021.org/wp-content/uploads/icdar2021-logo.png" width="30%">     
</div>

***

This is the official repo for the SauvolaNet (ICDAR2021). For metCancel changeshod details, please refer to 

```
  @inproceedings{Na,
      title={Na},
      author={Na},
      journal={Na},
      year={Na}
  }
```

***

# Overview

SauvolaNet is an end-to-end document binarization solutions. It optimal three hyper-parameters of classic Sauvola algriothim. Compare with exisiting solutions, SauvolaNet has follow advantages:

- SauvolaNet do not have any Pre/Post-processing
- SauvolaNet has comparable performance with SoTA
- SauvolaNet has super light network horticulture, faster than SoTA

# Dependency

SauvolaNet is written in the TensorFlow.
  
  - TensorFlow-GPU: 2.3.0
  
Other versions might also work, but are not tested.


# Demo

Donwload the repo and create virtual environment by follow commands

```
conda create --name Sauvola --file spec-env.txt
conda activate Sauvola
pip install tensorflow-gpu==2.3
pip install opencv-python
pip install pandas
pip install parse
```

Then play with the provided ipython notebook

Alternatively, one may play with the inference code using this [google colab link](https://colab.research.google.com/drive/1aGYXVRuTf1dhoKSsOCPcB4vKULtplFSA?usp=sharing).


# Concat

For any paper related questions, please feel free to contact leedengsh@gmail.com
