# Stacking Networks Dynamically for Image Restoration Based on the Plug-and-Play Framework (ECCV2020)

## Inventory

code: Root directory of our framework implementation.

Supplementary material.pdf: It includes supplementary visualization results of our framework compared with the current state-of-the-art. They are compared on the dataset Set12 used in our paper but are not provided due to the page limit.

## Getting started 

The below sections elaborate on how to run our framework.

### Requirements 

Python 3.6 
PyTorch 1.1 
Jupyter Notebook 
Cuda 10.0


```python
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```

### Pretrained priors 

In our paper, we exploit three main pretrained deep denoiser priors and a deblurrer prior according to their own training methods.

https://github.com/joeylitalien/noise2noise-pytorch

https://github.com/SaoYan/DnCNN-PyTorch

https://github.com/zsyOAOA/VDNet

https://github.com/HongguangZhang/DMPHN-cvpr19-master

For convenience, we also put them in the folder train.
Then the pretrained deep priors are directly inserted into our framework. We also provide our
pretrained models in the next.


### Framework 

The implementation details are provided in the folder framework.

### Dataset

The datasets exploited in the paper are BSD68 and Set12 both under the folder framework/data. And the motion blur dataset is accessed online easily.

### Pretrained deep priors 

The pretrained deep priors used in our framework are under the folder framework/checkpoints.

### Results 

Run the Jupter Notebook files under the folder framework to evaluate performances of our framework.

