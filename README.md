# SharpNet
Fast and Accurate Recovery of Occluding Contours in Monocular Depth Estimation

## Installation

Make sure you have installed the following requirements:

- Python3
- [Pytorch](https://pytorch.org/get-started/locally/)
- OpenCV
- numpy, scipy, Pillow, matplotlib

Clone the repository and download the trained weights:

```
git clone https://github.com/MichaelRamamonjisoa/SharpNet.git
cd SharpNet
mkdir models && cd models
wget https://drive.google.com/open?id=1UTruzxPxQdoxF44X7D27f8rISFU0bKMK
```

##Demo

Try the [demo.py](https://github.com/MichaelRamamonjisoa/SharpNet/blob/master/demo.py) 
script to test our network on your image :

```
python3 demo.py --image $YOURIMAGEPATH \
--cuda $CUDA_DEVICE_ID\
--model models/final_checkpoint_NYU.pth \
--normals \
--depth \
--boundary \
--bias \
--scale $SCALEFACTOR 
```

The network was trained using ***640x480 images***, therefore better results might be 
observed after rescaling the image with $SCALEFACTOR different than 1. 

##Training

TODO

##Evaluation

TODO

##Citation

If you find SharpNet useful in your research, please consider citing:
```
@article{ramamonjisoa2019sharpnet,
    Title = {SharpNet: Fast and Accurate Recovery of Occluding Contours in Monocular Depth Estimation},
    Author = {Michael Ramamonjisoa and Vincent Lepetit},
    Journal = {arXiv preprint arXiv:TODO},
    Year = {2019}
}
```