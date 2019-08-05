# SharpNet
*Fast and Accurate Recovery of Occluding Contours in Monocular Depth Estimation*, 
by Michaël Ramamonjisoa and Vincent Lepetit.

Link to the paper: [arXiv](https://arxiv.org/abs/1905.08598)

<p align="center">
  <img src="architecture.png" width="800"/>
</p>

### Installation

Make sure you have installed the following requirements:

- Python3
- [Pytorch](https://pytorch.org/get-started/locally/)
- OpenCV
- numpy, imageio, Pillow, matplotlib

Clone the repository and download the trained weights:
- [on PBRS](https://drive.google.com/open?id=1NahBpG1AXNlWItcb9Uf9VXHmD8iSCntZ)
- [after finetuning on NYUv2](https://drive.google.com/open?id=1UTruzxPxQdoxF44X7D27f8rISFU0bKMK)

Weights trained on NYUv2 should be used for depth estimation, ***however*** weights trained only on synthetic data 
provide sharper normals and contours predictions.

```
git clone https://github.com/MichaelRamamonjisoa/SharpNet.git
cd SharpNet
mkdir models && cd models
```

Put the trained weights in the models/ directory.

## Demo

### On your test image
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

The network was trained using 640x480 images, therefore better results might be 
observed after rescaling the image with $SCALEFACTOR different than 1. 

Here is what you can get on your test image:
![alt_text](https://github.com/MichaelRamamonjisoa/MichaelRamamonjisoa.github.io/blob/master/images/SharpNet_thumbnail.gif)

If you want to display the predictions, use the --display flag.

### Live demo
To run the live version of SharpNet, connect a camera and run demo.py with the --live flag.
- Make sure your camera is detected by OpenCV beforehand.
- Press 'R' on your keyboard to switch between normals, depth, contours and RGB
- Press 'T' to save your image and its predictions
- Press 'Q' to terminate the script


## Training

The PBRS dataset is currently offline due to instructions of SUNCG author (see 
[this](https://github.com/yindaz/pbrs/issues/11) and [this](https://github.com/shurans/SUNCGtoolbox/issues/32)). 
Therefore reproduction of our training procedure cannot be done properly. However we will provide code for loss
computation, finetuning on the NYUv2 Depth dataset as well as our pretrained weights on the PBRS dataset only.

## Evaluation

TODO

## Citation

If you find SharpNet useful in your research, please consider citing:
```
@article{ramamonjisoa2019sharpnet,
    Title = {SharpNet: Fast and Accurate Recovery of Occluding Contours in Monocular Depth Estimation},
    Author = {Michael Ramamonjisoa and Vincent Lepetit},
    Journal = {arXiv preprint arXiv:1905.08598},
    Year = {2019}
}
```
