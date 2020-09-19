# Artifact Removal with CNN in Low-Dose Computed Tomography
In this project, I train U-Net type Neural Network from scratch to remove artifacts in low-dose Computed Tomography (CT) in medial imaging. The use of U-Net type architecure in CT was proposed in the paper <a href="https://arxiv.org/abs/1611.03679">[1]</a> of Jin et al.

## Network Architecture

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/u_net_architecture.png?raw=true)<br>

## Training Details

I use noise-to-signal ratio (<img src="https://render.githubusercontent.com/render/math?math=\text{NSR}">) as a loss function.  If <img src="https://render.githubusercontent.com/render/math?math=\displaystyle x=(x_{ij})_{i,j=1}^n"> is a ground <img src="https://render.githubusercontent.com/render/math?math=\displaystyle n\times n"> truth image and <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \hat x=(\hat x_{ij})_{i,j=1}^n"> is the reconstructed <img src="https://render.githubusercontent.com/render/math?math=\displaystyle n\times n"> image, then<br>
<center><img src="https://render.githubusercontent.com/render/math?math=\displaystyle\text{NSR}(x,\hat x)=\frac{\sqrt{\sum_{i,j=1}^n (x_{ij} - \hat x_{ij})^2}}{\sqrt{\sum_{i,j=1}^n x_{ij}^2}}."> </center>


## References

<a href="https://arxiv.org/abs/1611.03679">[1]</a> K.H. Jin, M.T. McCann, E. Froustey and M. Unser, <i>Deep Convolutional Neural Network for Inverse Problems in Imaging</i>, IEEE Transactions on Image Processing (2017), 26, no. 9, 4509-4522
