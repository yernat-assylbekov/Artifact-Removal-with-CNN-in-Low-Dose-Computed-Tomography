# Artifact Removal with CNN in Low-Dose Computed Tomography
In this project, I train U-Net type Neural Network known as FBPConvNet from scratch to remove artifacts in low-dose Computed Tomography (CT) in medial imaging. The use of U-Net type architecure in CT was proposed in the paper <a href="https://arxiv.org/abs/1611.03679">[1]</a> of Jin et al.

## Dataset

The dataset was downloaded from https://www.kaggle.com/kmader/nih-deeplesion-subset which consists of 32,120 axial CT slices from 10,594 CT scans (studies) of 4,427 unique patients. Due to a limited resources, the images were preprocessed and resized to 128x128. Those images are used as ground truth. To make low-dose CT reconstruction images, we first create sinograms via Radon transform with 50 sparse angles. Then we invert these sinograms via filtered backprojection (FBP) algorithm. To implement these two steps we use `radon` and `iradon` functions from `skimage.transform` package. Here are sample images from the dataset

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/images/sample_data.png?raw=true)<br>

Top row is for reconstructed images, which contain some artifacts, via FBP and the bottom row is for ground truth images.<br>

I trained on 32,256 images, validated on 512 images and tested on the rest 566 images.

## Network Architecture

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/images/u_net_architecture.png?raw=true)<br>

## Training Details

I use noise-to-signal ratio (<img src="https://render.githubusercontent.com/render/math?math=\text{NSR}">) as a loss function.  If <img src="https://render.githubusercontent.com/render/math?math=\displaystyle x=(x_{ij})_{i,j=1}^n"> is a ground nxn truth image and <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \hat x=(\hat x_{ij})_{i,j=1}^n"> is the reconstructed nxn image, then<br>
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\text{NSR}(x,\hat x)=\frac{\sqrt{\sum_{i,j=1}^n (x_{ij} - \hat x_{ij})^2}}{\sqrt{\sum_{i,j=1}^n x_{ij}^2}}.">
</p>

I use the Adam optimizer with `learning_rate = 0.001`, `beta_1 = 0.9` and `beta_2 = 0.999`. I trained the model with an NVIDIA K80 GPU for 30 epochs for approximately 1 hour and 17 minutes. 

## Results

We use logSNR (logarithmic signal-to-noise ratio) and SSIM (structural similarity index measure) as quantitative metrics. If <img src="https://render.githubusercontent.com/render/math?math=\displaystyle x=(x_{ij})_{i,j=1}^n"> is a ground nxn truth image and <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \hat x=(\hat x_{ij})_{i,j=1}^n"> is the reconstructed nxn image, then<br>
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\text{logSNR}(x,\hat x)=20\log_{10}\frac{\sqrt{\sum_{i,j=1}^n x_{ij}^2}}{\sqrt{\sum_{i,j=1}^n (x_{ij} - \hat x_{ij})^2}}.">
</p>

Here is the learning curve:

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/images/learning_curve.png?raw=true)<br>

Here is the progress for the average logSNR:

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/images/avg_logSNR.png?raw=true)<br>

Below we give examples of comparison of reconstructed images using FBP and FBPConvNet on the training, validation and test sets:

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/images/training_data_ex.png?raw=true)<br>

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/images/validation_data_ex.png?raw=true)<br>

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/images/testing_data_ex.png?raw=true)<br>

Below is the table of quantitative comparison of FBP and FBPConvNet on the test set using logSNR and SSIM:

![alt text](https://github.com/yernat-assylbekov/Artifact-Removal-with-CNN-in-Low-Dose-Computed-Tomography/blob/master/images/table_of_comparison_numeric.png?raw=true)<br>

## References

<a href="https://arxiv.org/abs/1611.03679">[1]</a> K.H. Jin, M.T. McCann, E. Froustey and M. Unser, <i>Deep Convolutional Neural Network for Inverse Problems in Imaging</i>, IEEE Transactions on Image Processing (2017), 26, no. 9, 4509-4522
