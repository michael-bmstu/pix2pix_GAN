# pix2pix with GAN
## Description
___
This model transforms people's faces into a comic.
___
### Architecture
Using the PyTorch library, the pix2pix GAN architecture was implemented: 
1) U-net was used as a generator for images with a resolution of 128x128;
2) A classical neural network with convolution in the last layer (PathGAN architecture) was used as a discriminator

![model architecture](img/archpng.png)
### Dataset
[Comic faces (paired, synthetic)](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic) was used as a dataset:

![sample](img/face2_comics_sample_large.jpg)

### Learning
During the training, nn.L1Loss() (Mean Absolute Error) and nn.BCEWithLogitsLoss() were used.

torch.optim.Adam() was used as optimazer.

> Start training:

![image](https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/a4f9b0d0-c8c9-443b-9a9a-46907e37691c)

> During training:

![image](https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/37097430-adc7-400d-beba-9e0ecae1b6c9)

> End of training:

...

## Usage
In the arguments of the load_model method, after the model, specify the paths to the discriminator and generator weights
