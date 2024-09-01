# pix2pix with GAN
## Description
___
This model transforms people's faces into a comic.
___
### Architecture
Using the PyTorch library, the pix2pix GAN architecture was implemented: 
1) U-net was used as a generator for images with a resolution of 128x128;
2) A classical neural network with convolution in the last layer (PathGAN architecture) was used as a discriminator

<img src='img/archpng.png' heigth='300'>

### Dataset
[Comic faces (paired, synthetic)](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic) was used as a dataset:

<img src='https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/8f0af1e5-5587-468f-acb5-d8ad4796d8a1' width="300">


### Learning
During the training, [nn.L1Loss()](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) (Mean Absolute Error) and [nn.BCEWithLogitsLoss()](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) were used.

[torch.optim.Adam()](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) was used as optimazer.

> Start training:

<img src='https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/fc63c7f4-4458-4df1-816c-29df34af849d' width="400">

> During training:

<img src='https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/a4f9b0d0-c8c9-443b-9a9a-46907e37691c' width="400">

> End of training:

<img src='https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/37097430-adc7-400d-beba-9e0ecae1b6c9' width="400">

## Usage
In the arguments of the load_model method, after the model, specify the paths to the [discriminator and generator weights](https://www.kaggle.com/datasets/markovka/pix2pix128)

![image](https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/d2d74f98-faa5-4579-8fad-4a81e5d4b04b)

Run file bot.py
1. In console of your computer
2. In cell of notebook (.ipynb) (example: `!python3 /kaggle/input/gan-tg-bot/bot.py`)

### Link to the Bot
https://t.me/face2comic_bot

<img src='img/qr.png' width="300">

To get started, write "/start"

### Examples

<img src='https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/e48b17e6-5458-4e51-8b89-41b81e385ba3' width="400">

<img src='https://github.com/Mikhail-bmstu/pix2pix_GAN/assets/83812505/4d8c4e87-64f8-48e0-aab3-fc3d9ef902d4' width="400">

