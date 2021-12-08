# Who is the best painter? 

Project of the course ML701 at MBZUAI. Our main motivation was to train a model using AI to learn the style of Claude Monet. 

Checking the techniques, we choose to explore state of the art and find a way to compare what had better results. 
We used as base two repositories:
- StyleGAN: [https://github.com/dvschultz/stylegan2-ada-pytorch](https://github.com/dvschultz/stylegan2-ada-pytorch)
- LapStyle: [https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/lap_style.md](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/lap_style.md)

Some of our results can be found at:
[Tensorboard and Results](https://github.com/bermudezarii/ML701_Who-s_the_best_painter/tree/main/StyleGAN_ADA/StyleGAN_ADA_Tensorboard_Images)

### With StyleGAN-ADA
![With StyleGAN-ADA](https://github.com/bermudezarii/ML701_Who-s_the_best_painter/blob/main/images_readme/StyleGAN_ADA_Results%20(1).jpg?raw=true)

### With LapStyle
![With LapStyle](https://github.com/bermudezarii/ML701_Who-s_the_best_painter/blob/main/images_readme/LapStyle_Results.png?raw=true)


# StyleGAN-Ada

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements. If using computers from the labs, load the module of cuda-11.1, to avoid problems with StyleGAN. Erase cache from nvcc if needed. 

```bash
which nvcc
module load cuda-11.1
which nvcc

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install psutil scipy

pip install click opensimplex requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

pip install opensimplex

pip install tensorboard

pip install ninja opensimplex torch==1.7.1 torchvision==0.8.2
```
You can do the following step (below) and put the weights in results, or download the zip from: 

[https://mbzuaiac-my.sharepoint.com/:u:/g/personal/ariana_venegas_mbzuai_ac_ae/EToiZYOXKtpGq93fga6OR7wBwbBuxH3WV7WX6f9aTT2dyQ?e=qqdjkF ](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/ariana_venegas_mbzuai_ac_ae/EToiZYOXKtpGq93fga6OR7wBwbBuxH3WV7WX6f9aTT2dyQ?e=qqdjkF )

and put it inside (pasting the zip and unzip there).


```bash
cd /home/{username_lab}/Documents/colab-sg2-ada-pytorch
```

have in mind to change the {username_lab} like 


```bash
cd /home/ariana.venegas/Documents/colab-sg2-ada-pytorch
```

## Usage

Open the Jupyter Notebooks provided for each style. If you need to run without notebooks, after being in the directory ('colab-sg2-ada-pytorch/stylegan2-ada-pytorch').  This is the configuration of how we started our 3rd experiment: 
```python
python train.py --nkimg=0 --snap=1 --gpus=1 --cfg='24gb-gpu' -- metrics=fid50k_full --outdir=./metrics_ml --data='/home/ariana.venegas/Documents/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/datasets/Monet_folder.zip' --resume='/home/ariana.venegas/Documents/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/pretrained/wikiart.pkl' --augpipe='bg' --initstrength=0 --gamma=50 --mirror=True --mirrory=False --nkimg=0
```

## Demo 
After installation, you can run the following command that will use the third experiment: 

```bash
python generate.py --outdir=/content/out/images/ --trunc=0.8 --size=256-256 --seeds=0 --network=$network_path
```

## Tracking with Tensorboard 
Go to the following path: 
```bash
cd /home/{username_lab}/Documents/colab-sg2-ada-pytorch
```

And run the following command to see the dashboard: 

```bash
tensorboard --logdir ./
```

## License
Copyright © 2021, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License.

Attribution to Derrick Schultz. 


# LapStyle

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements. If using computers from the labs, load the module of cuda-11.2, to avoid problems with LapStyle. Erase cache from nvcc if needed. 
```bash
which nvcc
module load cuda-11.2
which nvcc
```

For smoother installation create a conda environment
```bash
conda create -n LapEnv
conda activate LapEnv
```

Install Jupyter Notebook if not already installed
```bash
conda install jupyter notebook
```

Install [PaddlePaddle Framework](https://www.paddlepaddle.org.cn/install/quick)
```bash
pip install paddlepaddle-gpu==2.2.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

Set correct path to avoid any errors, replace {username_lab} with your username
```bash
export LD_LIBRARY_PATH='/home/{username_lab}/.conda/envs/LapEnv/lib'
```

Test if Paddle Paddle was installed successfully (It should take less than 1 minute to run), run a python instance
```bash
python
```

Inside the python instance run the following instructions
```python
import paddle
paddle.utils.run_check()
```

A modified version of the [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/install.md) is already included in this repo 

Just, ensure you are inside the "PaddleGAN" folder if not move 
```bash
cd PaddleGAN
```

Install the PaddleGAN requirements, 
```bash
pip install -v -e . 
pip install -r requirements.txt
```

If the previous commands did not work, run this one
```bash
python setup.py develop
```

The dataset is already included in the repo, go back to "NewLapStyle" folder and into the "dataset" folder.
```bash
cd ..
```
The original dataset can be found [here](https://www.kaggle.com/c/gan-getting-started/data). The dataset split is 70/30.

Unzip both files and three folders should appear
```bash
unzip “*.zip”
```

Go back to PaddleGAN directory
```bash
cd ../PaddleGAN
```

To train the model from scratch run the following commands: (Skip this section to run a pretrained model) Replace {{Previously Generated Folder by Draft}}. 
(1)Train the Draft Network of LapStyle under 128*128 resolution:
```bash
python -u tools/main.py --config-file configs/lapstyle_draft.yaml
```

(2) Train the Revision Network of LapStyle under 256*256 resolution:
```bash
python -u tools/main.py --config-file configs/lapstyle_rev_first.yaml --load 'output_dir/{{Previously Generated Folder by Draft}}/iter_30000_checkpoint.pdparams'
```

(3) You can train the second Revision Network under 512*512 resolution:
```bash
python -u tools/main.py --config-file configs/lapstyle_rev_second.yaml --load 'output_dir/{{Previously Generated Folder by First Rev}}/iter_30000_checkpoint.pdparams'
```

To change the style image go to the configs folder and change the name of the "style_root" property for train and test sections in the 3 config files. 

To run pretrained model download the weights from: 
[One Drive Link ](https://mbzuaiac-my.sharepoint.com/personal/roberto_guillen_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Froberto%5Fguillen%5Fmbzuai%5Fac%5Fae%2FDocuments%2FML701%2DWeights%2DLapStyle)
and put it inside (paste the zip and unzip there). There should be 3 folders with 1 pdparams file each

```bash
cd NewLapStyle/PaddleGAN/output_dir
```

## Set up WandB monitoring
Create a WandB account https://wandb.ai/ 
Follow the terminal instructions 
```bash
pip install wandb
wandb login
```

Now you are ready to run the model!!

## Usage
Change the parameter validate/save_img in the configuration file to true to save the output image. To test the trained model, you can directly test the "lapstyle_rev_second", since it also contains the trained weight of previous stages:
```bash
python tools/main.py --config-file configs/lapstyle_rev_second.yaml --evaluate-only --load 'output_dir/LongerVarSecond iter_30000_checkpoint.pdparams'
```

The image will be outputed in the following folder
```bash
cd NewLapStyle/PaddleGAN/output_dir
```

## Demo 

```bash
python applications/tools/lapstyle.py --content_img_path '../dataset/photo_jpg_train/46e84039a1.jpg' --style_image_path '../dataset/monet_jpg/82991e742a.jpg'
```

You can replace the content and style image:
```bash
python applications/tools/lapstyle.py --content_img_path ${PATH_OF_CONTENT_IMG} --style_image_path ${PATH_OF_STYLE_IMG}
```

## Dependencies 
Below you can find a list of dependencies:
* PaddlePaddle
* PaddleGAN
* tqdm
* PyYAML>=5.1
* scikit-image>=0.14.0
* scipy>=1.1.0
* opencv-python
* imageio==2.9.0
* imageio-ffmpeg
* librosa
* numba==0.53.1
* easydict
* munch
* natsort
* cudatoolkit


## Important files 

#### Lapstyle_model.py
This is the model file of LapStyle, the loss functions are selected here, as well, the generators, how the forward pass and backpropagation is performed. This file was modified to be able to calculate the FID while the model is running. The functions test_iter were added to test the model for the selected metrics (currently only running FID) and to evaluate the networks. One new function per network. 

Found in the following folder:  NewLapStyle/PaddleGAN/ppgan/models/lapstyle_model.py

#### lapstyle_predictor.py
Handles how the model loads the pretrained weights and saving the images in the output file. Last but not least, it handles all the laplacian pyramid functions. 

Found in the following folder: NewLapStyle/PaddleGAN/ppgan/apps/lapstyle_predictor.py

#### Trainer.py
The most important file, where the following things happen:
* Training loop is executed
* Metrics are measured
* Log is printed
* WandB monitoring was added
* FID calculation for LapStyle was added

Found in the following folder: NewLapStyle/PaddleGAN/ppgan/engine/trainer.py

#### Fid.py
Using pretrained InceptionV3 weights trained on ImageNet, this file calculates de FID between the style image and the stylized image.
File was modified to work with LapStyle, as it was originally done for cyclegan.

Found in the following folder: NewLapStyle/PaddleGAN/ppgan/metrics/fid.py

#### Config Files
In these files the iterations, dataset location, batch size, optimizer  images and weights for content and style are selected.

Found in the following folder: NewLapStyle/PaddleGAN/configs


## License
Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved

PaddleGAN is released under the Apache 2.0 license.
