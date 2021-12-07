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


# CycleGAN

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
```bash
import paddle
paddle.utils.run_check()
```

A modified version of thePaddleGAN Framework is already included in the repo [PaddlePaddle Framework](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/install.md)

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