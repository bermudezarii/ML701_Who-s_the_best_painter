ML701_Who's_the_best_painter

# Who is the best painter? 

Project of the course ML701 at MBZUAI. Our main motivation was to train a model using AI to learn the style of Claude Monet. 

Checking the techniques, we choose to explore state of the art and find a way to compare what had better results. 
We used as base two repositories:
- StyleGAN: [https://github.com/dvschultz/stylegan2-ada-pytorch](https://github.com/dvschultz/stylegan2-ada-pytorch)
- LapStyle: [https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/lap_style.md](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/lap_style.md)


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements (find requirements.txt). If using computers from the labs, load the module of cuda-11.1, to avoid problems with StyleGAN. Erase cache from nvcc if needed. 

```bash
pip install -r requirements.txt
```

## Usage

Open the Jupyter Notebooks provided for each style. If you need to run without notebooks, after being in the directory ('colab-sg2-ada-pytorch/stylegan2-ada-pytorch').  This is the configuration of how we started our 3rd experiment: 
```python
python train.py --nkimg=0 --snap=1 --gpus=1 --cfg='24gb-gpu' -- metrics=fid50k_full --outdir=./metrics_ml --data='/home/ariana.venegas/Documents/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/datasets/Monet_folder.zip' --resume='/home/ariana.venegas/Documents/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/pretrained/wikiart.pkl' --augpipe='bg' --initstrength=0 --gamma=50 --mirror=True --mirrory=False --nkimg=0
```

put how to use tensorboard. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)