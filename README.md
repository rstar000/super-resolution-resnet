# super-resolution-resnet
This is a program that upscales your images using a ResNet neural network! It is written in Pytorch.
The program is based on Enhanced Deep Super Resolution paper([arXiv](https://arxiv.org/abs/1707.02921)). It uses ResNet architecturee with some tweaks. You can train and modify the parameters in the jupyter notebooks! There is a script to quickly upscale an image of any format.

## Running the script
```
python3 upscale.py my_crappy_image.png awesome_upscaled.png
```

## Architecture and training
To train the network, open the 'edsr.ipynb' notebook. Set the path to your image directory. The images must be high quality and at least 100x100 pixels. You can also change the number of risidual blocks(depth) and number of convolution filters in a block(width) in a network.

## Examples
![Example 1](examples/demo.jpg)
![Example 2](examples/demo2.jpg)

