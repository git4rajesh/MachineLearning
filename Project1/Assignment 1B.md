1. What are Channels and Kernels (according to EVA)?

Channels:
=========
Channels are the primary features of an image.
How many numbers are used to specify the color of each pixel is the number of channels each pixel has.
Incase of a colored image, we have 3 channels such as Red, Green and Blue.

A monochrome image that has one number per pixel has one channel.

Feature Maps and Channels in CNN are one and the same. Each channel after the first layer of a CNN is a feature map. 
Before the first layer of CNN, RGB images have 3 channels (red, green & blue channels).

For example in this code:

Conv2d(3, 32, kernel_size=3, stride=2, padding=2)

The 3 is the number of input channels (R, G, B). 
That 32 is the number of channels (i.e. feature maps) in the output of the first convolution operation. 
So, the first conv layer takes a color (RGB) image as input, applies 3x3 kernel with a stride 2, and outputs 32 feature maps


Kernels:
=========


