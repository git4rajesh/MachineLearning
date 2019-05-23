# Architectural Basics



## How many layers :

Keep convolving till we have an input of size n * n Size where we have more number of Pixels that get convolved uniformly say 9 times ( using 3 * 3 Kernel)

- Deep level of convolution until the input channel size becomes 3 * 3 is not advisable as we get only very less number of pixels(1) that gets convolved 9 times.

  

## MaxPooling :

- Use MaxPooling in the transition block when we  want to reduce the input channel size. This will help us in reducing the number of layers and in turn the number of parameters.It also doubles the Receptive Field.

- Avoid Max pooling when we are close to Prediction layer.

  

## 1 * 1 Convolutions :

- 1  * 1 Convolution kernel is used commonly in the Transition block so as to reduce the number of channels gradually. Its used so that all the features extracted at previous layer will not be lost.
- 1 * 1 convolution kernel is used at times to increase the channel size. But these are corner cases and not advisable.



## 3 * 3 Convolutions :

- 3 * 3 Kernel is the most commonly used kernel as its optimized for any input image size.

- The odd size kernels can symmetrically demarcate the input images into left and right. Without this symmetry, we will have to account for distortions across the layers.

  

## Receptive Field :

- Global receptive Field is a parameter we account for to design the number of layers of convolution in a CNN.

- Its the size of the input image we are able to see at a layer after convolution.

  

## SoftMax:

- Softmax is a kind of activation function but not in true sense.

- Its a probability like function which uses exponents. This helps in distancing the output values and used mainly in Classification kind of problems.

- May not be suited for other problem types.

  

## Learning Rate:

- Learning rate alpha is a key parameter that helps in Gradient Descent.

- An optimal learning rate helps in faster convergence during Gradient Descent.

  

## Kernels and how do we decide the number of kernels:

- Kernels are feature extractors and we generally opt for 3 * 3 kernels.

- In a transition block , we do opt for 1 * 1 kernel so as to reduce the channel size.

- Towards the last few layers in convolution we opt for higher sized kernels like 7 * 7 or 9 * 9 based on where want to stop convolution .

  

## Batch Normalization:

- Normalizes the images in a batch of a Epoch so as to minimize the covariance shift

- To avoid overfitting issue we go for Batch Normalization.

  

## Image Normalization:

- The size of an image is normalized 

  

## Position of MaxPooling:

- MaxPooling is usually done in transition block 

- We avoid maxpooling when we are close to prediction layer.

  

## Concept of Transition Layers:

- In a CNN, we want to have layers that help in transition basically to reduce the channel size without loosing information.
- So we use a 1 * 1 kernel to reduce the channel size.
- Its usually preceded with a maxpooling layer.



## Position of Transition Layer:

- Transition layers occur after we have found a. Edges and Gradients b. Textures c. Patterns d. Parts Of Objects e. Scenes

  

## Number of Epochs and when to increase them:

- ## `Number of Epochs = Total Data / Batch Size`

- We increase the number of Epochs when we see a gradual increase in test accuracy.

  

## DropOut:

- Used to minimize overfitting.

- Randomly selected neurons get masked or dropped out.

- This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

  

## When do we introduce DropOut, or when do we know we have some overfitting:

- We introduce drop out in a model when we see our training accuracy increasing whereas the test accuracy lags behind it by quite a distance. Such a model is supposed to be Overfitting.

- Dropout used to minimize overfitting ie. reduces the gap between Test Accuracy  and Train Accuracy.

  

## The distance of MaxPooling from Prediction:

- We avoid maxpooling as we come closer to Prediction.

- Reason being we dont want to loose any extracted feature when we are close to prediction.

- We generally place the maxpooling layer in the transition block.

  

## The distance of Batch Normalization from Prediction:

- Batch normalization is a technique to standardize the inputs to a network, applied to ether the activations of a prior layer or inputs directly.

- We avoid maxpooling as we come closer to Prediction as we dont want to tamper with the inputs when we are closer to prediction.

  

## When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)

- Keep convolving till we have an input of size n * n Size where we have more number of Pixels that get convolved uniformly say 9 times ( using 3 * 3 Kernel)
- Deep level of convolution until the input channel size becomes 3 * 3 is not advisable as we get only very less number of pixels(1) that gets convolved 9 times



## How do we know our network is not going well, comparatively, very early

- Check the first four training accuracy values in our epoch runs and then compare it with our earlier best model initial accuracy values.

- If the current model has low values then we are sure that our network is not going well, comparatively, very early.

  

## Batch Size, and effects of batch size:

- With BatchSize for a fixed learning rate, the validation accuracy increase initially.
- After a threshhold increase in BatchSize, the validation accuracy starts to drop.
- Batch Size can be increased so as to optimally use the GPU available.This helps in reducing the computational time.

When to add validation checks

LR schedule and concept behind it

Adam vs SGD

