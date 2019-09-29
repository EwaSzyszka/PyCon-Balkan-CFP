## Sign Lanaguage Alphabet Recognition with Python
Welcome to the workshop in which will delve into basics of Convolutional Neural Networks via a case-study of a Sign Lanaguage Alphabet Recognition in Keras. We will first introduce a state-of-art Convolutional Neural Network architecture - [VGG-16](https://arxiv.org/pdf/1409.1556.pdf) - and break it down to its smallest building blocks.  After this workshop you will be familiar with the basic Neural Network terminology and you will build an understanding how they can be utilized. At the end of this document there is a glossary list, which you are free to refer to at any point. For those interested in pursuing further their adventure with Convolutional Neural Nets there is a suggested reading list provided at the end as well. 

***** 


##### WHAT ARE CNN: 
It's a deep learning algorithm, that takes an image as an input, extracts features from it, assignes weights and biases to various aspects of the image to be able at the end to classify the image and assign a label to the input. The following problems can be solved using CNNs:

Image classification:  
![Image classification](https://miro.medium.com/max/3840/1*oB3S5yHHhvougJkPXuc8og.gif) 

Semantic Segmentation:  
![Semantic segmentation](https://miro.medium.com/max/4080/1*wninXztJ90h3ZHtKXCNKFA.jpeg) 

Object Localization:  
![Object detection](https://hackernoon.com/hn-images/1*mGXlIHIjFLa3ZiTOJcQmyw.jpeg) 


##### WHAT IS A CNN ARCHITECTURE: 
There are different ways of reach the same goal. Various CNN Architectures came to exsistence as alternative approaches to similar problem types. Each architecture aims at improving the previous design with respect to factors such as minimization of false positive rate, improving the accuracy of the model, decreasing the computational power required to train the model ect. A CNN has basic building blocks such as Convolutional Layers, Activation Functions, Pooling Layers, Fully Connected Layers and many more which can be re-arranged in multiple ways. The ways in which those blocks are re-arranged are so called the CNN architectures.  

##### VGG-16 ARCHITECTURE: 

There are many CNN Architectures that researchers came up with such as [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [ResNet](https://arxiv.org/pdf/1512.03385.pdf) or [VGG-16](https://arxiv.org/pdf/1409.1556.pdf). The VGG-16 is an architecture that is easy to understand and illustrates well the main building blocks of a CNN network. The Convolutional Neural Networks VGG-16 architecture consists of two main building blocks. The task of the first block colored in yellow is to `extract the high level features` from the images and the second block colored in blue aims at `classifying` which of the given labels can be attributed to a specific image. 

<a href="https://imgbb.com/"><img src="https://i.ibb.co/C2z0jWM/1.jpg" alt="1" border="0"></a>

The first step is to feed input image into the Convolutional Network, which is an RGB email 224 pixels by 224 pixels

<a href="https://imgbb.com/"><img src="https://i.ibb.co/Qvk53WR/2.jpg" alt="2" border="0"></a>

Next, there are two `Convolutional layers`, which perform a convolution operation on the input image. The convolution operation is a `dot product` between the input image and `The kernel` and produces the `Feature space` as a result. `The kernel` is a 3 x 3 matrix, which by "sliding" through the input image detects features of the input such as horizontal or vertical edges. The kernel is said to have a `0-padding`, which is a padding added to the input to ensure that the output after convolution operation has the same dimentions as the input. The convolution has a depth of 64 (meaning that there are 64 different kernels applied, each detecting a different feature). After each Convolution a `ReLU activation function` is applied, which is placed in order to introduce non-linearity to the model. Non-linearity is important, as if we are solving a non-linear problem, the convolution operation is linear and rectification with ReLU is a way to break that linearity further. Without ReLU no matter how many hidden layers we would have added the model would behave like a single layer perceptron. 

<a href="https://imgbb.com/"><img src="https://i.ibb.co/L0YbFJJ/3.jpg" alt="3" border="0"></a>

Animated convolution operation:

<img src="https://media.giphy.com/media/i4NjAwytgIRDW/giphy.gif">

Here is an illustration of how the input image changes after the ReLU function is applied (all black pixels are turned gray)

<img src="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_3.png">

<img src="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_4.png">

The next step is `max pooling`, with a window size 2 x 2, which means that from 2 x 2 region we chose the highest value and we re-write it to produce new layer of smaller dimentions 112 x 112.

<a href="https://imgbb.com/"><img src="https://i.ibb.co/9yCKMFB/4.jpg" alt="4" border="0"></a>

Once again we perform convolution twice with 0-padding added and with the kernel depth 128. 

<a href="https://imgbb.com/"><img src="https://i.ibb.co/K2pXqTc/5.jpg" alt="5" border="0"></a>

We perform Max Pooling and reduce the dimentions to 56 x 56

<a href="https://imgbb.com/"><img src="https://i.ibb.co/YPQLLRh/6.jpg" alt="6" border="0"></a>

Next, three convolutions are performed with the kernel depth of 256

<a href="https://imgbb.com/"><img src="https://i.ibb.co/HqR3yKh/8.jpg" alt="8" border="0"></a>

We perform Max Pooling and reduce the dimentions to 28 x 28

<a href="https://imgbb.com/"><img src="https://i.ibb.co/zPNKLJS/9.jpg" alt="9" border="0"></a>

Next, three convolutions are performed with the kernel depth of 512

<a href="https://imgbb.com/"><img src="https://i.ibb.co/5BG9ddt/10.jpg" alt="10" border="0"></a>

We perform Max Pooling and reduce the dimentions to 14 x 14

<a href="https://imgbb.com/"><img src="https://i.ibb.co/djZW6Gh/11.jpg" alt="11" border="0"></a>

Next, three convolutions are performed with the kernel depth of 512

<a href="https://imgbb.com/"><img src="https://i.ibb.co/d0tMfh0/12.jpg" alt="12" border="0"></a>

Next, we 


### _______   EXTRACTING THE HIGH LEVEL FEATURES  __________

1. Convolutional layers  
  

  
 - INPUT: Image as an array of pixels 
 - OUTPUT : produce the feature maps 
 - WHAT: extract various high level features based on the Kernel used 
 
2. Max pooling layer 
  - INPUT: Convoluted image
  - OUTPUT: Downsized array 
  - WHAT: downsising the layer 
  
3. ReLU activation

 <img src="https://latex.codecogs.com/gif.latex?f(x)=&space;max(0,x)" title="f(x)= max(0,x)" />
 
  - INPUT: Downsized array 
  - OUTPUT: Rectified Feature Map 
  - WHAT: turn all negative numbers to 0 and return the positive values as they were 
  
 
  
 ________USING HIGH LEVEL FEATURES TO CLASIFY THE  INPUT IMAGE ______
 
1. Fully connected layers 
  - INPUT:Rectified and downsized Feature map
  - OUTPUT: 
  - WHAT: 
  
5. Softmax activation function 

<img src="https://latex.codecogs.com/gif.latex?f(x_{i})=&space;\frac{e^{x^{i}}}{\sum&space;e^{x^{i}}}" title="f(x_{i})= \frac{e^{x^{i}}}{\sum e^{x^{i}}}" />

  - INPUT: vector of arbitrary real-valued score
  - OUTPUT: vector of values that add up to 1 
  - WHAT: translates the values into probabilities
  


### Project methodology

__________ DATA PREPARATION AND PRE_PROCESSING __________


1. Get the training data 

  - DATASET: https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset

2. Organize the data in the folder 

  - INPUT: Kagle images of each letter in the Sign Alphabet 
  - OUTPUT: A folder of subfolders for each letter
  - WHAT: Organize the data for training the VGG-16 model

3. Resize the data 

  - INPUT: 
  - OUTPUT: 
  - WHAT: 
  
4. One hot encode the categories of the dat a

  - INPUT: 
  - OUTPUT: 
  - WHAT: 
  
5. Save images as numpy arrays 

  - INPUT: 
  - OUTPUT: 
  - WHAT: 
  
6. Normalizing the data 

  - INPUT: 
  - OUTPUT: 
  - WHAT: 
  
 7. Train-Test split the data
 
  - INPUT: 
  - OUTPUT: 
  - WHAT: 
  
 8. Reshaping the numpy arrays 
 
  - INPUT: 
  - OUTPUT: 
  - WHAT: 
  
 __________ SETTING UP THE VGG-16 ARCHITECTURE __________
 
The VGG16 architecture consists of twelve convolutional layers, some of which are followed by maximum pooling layers and then four fully-connected layers and finally a 1000-way softmax classifier. HIstory and a bit of introduction of VGG-16 and CNN.
 
 1. Setting up the Keras implementation
 
 Here paste a short code sample in Keras
 
 2. Make a test prediction 
 
  - INPUT: 
  - OUTPUT: 
  - WHAT: 
  
  3. Assess the model 
  
  Here give some metrix to how the model performed
  
  4. Save the model 
  
  __________ CAPTURING THE IMAGES FOR SIGN RECOGNITION __________
 
 1. Connect to the camera 
 
 2. Press enter to capture the frame 
 
 3. Press escape when all of the frames of your ineterst are already captured 
 
   __________ SIGN RECOGNITION FROM CAPTURED IMAGES __________
 
 1. Load the model 
 
 2. Use the model to classify the unseen data 

Here paste images of sign recognition screen shots 

__________ MODEL ASSESMENT __________

1. Confusion matrix
2. Accuracy across epochs
3. Model loss

### Glossary 

Kernel/filter =  
Feature Map =   
Stride =   
Zero-padding =   
Whide convolution =   
Narrow convolution =   
ReLU activation function =   
Tahn activation function =   
Spatial pooling =   
Deep Neural Network =   
Hidden Layers =   
Convolutional Layer =   
Pooling Layer =   
Fully connected layer =   
Activation layer =   
Max and min pooling =   
Feature learning =   
Perceptron =   
feed-forward neural network =  
backpropagation =  
Softmax Classification =  
Dot product = filter * pixel wise representation of the input  
Overfitting =   
Multilayer perceptron =   
feedforward network = will only have a single input layer and a single output layer, it can have zero or multiple Hidden Layers.  
Forward propagation =   


### Further Readings
http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
http://cs231n.github.io/
https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5

### About me 
I am a Computer Science student at Minerva Schools at KGI and Electronics Engineering student at AGH. This workshop is based on my [Bachelor's Thesis Proposal](https://ewaszyszka.myportfolio.com/bachelor-thesis-proposal). If you are interested delving further into the topic and exploring it further feel free to reach out (ewa.szyszka@minerva.kgi.edu).


