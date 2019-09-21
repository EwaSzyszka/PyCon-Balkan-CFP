## Sign Lanaguage Alphabet Recognition with Python
Welcome to the workshop in which will delve into basics of Sign Lanaguage Recognition in Keras. 
We will first introduce Convolutional Neural Networks and see what are the basic building blocks of them, next we will look at the Sign Language Alphabet Recognition project as an example of Neural Network Implementation. After this workshop you will be familiar with the basic Neural Network terminology and you will build an understanding of a sample Neural Network Architecture, namely the [VGG-16 architecture](https://arxiv.org/pdf/1409.1556.pdf). At the end of this document there is a glossary list, which you are free to refer to at any point. For those interested in pursuing further their adventure with Convolutional Neural Nets there is a short suggested literature list provided at the end as well.

### What are Convolutional Neural Networks

_______EXTRACTING THE HIGH LEVEL FEATURES __________

1. Convolutional layers 
 - INPUT: Image as an array of pixels 
 - OUTPUT : produce the feature maps 
 - WHAT: extract various high level features based on the Kernel used 
 
2. Max pooling layer 
  - INPUT: Convoluted image
  - OUTPUT: Downsized array 
  - WHAT: downsising the layer 
  
3. ReLU activation
  - INPUT: Downsized array 
  - OUTPUT: Rectified Feature Map 
  - WHAT: turn all negative numbers to 0 and return the positive values as they were 
  
  
 ________USING HIGH LEVEL FEATURES TO CLASIFY THE  INPUT IMAGE ______
 
1. Fully connected layers 
  - INPUT:Rectified and downsized Feature map
  - OUTPUT: 
  - WHAT: 
  
5. Softmax activation function 
  - ensuring that the output of the fully connected layer adds up to 1 
  - Softmax function takes the logits vector and turns it into a vector which nicely sums up to 1 and is normalized

Softmax function:


<img src="https://latex.codecogs.com/gif.latex?f(y_{i})=&space;\frac{e^{y^{i}}}{\sum&space;e^{y^{i}}}" title="f(y_{i})= \frac{e^{y^{i}}}{\sum e^{y^{i}}}" />

### Methodology


1. Get the training data 

Dataset: https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset

2. Pre-process the data 
2. Train a Keras model on a VGG-16 Architecture for the sign language alphabet 

https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5 

3. Save the model as .h5 file
4. Capture the images from camera 
5. Test the captured images with the model 

### Outcomes


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
feed-forward neural network
backpropagation 
Softmax Classification 
Dot product = filter * pixel wise representation of the input
Overfitting = 
Multilayer perceptron = 
feedforward network = will only have a single input layer and a single output layer, it can have zero or multiple Hidden Layers.


### Further Readings


### About me 
I am a Computer Science student at Minerva Schools at KGI and Electronics Engineering student at AGH. This workshop is based on my [Bachelor's Thesis Proposal](https://ewaszyszka.myportfolio.com/bachelor-thesis-proposal). If you are interested delving further into the topic and exploring it further feel free to reach out (ewa.szyszka@minerva.kgi.edu).


