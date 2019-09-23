## Sign Lanaguage Alphabet Recognition with Python
Welcome to the workshop in which will delve into basics of Convolutional Neural Networks via a case-study of a Sign Lanaguage Alphabet Recognition in Keras. We will first introduce a state-of-art Convolutional Neural Network architecture - [VGG-16](https://arxiv.org/pdf/1409.1556.pdf) - and break it down to its smallest building blocks.  After this workshop you will be familiar with the basic Neural Network terminology and you will build an understanding how they can be utilized. At the end of this document there is a glossary list, which you are free to refer to at any point. For those interested in pursuing further their adventure with Convolutional Neural Nets there is a suggested reading list provided at the end as well.

### What are Convolutional Neural Networks: VGG-16 case study 
<img src='https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png' width=80% height=80% >


The Convolutional Neural Networks architecture can be broken down into two main building blocks. The task of the first block is to `extract the high level features` from the images and the second block aims at `classifying` which of the given labels can be attributed to a specific image. Let's take a deeper dive:

### _______   EXTRACTING THE HIGH LEVEL FEATURES  __________

1. Convolutional layers  
  
  <img src="https://media.giphy.com/media/i4NjAwytgIRDW/giphy.gif">
  
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


