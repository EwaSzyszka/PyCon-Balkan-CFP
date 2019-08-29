## Sign Lanaguage Recognition with Python
Welcome to the 30 minute Workshop which will delve into basics of Sign Lanaguage Recognition in Python. 


### About me 
I am a Computer Science student at Minerva Schools at KGI and Electronics Engineering student at AGH. This workshop is based on my [Bachelor's Thesis Proposal](https://ewaszyszka.myportfolio.com/bachelor-thesis-proposal). If you are interested delving further into the topic and exploring it further feel free to reach out (ewa.szyszka@minerva.kgi.edu).

### Dataset

https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset

### Methodology

1. Pre-process the data 
2. Train a Keras model on a VGG-16 Architecture for the sign language alphabet 
3. Save the model as .h5 file
``` 
ImportError: `save_model` requires h5py.
```
3. Edit the following code to grab images from the camera
``` 
from keras import load_model
model = load_model(path) # open saved model/weights from .h5 file

def predict_image(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)

    # model.predict() returns an array of probabilities - 
    # np.argmax grabs the index of the highest probability.
    result = gesture_names[np.argmax(pred_array)]
    
    # A bit of magic here - the score is a float, but I wanted to
    # display just 2 digits beyond the decimal point.
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(f'Result: {result}, Score: {score}')
    return result, score
   
```

### Architecture

https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5 
### PowerPoint presentation
