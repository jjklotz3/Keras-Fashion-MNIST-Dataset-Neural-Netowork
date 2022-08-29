#1st Step: import libraries needed 
import tensorflow as tf 
from tensorflow import keras                        #imports keras libraries
from keras import layers                            #imports keras layers libraries
from keras.datasets.fashion_mnist import load_data
import matplotlib.pyplot as plt
import numpy as np 

#2nd step: Import Data from Keras and Load into a Variable 
fashion_data = keras.datasets.fashion_mnist               #Dataset that has This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, 
                                                          #along with a test set of 10,000 images. 
                                                          #This dataset can be used as a drop-in replacement for MNIST
                                                          #Dataset stored as a variable 

(train_images, train_labels), (test_images, test_labels) = fashion_data.load_data()  #With any machine learning algorithim, we want to split the data into training and validation.
                                                                                     #Usually we want to split 80/20 or 90/10 between train and test. We want to do this in order to ensure our model is correct with NEW DATA

print(train_images.shape, train_labels.shape,test_images.shape, test_labels.shape)

#train_images: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data.
#train_labels: uint8 NumPy array of labels (integers in range 0-9) with shape (60000,) for the training data.
#test_images: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data.
#test_labels: uint8 NumPy array of labels (integers in range 0-9) with shape (10000,) for the test data.

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'] #The model is going to return a number 0-9. THe number corresponds witht the list on the right so we know what article of clothing the model is classifying 

train_images, test_images = train_images/255, test_images/255   #We want to shrink the RGB scale (255) down to just 1 so we divide the arrays by 255
 
#Step 3: Build the Model
model = keras.Sequential([
   keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),     #32 is the number of filters in CNN but the kernel_size (3,3) is pretty common. The input shape is the shape of each image going through the network (height, image, color)
                                                                                          #The color is one in this scenario because we converted the image to grey scale so one color
   keras.layers.MaxPooling2D(pool_size=(2,2)),                                            #The Maxpooling layer is an operation that follows a Conv2D which will reduce the dimensions of the image output
   keras.layers.Dropout(rate=0.25),                                                       #This is the idea behind DROPOUT LAYER. To break up these conspiracies, we randomly drop out some fraction of a 
                                                                                          #layer's input units every step of training, making it much harder for the network to learn those spurious patterns in the training data.
   keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
   keras.layers.MaxPooling2D(pool_size=(2,2)),
   keras.layers.Dropout(rate=0.25),
   keras.layers.Flatten(),                             #Needs to be flattened so it can be passed to an individual neuron as currently each image is a 2D array (28x28)
   keras.layers.Dense(256, activation = 'relu'),       #A dense layer is a layer that is connected to all of the neurons in the previous layer. 128 nuerons
   keras.layers.Dense(10, activation = 'softmax')      #Another fully connected output layer. The softmax activation function assigns a value to all of the possible outputs while all adding up to one.
                                                       #Basically, the network is telling us the probability of the image being a paticular item of clothing.
  
])
 
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy']) #Sets up parameters of the model
 
#The OPTIMIZER is an algorithm that adjusts the weights to minimize the loss. Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems
#without any parameter tuning (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.
#The LOSS function measures the disparity between the the target's true value and the value the model predicts. If your 𝑌𝑖's are integers, use sparse_categorical_crossentropy. Examples for above 3-class classification problem: [1] , [2], [3].
#Classes are one-hot encoded (Examples (for a 3-class classification): [1,0,0] , [0,1,0], [0,0,1]), use categorical_crossentropy
#The METRICS is what we care about so for 'accuracy', how low we can get the loss function
 
#Step 4: Train the model
 
history = model.fit(train_images,train_labels,          #Always store the fit in a variable called history so it can be plotted later
        epochs = 10,
        batch_size = 250,
        validation_split = 0.2)  #Trains the model (X , Y, Epochs).  
 
#EPOCHS are the amount of times the model is going to see the training data. Its going to randomly select and pair images and labels
#and how many times the model will see those same pairs. There is no correct smount of epochs at first but you have to play around and look at the data to see how it differs
#There is a point where the more epochs you use to train, the more inaccurate it becomes
#We specify a BATCH SIZE of 250, which means that during training 250 images at once will be processed.
 
#Step 5: Evalute model accuracy with test variables
test_loss, test_acc = model.evaluate(test_images, test_labels)
#Creating loss and accuracy variables for the test variables to see how accurate our model is. .evaluate(X variable, Y variable )

print("Test Acc:", test_acc)

