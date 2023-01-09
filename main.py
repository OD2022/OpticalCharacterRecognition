# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation,Flatten, Conv2D, MaxPooling2D




###Loading digits data
def load_digits_data():
  ((digits_training_data, digits_training_labels), (digits_testing_data, digits_testing_labels)) = mnist.load_data() 
  
  
  ##Casting to appropriate data type
  digits_training_data = digits_training_data.astype('float')
  digits_testing_data = digits_testing_data.astype('float')
  digits_training_labels = digits_training_labels.astype('float')
  digits_testing_labels = digits_testing_labels.astype('float')
  
  ##Normalizing data 
  keras.utils.normalize(digits_training_data, axis=1)
  keras.utils.normalize(digits_testing_data, axis=1)
  

  ###Resizing to four dimensions
  IMAGE_SIZE = 28
  digits_training_datar = np.array(digits_training_data).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
  digits_testing_datar = np.array(digits_testing_data).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


  return digits_training_datar, digits_training_labels, digits_testing_datar, digits_testing_labels



##Loading character_dataset
def load_character_dataset():
    data = []
    labels = []
    
    IMAGE_SIZE = 28
    
    for row in open('azdataset.csv'): 
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        data.append(image)
        labels.append(label)


    ##Normalizing the character data
    keras.utils.normalize(data, axis=1)
    
    ###Reshaping the character data
    data_reshaped = np.array(data).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    
    ##Changing data type of the lables to int & data to float
    labels = np.array(labels, dtype="int")
    data_reshaped = data_reshaped.astype('float')
    
    
    char_train_data, char_test_data, char_train_labels, char_test_labels = train_test_split(data_reshaped, labels, test_size = 0.2, random_state=1)
    return char_train_data, char_test_data, char_train_labels, char_test_labels



##Combining Datasets
def combine_datasets():
    digits_training_data, digits_training_labels, digits_testing_data, digits_testing_labels = load_digits_data()
    char_train_data, char_test_data, char_train_labels, char_test_labels = load_character_dataset()
    char_train_labels +=10
    char_test_labels +=10
    
    
    combined_training_data = np.concatenate((digits_training_data, char_train_data), axis=0)
    combined_training_labels = np.concatenate((digits_training_labels, char_train_labels), axis=0)
    combined_testing_data = np.concatenate((digits_testing_data, char_test_data), axis=0)
    combined_testing_labels = np.concatenate((digits_testing_labels, char_test_labels), axis=0)
    return combined_training_data, combined_testing_data, combined_training_labels, combined_testing_labels

    

###Creating a deep neural network
def createNeuralNetwork():
    
  training_data, testing_data, training_labels, testing_labels = combine_datasets()


  ###FirstConvolutional Layer
  engine = Sequential()
  engine.add(Conv2D(128, (3,3), input_shape=training_data.shape[1:]))
  engine.add(Activation('relu'))
  engine.add(MaxPooling2D(pool_size=(2,2)))

  ###SecondConvolutional Layer
  engine = Sequential()
  engine.add(Conv2D(128, (3,3)))
  engine.add(Activation('relu'))
  engine.add(MaxPooling2D(pool_size=(2,2)))

  ###ThirdConvoluntionalLayer
  engine = Sequential()
  engine.add(Conv2D(128, (3,3)))
  engine.add(Activation('relu'))
  engine.add(MaxPooling2D(pool_size=(2,2)))

    
  ###FourthConvoluntionalLayer
  engine = Sequential()
  engine.add(Conv2D(128, (3,3)))
  engine.add(Activation('relu'))
  engine.add(MaxPooling2D(pool_size=(2,2)))


  ##Fully connected Layer1
  engine.add(Flatten())
  engine.add(Dense(128))
  engine.add(Activation('relu'))

  ##Fully connected Layer2
  engine.add(Dense(64))
  engine.add(Activation('relu'))

  ##Output Layer
  engine.add(Dense(36))
  engine.add(Activation('softmax'))
  engine.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
  print(engine.summary)

  engine.fit(training_data, training_labels, epochs=50, validation_split=0.3)
  test_loss, test_accuracy = engine.evaluate(testing_data, testing_labels)
  engine.save("model2", save_format="h5")
  
  print(test_loss)
  print(test_accuracy)
  prediction = engine.predict(testing_data)
  print(prediction)


createNeuralNetwork()

