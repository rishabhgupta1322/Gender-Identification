#header files
import io
import pandas as pd
import numpy as np
from google.colab import files
from __future__ import print_function
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras import optimizers
from keras.layers.normalization import BatchNormalization

#To upload file in google collaborator
uploaded = files.upload()

#Checking upload of csv dataset
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
      
#Reading data from csv file and store it into npy file
data = pd.read_csv(io.StringIO(uploaded['wiki5.csv'].decode('utf-8')))
np.save('data.npy',data)

#Loading dataset
data = np.load('data.npy')
y = data[:,0]
x = data[:,2:]

#classifying training and testing dataset
X_train =x[0:2500]
X_test = x[2500:]
y_train = y[0:2500]
y_test = y[2500:]

#image size in form of rows and columns
img_rows, img_cols = 100, 100

#Specifying number of classes
num_classes = 2

#Reshaping dataset into matrix and classifying dataset according to classes
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32')
X_train /= 250
X_test /= 250
# convert class labels to binary class labels
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#CNN training data model
model = Sequential()
model.add(Conv2D(32, (2, 2),padding='valid', kernel_initializer='uniform', input_shape=(1, 100, 100), data_format="channels_first"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation("softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Predict testing dataset and check accuracy
result = model.predict(X_test)
predicted_class = np.argmax(result, axis=1)
true_class = np.argmax(y_test, axis=1)
num_correct = np.sum(predicted_class == true_class) 
accuracy = (float(num_correct)/result.shape[0])*100

#Result
print ("Accuracy on test data is: %0.2f", accuracy)
