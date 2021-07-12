# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:42:48 2021

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
TensorFlow 2.3.0
Keras is installed as part of TensorFlow 2

Tutorial: https://www.geeksforgeeks.org/python-image-classification-using-keras/
Data: https://drive.google.com/open?id=1dbcWabr3Xrr4JvuG0VxTiweGzHn-YYvW

"""
import sys
import tensorflow as tf

print("Python version is: ",sys.version)
print("TensorFlow version is: ", tf.__version__,"\n")

import tensorflow as tf

#'tf' can not be used from ... import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
  
img_width, img_height = 224, 224

train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples = 400 
nb_validation_samples = 100
epochs = 10
batch_size = 16

# Checking format of Image:
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#create model
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# divide datadset
train_datagen = ImageDataGenerator(
                rescale = 1. / 255,
                 shear_range = 0.2,
                  zoom_range = 0.2,
            horizontal_flip = True)
  
test_datagen = ImageDataGenerator(rescale = 1. / 255)
  
train_generator = train_datagen.flow_from_directory(train_data_dir,
                              target_size =(img_width, img_height),
                     batch_size = batch_size, class_mode ='binary')
  
validation_generator = test_datagen.flow_from_directory(
                                    validation_data_dir,
                   target_size =(img_width, img_height),
          batch_size = batch_size, class_mode ='binary')

# actual traininig
model.fit(train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs, 
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

# save model
model.save_weights('model_saved.h5')

  