# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:55:45 2024

@author: antona

This script changes the size of an image file.
For going from lower to higher resolution it does use interpolation.

"""

import tensorflow as tf
import numpy as np
import cv2
import os

#change both source and dest
#source_folder = "C:/Work/PythonCode/ML_examples/EEG/moabb.bi2013a/data/rp_m_5_tau_30_f1_1_f2_24_el_16_nsub_10_per_20_nepo_800_set_BNCI2014009_as_image/label_0"
source_folder = "C:/Work/PythonCode/ML_examples/EEG/moabb.bi2013a/data/rp_m_5_tau_30_f1_1_f2_24_el_16_nsub_10_per_20_nepo_800_set_BNCI2014009_as_image/label_1"

#destination_folder = "C:/Work/PythonCode/ML_examples/EEG/moabb.bi2013a/data/resized/label_0"
destination_folder = "C:/Work/PythonCode/ML_examples/EEG/moabb.bi2013a/data/resized/label_1"

import os
files = os.listdir(source_folder)   

for f in files:
    
    #you can add file filtering here
    
    print(f)
 
    path_image = os.path.join(source_folder, f)
    image_open = open(path_image, 'rb')
    read_image = image_open.read()
    
    image_decode = tf.image.decode_jpeg(read_image)
    #print("This is the size of the Sample image:",image_decode.shape, "\n")
    
    resize_image = tf.image.resize(image_decode, [224, 224]).numpy()
    
    #print("This is the Shape of resized image",resize_image.shape)
    
    cv2.imwrite(os.path.join(destination_folder , f), resize_image)
    
print("Done.")


