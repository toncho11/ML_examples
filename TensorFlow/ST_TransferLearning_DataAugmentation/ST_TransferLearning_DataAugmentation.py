# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:06:58 2022

@author: antona

The process of training a deep learning model via transfer learning on a 
custom dataset as well as the quantization of this model in order to 
use it on a STM32 thanks to STM32Cube.AI.

pip install imgaug --upgrade
pip install tensorflow_datasets

imgaug is a library for image augmentation in machine learning experiments.
It uses a dataset from Keras called "tf_flowers".
Model is saved to the file "model_quant.tflite".

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.metrics as sk_metrics
import seaborn as sns
import pathlib

from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import datasets
from random import randint
import tensorflow_datasets as tfds

#Model selection
#You can choose between a MobileNet V1 and V2.

MODEL_VERSION = 'V2' #@param ["V1", "V2"]
IMG_SIZE = [128, 128]
IMG_SIZE_TUPLE = (IMG_SIZE[0], IMG_SIZE[1])
BATCH_SIZE = 16

#Dataset selection
DATASET_NAME = 'tf_flowers'

#Dataset import
(train_dataset, validation_dataset, test_set), dataset_info = tfds.load(
            DATASET_NAME,
            split=['train[:70%]', 'train[70%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
            shuffle_files= True
        )

class_number = dataset_info.features['label'].num_classes
class_names = dataset_info.features['label'].names

# Cast data images to 8 bits and use one hot for labels 
def prepare_data(image, label):
  image = tf.cast(tf.image.resize(image, IMG_SIZE_TUPLE), tf.uint8)
  label = tf.one_hot(label, class_number)
  return image, label

train_dataset = train_dataset.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
validation_dataset = validation_dataset.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
test_set = test_set.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

#Data visualization
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[tf.argmax(labels[i])]) 
        plt.axis('off')
        
#Augment the data for training set
augmenter = iaa.Sequential([
    iaa.Sometimes(0.3, iaa.MotionBlur((3, 4)), name='SometimesMotionBlur'),
    iaa.Sometimes(0.3, iaa.GaussianBlur((0.0, 0.75)), name='SometimesGaussianBlur'),
    iaa.GammaContrast((0.7, 1.5)),
    iaa.MultiplySaturation((0.9, 1.5)),
    iaa.MultiplyAndAddToBrightness(),
    iaa.Fliplr(p=0.5),
    iaa.Affine(scale=(1, 1.3),
              translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
              rotate=(-25, 25) )
])
def augmentation_function(images, labels):
    img_dtype = images.dtype
    img_shape = tf.shape(images)
    images = tf.numpy_function(augmenter.augment_images,
                                [images],
                                img_dtype)
    images = tf.reshape(images, shape=img_shape)
    return images, labels

#Apply the data augmentation to the whole database.
train_dataset = train_dataset.map(augmentation_function, num_parallel_calls=tf.data.AUTOTUNE)

#Normalize the data before training
IMG_SHAPE = IMG_SIZE_TUPLE + (3,)
if MODEL_VERSION == "V1":
    normalization = tf.keras.applications.mobilenet.preprocess_input
    base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                                  alpha=0.25,
                                                  include_top=False,
                                                  weights="imagenet")
elif MODEL_VERSION == "V2":
    normalization = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                                                alpha=0.35,
                                                                include_top=False,
                                                                weights='imagenet')
else:
    print('Bad model_version argument, are only accepted : "V1", "V2"')

train_dataset = train_dataset.map(lambda img, label: (normalization(tf.cast(img, tf.float32)), label), num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(lambda img, label: (normalization(tf.cast(img, tf.float32)), label), num_parallel_calls=tf.data.AUTOTUNE)
test_set = test_set.map(lambda img, label: (normalization(tf.cast(img, tf.float32)), label), num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#Model instantiation
base_model.trainable = False
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(class_number, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

#Train the model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(train_dataset,
                        epochs=7, #default 100
                        validation_data=validation_dataset, callbacks=callback)

print("Total Validation Accuracy: ", history.history['val_accuracy'][len(history.history['val_accuracy'])-1] )

#Display Training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

#Test the model
test_set = test_set.cache()
print("Evaluate on test data")
results = model.evaluate(test_set)
print(f"test loss: {results[0]}, test acc: {results[1]}")
predictions = model.predict(test_set)
predictions = tf.argmax(predictions, axis=1)

true_categories = tf.argmax( np.concatenate([y for x, y in test_set], axis=0), axis=1)

confusion = sk_metrics.confusion_matrix(true_categories, predictions)
confusion_normalized = [element/sum(row) for element, row in zip([row for row in confusion], confusion)]
axis_labels = list(class_names)
plt.figure(figsize=(10, 10))
ax = sns.heatmap(confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels, cmap='Blues',
                  annot=True,
                  fmt='.2f', square=True)
plt.title("Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label");

#Missclassified samples
misclassified_images = []
misclassified_category = []
misclassified_prediction = []
for (image, true_category), prediction in zip(test_set.unbatch(), predictions):
  true_category = tf.argmax(true_category)
  if true_category != prediction:
    misclassified_images.append(image.numpy())
    misclassified_category.append(true_category)
    misclassified_prediction.append(prediction)
print('Number of misclassified images : ', len(misclassified_images))

#Quantize and export
def representative_data_gen():
  for x, y in validation_dataset.take(100):
    yield [x]

# Needed for quantization in case of unfrozen MobileNet model
model.trainable = False
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

import pathlib
tflite_model_quant = converter.convert()
tflite_model_quant_file = pathlib.Path("./model_quant.tflite")
tflite_model_quant_file.write_bytes(tflite_model_quant)

#Testing the quantized model
#Performance can diminish because of lack of precision
# Initialize the interpreter
interpreter = tf.lite.Interpreter(model_path=str("./model_quant.tflite"))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

quant_correct_predictions = 0
quant_incorrect_predictions = 0

for (image, true_category) in test_set.unbatch():
  test_image = image
  test_label = tf.argmax(true_category)

  # Check if the input type is quantized, then rescale input data to uint8
  if input_details['dtype'] == np.uint8:
    input_scale, input_zero_point = input_details["quantization"]
    test_image = test_image / input_scale + input_zero_point

  test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
  interpreter.set_tensor(input_details["index"], test_image)
  interpreter.invoke()
  output = interpreter.get_tensor(output_details["index"])[0]
  if(output.argmax() == test_label):
    quant_correct_predictions += 1
  else:
    quant_incorrect_predictions += 1
    
print(f'Accuracy : {quant_correct_predictions / (quant_correct_predictions + quant_incorrect_predictions)}')
