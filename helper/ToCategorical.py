import tensorflow as tf

a = tf.keras.utils.to_categorical([0, 1, 0, 3, 3], num_classes=4)
#a = tf.constant(a, shape=[4, 4])
print(a)




