import tensorflow as tf
import os
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.contrib.tensorboard.plugins import projector

################################################################################
###### 1. automatically loading mnist data #####################################
################################################################################
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("image_data", one_hot=True)
mnistTwo = input_data.read_data_sets("image_data", one_hot=False)
images = tf.Variable(mnist.test.images, name='images')

#testing if the data could be fetched by printing the first image
testImage = mnist.test.images[240].reshape(28,28)
plt.imshow(testImage, cmap = matplotlib.cm.binary, interpolation = "nearest")
print(mnist.test.labels[240])
plt.show()
################################################################################
