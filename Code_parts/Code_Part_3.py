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



################################################################################
###### model and variables #####################################
################################################################################
#speichert die images
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
#speichert die dazugehörigen labels
y = tf.placeholder(tf.float32, [None, 10], name = 'LabelData')

#setup the model Parameters
W = tf.Variable(tf.zeros([784, 10]), name='Weights')
b = tf.Variable(tf.zeros([10]), name = 'Bias')

#operationen
with tf.name_scope('Model'):
    #erstellen des models
    pred = tf.nn.softmax(tf.matmul(x,W) + b) #softmax
with tf.name_scope('Loss Function'):
    #cross entropy als größe die es zu minimieren gilt
    #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), axis = 1))
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))
