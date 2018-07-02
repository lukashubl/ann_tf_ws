import tensorflow as tf
import cv2
import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage
import time

sess=tf.Session()

cap = cv2.VideoCapture(0)

#LOADING THE MNIST DATA FOR LABELS
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#First let's load meta graph and restore weights
#LOADING THE MODEL
saver = tf.train.import_meta_graph('mnist_workshop_model-1.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))




################################################################################
#########################get the needed tensors ################################
################################################################################
graph = tf.get_default_graph()
y = graph.get_tensor_by_name("model:0")
x = graph.get_tensor_by_name("InputData:0")
y_ = graph.get_tensor_by_name("LabelData:0")
pred = tf.argmax(y,1)
################################################################################
################################################################################
################################################################################



ctr = 1

#showing video and doing live prediction
while ctr==1:
    time.sleep(0)
    ret, image_np = cap.read()
    image_np = image_np[100:660, 100:660]
    image_np = cv2.resize(image_np, (28, 28))

    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    #VERSUCHE DIE GRAUSTUFEN MÖGLICHST AUSZURECHNEN DAMIT KONTUREN SICHTBARER WERDEN
    for i in range(0,28):
        for a in range(0,28):
            #minvalue abziehen um an licht anzupassen
            #  image_np[i,a] = image_np[i,a] + maxValue
            if image_np[i,a] > 50:
                image_np[i,a] = 255

    #resizing the image and converting rgb into grayscale
    #cover = resizeimage.resize_cover(image_np, [28, 28])
    newarr = np.array(image_np)
    newarr = abs((newarr / 255) - 1)
    newshape = newarr.shape

    # make a 1-dimensional view of arr
    new_flat_arr = newarr.ravel()

    # convert it to a matrix
    newvector = np.matrix(new_flat_arr)
    erg = sess.run(pred, feed_dict={x: newvector, y_: mnist.test.labels})
    print(erg)

    #ergebnisse zeigen im realtime window
    font = cv2.FONT_HERSHEY_SIMPLEX
    pts = np.array([[120,100],[420,100],[420,420],[120,420]], np.int32)
    pts = pts.reshape((-1,1,2))
    cross = np.array([[300,0],[300,560],[0,560],[0,280],[560,280],[560,0]], np.int32)

    image_np = cv2.resize(image_np, (560,560))

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale = 1
    fontColor = (0,0,0)
    lineType = 2

    cv2.putText(image_np, str(erg[0]),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    cv2.polylines(image_np,[cross],True,(0,0,0))

    cv2.imshow('object detection', image_np)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
