import tensorflow as tf
import os
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.contrib.tensorboard.plugins import projector


################################################################################
############### parameter ######################################################
################################################################################
learning_rate = 0.01
training_epochs = 30
batch_size = 100
logs_path = 'tf_logs'

################################################################################
###### 1. automatically loading mnist data #####################################
################################################################################
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("image_data", one_hot=True)
mnistTwo = input_data.read_data_sets("image_data", one_hot=False)
images = tf.Variable(mnist.test.images, name='images')

#testing if the data could be fetched by printing the first image
#testImage = mnist.test.images[240].reshape(28,28)
#plt.imshow(testImage, cmap = matplotlib.cm.binary, interpolation = "nearest")
#print(mnist.test.labels[240])
#plt.show()
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
with tf.name_scope('Loss'):
    #cross entropy als größe die es zu minimieren gilt
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), axis = 1))
with tf.name_scope('Optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(pred,1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

#initialisieren aller variablen
init = tf.global_variables_initializer()



#erstellen der scalar plots in tensorboard
#create summary to monitor cost tensor
tf.summary.scalar("loss", cost)
#create a summery to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
#merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

#erstellen der metadaten für die tensorboard ansicht
metadata = os.path.join(logs_path, 'metadata.tsv')
with open(metadata, 'w') as metadata_file:
    for row in mnistTwo.test.labels:
        metadata_file.write('%d\n' % row)



#starten der session
with tf.Session() as sess:
    sess.run(init)
    #writing summary logs for TensorBoard
    summary_writer = tf.summary.FileWriter(logs_path, graph= tf.get_default_graph())

    saver = tf.train.Saver([images])
    saver.save(sess, os.path.join(logs_path, 'images.ckpt'))
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.sprite.image_path = os.path.join(logs_path, 'sprite.png')
    embedding.metadata_path = metadata
    embedding.sprite.single_image_dim.extend([28,28])

    projector.visualize_embeddings(tf.summary.FileWriter(logs_path), config)

    #training
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples/batch_size)
        #loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            opt, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_xs, y: batch_ys})
            #write logs
            summary_writer.add_summary(summary, epoch*total_batch + i)
            avg_cost += c / total_batch

        if (epoch+1) % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    summary_writer.close()

    print("Accuracy: ", acc.eval({x: mnist.test.images, y: mnist.test.labels}))
    print('finished')
