# coding:utf-8

from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
# import module

#import data
train = pd.read_csv("bj_housing2.csv")
#
train = train[train['Area'] < 12000]
train_X = train['Area'].values.reshape(-1, 1)
train_Y = train['Value'].values.reshape(-1, 1)

n_samples = train_X.shape[0]
learning_rate = 2
# set learning rate
training_epochs = 1000
# set display_step
display_step = 50

# init placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
#define defined learning parameters
W = tf.Variable(np.random.randn(), name="weight", dtype=tf.float32)
b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)

# Build a forward propagation structure
pred = tf.add(tf.multiply(W, X), b)
# cost function
cost = tf.reduce_sum(tf.pow(pred-Y, 2)) / (2 * n_samples)
# Using the gradient descent optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# init
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# run the session
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.3f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # show the plot
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()
