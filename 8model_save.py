# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

#Generate simulation data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，added noise
print(train_X.shape)
print(train_Y.shape)
#reset graph
tf.reset_default_graph()

# create model
# placeholder
X = tf.placeholder("float")
Y = tf.placeholder("float")
# Model parameters
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
# Forward structure
z = tf.multiply(X, W)+ b

#Reverse optimization

cost =tf.reduce_mean( tf.square(Y - z))
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

# Initialize variables and operations
init = tf.global_variables_initializer()
#parameter settings
training_epochs = 10000
display_step = 2
saver = tf.train.Saver() # saver
savedir = "log/"
# run the session
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        #display the train
        if epoch % display_step >= 0:
            loss = sess.run (cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
                
