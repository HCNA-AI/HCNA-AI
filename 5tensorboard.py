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
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，Increase noise
#Graphic display
# plt.plot(train_X, train_Y, 'ro', label='Original data')
# plt.legend()
plt.show()


tf.reset_default_graph()

# Create model
# placeholder
X = tf.placeholder("float")
Y = tf.placeholder("float")
# Model parameters
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# Forward structure
z = tf.multiply(X, W)+ b
tf.summary.histogram('z',z)#Display predicted values ​​as histograms
#Reverse optimization
cost =tf.reduce_mean( tf.square(Y - z))
tf.summary.scalar('loss_function', cost)#show the cost
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

#Initialize variables
init = tf.global_variables_initializer()
#parameter settings
training_epochs = 20
display_step = 2

# run the session
with tf.Session() as sess:
    sess.run(init)
    
    merged_summary_op = tf.summary.merge_all()#merge all the summary
    #create summary_writer
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)

    #Enter data into the model
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            
            #生成summary
            summary_str = sess.run(merged_summary_op,feed_dict={X: x, Y: y});
            summary_writer.add_summary(summary_str, epoch);#write the summary 

        #display the train
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print (" Finished!")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    #print ("cost:",cost.eval({X: train_X, Y: train_Y}))

    #display the plot
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()
    
    # plotdata["avgloss"] = moving_average(plotdata["loss"])
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    # plt.xlabel('Minibatch number')
    # plt.ylabel('Loss')
    # plt.title('Minibatch run vs. Training loss')
     
    plt.show()

    print ("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))
    
