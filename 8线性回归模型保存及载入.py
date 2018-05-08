# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
print(train_X.shape)
print(train_Y.shape)
#重置图
tf.reset_default_graph()

# 创建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
# 前向结构
z = tf.multiply(X, W)+ b

#反向优化
cost =tf.reduce_mean( tf.square(Y - z))
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

# 初始化变量等操作
init = tf.global_variables_initializer()
#参数设置
training_epochs = 100000
display_step = 2
saver = tf.train.Saver() # 生成saver
savedir = "log/"
# 启动session
with tf.Session() as sess:
    sess.run(init)

    # 添加Sess中的训练代码
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        #显示训练中的详细信息
        if epoch % display_step >= 0:
            loss = sess.run (cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
                
