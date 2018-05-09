# -*- coding: utf-8 -*-
import tensorflow as tf

sess = tf.InteractiveSession()


#tf.random_normal(shape,
#                 mean=0.0,
#                 stddev=1.0,
#                 dtype=dtypes.float32,
#                 seed=None,
#                 name=None)


w1 = tf.Variable(tf.random_normal([2,3],mean=1.0, stddev=1.0))
w2 = tf.Variable(tf.random_normal([3,1],mean=1.0, stddev=1.0))

#Define a two-dimensional constant matrix
x = tf.constant([[0.7, 0.9]])

#init variable
tf.global_variables_initializer().run()


#Matrix Multiplication
a = tf.matmul(x ,w1)
#Matrix Multiplication   y=a*w2
y = tf.matmul(a, w2)
#print the result
print(y.eval())
