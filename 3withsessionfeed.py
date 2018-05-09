# -*- coding: utf-8 -*-
"""

"""
import tensorflow as tf
a = tf.placeholder(tf.int16)     define palceholder
b = tf.placeholder(tf.int16)     
add = tf.add(a, b)
mul = tf.multiply(a, b)                      #a*b
with tf.Session() as sess:
    #Calculate the value
    print ("相加: %i" % sess.run(add, feed_dict={a: 3, b: 4}))
    print ("相乘: %i" % sess.run(mul, feed_dict={a: 3, b: 4}))
mul=tf.multiply(a,b)
with tf.Session() as sess:
    #run the session
    print("相加： %i"%sess.run(add,feed_dict={a:3,b:4}))
    #add the op
    print("相乘：%i"%sess.run(mul,feed_dict={a:3,b:4}))
