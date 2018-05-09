# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.constant(3)                     #define variable 3
b = tf.constant(4)                     #define variable 4

with tf.Session() as sess:           #create session
    print ("add: %i" % sess.run(a+b))
    print( "mul: %i" % sess.run(a*b))
