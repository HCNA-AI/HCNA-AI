# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.constant(3)                     #define variable 3
b = tf.constant(4)                     #define variable 4

with tf.Session() as sess:           #create session
    print ("相加: %i" % sess.run(a+b))
    print( "相乘: %i" % sess.run(a*b))
