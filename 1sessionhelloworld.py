# -*- coding: utf-8 -*-

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')  #Define constants

sess = tf.Session()                            #Create a session

print (sess.run(hello))                        #Run the session
sess.close()                                    #colse session
