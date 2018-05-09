# -*- coding: utf-8 -*-
import tensorflow as tf

tf.reset_default_graph()    
# var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)      #define variable 1
# var2 = tf.get_variable("firstvar1",shape=[2],dtype=tf.float32)    #define variable 2
    
with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)   #define variable_scope test1
    
with tf.variable_scope("test2"):
    var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)     #define variable_scope test1
        
print("var1:",var1.name)   #print variable 1
print("var2:",var2.name)   #print variable 2
