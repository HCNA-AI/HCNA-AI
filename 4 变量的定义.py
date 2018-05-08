# -*- coding: utf-8 -*-
import tensorflow as tf

tf.reset_default_graph()    #用于清除默认图形堆栈并重置全局默认图形
    
# var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)      #变量1
# var2 = tf.get_variable("firstvar1",shape=[2],dtype=tf.float32)    #变量2
    
with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)   #定义一个作用域Test1
    
with tf.variable_scope("test2"):
    var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)     #定义一个作用域Test1
        
print ("var1:",var1.name) 
print ("var2:",var2.name)  
