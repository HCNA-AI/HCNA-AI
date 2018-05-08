#coding:utf-8

import tensorflow as tf

filename_queue = tf.train.string_input_producer(["D:/Project/HCNA-AI/tf_read.csv"])   #导入数据
reader = tf.TextLineReader()
# 出列数据（这是一个Tensor定义）
key, value = reader.read(filename_queue)

record_defaults = [[1.], [1.], [1.], [1.]]
col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)

features = tf.stack([col1, col2, col3])
#初始化op
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

with tf.Session() as sess:
#启动队列
    sess.run(init_op)
    sess.run(local_init_op)

    # 填充文件名队列，启动计算图中所有的队列线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
#主线程
    try:
        for i in range(30):
            example, label = sess.run([features, col4])
            print(example)
            # print(label)
    except tf.errors.OutOfRangeError:
        print 'Done !!!'    #处理异常

    finally:       # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        coord.join(threads)
