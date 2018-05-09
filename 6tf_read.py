#coding:utf-8

import tensorflow as tf

filename_queue = tf.train.string_input_producer(["D:/Project/HCNA-AI/tf_read.csv"])   #read the data
key, value = reader.read(filename_queue)

record_defaults = [[1.], [1.], [1.], [1.]]
col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)

features = tf.stack([col1, col2, col3])
#init the op
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

with tf.Session() as sess:
#run the session
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        for i in range(30):
            example, label = sess.run([features, col4])
            print(example)
            # print(label)
    except tf.errors.OutOfRangeError:
        print('Done !!!')    

    finally:       
        coord.request_stop()
        coord.join(threads)
