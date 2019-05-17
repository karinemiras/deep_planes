from __future__ import print_function
import tensorflow as tf
import numpy as np

w =  5#200 #400
h =  2#150 #300

image_path = 'dataset/tinynone.png'
image_file = tf.read_file(image_path)
image1 = tf.image.decode_jpeg(image_file, channels=1)
image1 = tf.image.resize_images(image1, (h, w))
#image1 = tf.image.decode_jpeg(image_file, channels=1)

image_path = 'dataset/or.png'
image_file = tf.read_file(image_path)
image2 = tf.image.decode_jpeg(image_file, channels=1)
image2 = tf.image.resize_images(image2, (h, w))

sess = tf.Session()
with sess.as_default():
    tensor1 = image1
    tensor2 = image2

    serial1 = tf.reshape(tensor1, [-1])
    serial2 = tf.reshape(tensor2, [-1])
    print_serial2 =  tf.print(serial2)

    comparison = tf.less_equal(serial2, 127)
    comparison2 = tf.less_equal(serial2, 127)

    zeros = tf.fill([1,tf.size(serial2)], 0)[0]
    uns =tf.fill([1,tf.size(serial2)], 1)[0]
    print_zeros = tf.print(zeros, summarize=1000)
    print_uns = tf.print(uns, summarize=1000)

    normalizers = tf.where(comparison, zeros, uns )
    print_normalizers = tf.print(normalizers, summarize=1000)

    #serial22 = tf.cond(serial2 <=127, lambda: np.repeat(0,len(serial2)), lambda: np.repeat(1,len(serial2)))
    #print_serial2 = tf.print(serial2, summarize=1000)


    print('dasjdnasjdn')
    sess.run(print_serial2)
    print('dasjdnasjdn')
    sess.run(print_zeros)
    print('dasjdnasjdn')
    sess.run(print_uns)
    print('dasjdnasjdn')
    sess.run(print_normalizers)
