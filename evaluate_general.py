from __future__ import print_function
import tensorflow as tf
import os

w =  200 #400
h =  150 #300



original_ = 'Plane_train'
class_ = 'Plane_train_labels'
pred_ = 'result1/test1_t'

path_original = 'dataset/'+original_
path_class = 'dataset/'+class_
path_pred = pred_

images_class = []
images_pred = []

for r, d, f in os.walk(path_original):
    for file in f:
        if '.jpg' in file or '.png' in file:
            if os.path.isfile(path_class+'/'+file.split('.')[0]+'.png'):
                images_class.append(path_class+'/'+file.split('.')[0]+'.png')
            else:
                images_class.append('dataset/none.png')

            images_pred.append(path_pred+'/'+file)

print('>> read')

for i in range(0, 2):#len(images_class)):
    print(images_class[i])
    image_file = tf.read_file(images_class[i])
    image_class = tf.image.decode_png(image_file, channels=1)
    image_class = tf.image.resize_images(image_class, (h, w))

    image_file = tf.read_file(images_pred[i])
    image_pred = tf.image.decode_png(image_file, channels=1)
    image_pred = tf.image.resize_images(image_pred, (h, w))

    serial_class = tf.reshape(image_class, [-1])
    serial_pred = tf.reshape(image_pred, [-1])

    comparison_class = tf.less_equal(serial_class, 127)
    comparison_pred = tf.less_equal(serial_pred, 127)

    zeros = tf.fill([1, tf.size(serial_class)], 0)[0]
    uns = tf.fill([1, tf.size(serial_class)], 1)[0]

    comparison_class = tf.where(comparison_class, zeros, uns)
    comparison_pred = tf.where(comparison_pred, zeros, uns)

    if i == 0:
        all_class = comparison_class
        all_pred = comparison_pred
    else:
        all_class = tf.concat([all_class, comparison_class], 0)
        all_pred = tf.concat([all_pred, comparison_pred], 0)

accuracy = tf.metrics.accuracy(all_class, all_pred)
precision = tf.metrics.precision(all_class, all_pred)
recall = tf.metrics.recall(all_class, all_pred)
auc = tf.metrics.auc(all_class, all_pred)
fscore = tf.contrib.metrics.f1_score(all_class, all_pred)

print('>> calculated')


sess = tf.Session()
global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
sess.run(global_init)
sess.run(local_init)

#sess.run(tf.print(accuracy))
#sess.run(tf.print(precision))
#sess.run(tf.print(recall))

sess.run(tf.print(fscore))

# accuracy = tf.strings.join(['accuracy ',tf.dtypes.as_string(tf.gather_nd(accuracy, [1]))])
# precision = tf.strings.join(['precision ',tf.dtypes.as_string(tf.gather_nd(precision, [1]))])
# recall = tf.strings.join(['recall ',tf.dtypes.as_string(tf.gather_nd(recall, [1]))])
# auc = tf.strings.join(['auc ',tf.dtypes.as_string(tf.gather_nd(auc, [1]))])
# fscore = tf.strings.join(['fscore ',tf.dtypes.as_string(tf.gather_nd(fscore, [1]))])
#
# output = tf.strings.join([accuracy, '\n',
#                           precision, '\n',
#                           recall, '\n',
#                           auc, '\n',
#                           fscore
#                          ])
#
# sess.run(tf.io.write_file(
#     tf.strings.join([original_, '_measures.txt']),
#      output
# ))