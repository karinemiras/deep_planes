from __future__ import print_function
import tensorflow as tf
import os
import argparse

w = 200 #400
h = 150 #300



parser = argparse.ArgumentParser()

parser.add_argument('--original',
                    default='Plane_train')
parser.add_argument('--class_',
                    default='Plane_train_labels')
parser.add_argument('--pred',
                    default='planetrain_model_plane_040905')

flags = parser.parse_args()
#
# original_ = 'Plane_train'
# class_ = 'Plane_train_labels'
# pred__ = 'planetrain_model_plane_040905'

original_ = flags.original
class_ = flags.class_
pred__ = flags.pred

path_original = 'dataset/'+original_
path_class = 'dataset/'+class_
path_pred = 'result1/'+pred__

images_class = []
images_pred = []

accuracy = []
precision = []
recall = []
#fscore = []

sess = tf.Session()


for r, d, f in os.walk(path_original):
    for file in f:
        if '.jpg' in file or '.png' in file:
            if os.path.isfile(path_class+'/'+file.split('.')[0]+'.png'):
                images_class.append(path_class+'/'+file.split('.')[0]+'.png')
            else:
                images_class.append('dataset/none.png')

            images_pred.append(path_pred+'/'+file)

print('>> read')

batch = 300
batch_counter = 0
for i in range(0, len(images_class)):
    print(images_class[i])
    image_file = tf.read_file(images_class[i])
    image_class = tf.image.decode_png(image_file, channels=1)
    image_class = tf.image.resize_images(image_class, (h, w))

    image_file = tf.read_file(images_pred[i])
    image_pred = tf.image.decode_png(image_file, channels=1)
    image_pred = tf.image.resize_images(image_pred, (h, w))

    serial_class = tf.reshape(image_class, [-1])
    serial_pred = tf.reshape(image_pred, [-1])

    class_ = tf.less_equal(serial_class, 127)
    pred_ = tf.less_equal(serial_pred, 127)

    zeros = tf.fill([1, tf.size(serial_class)], 0)[0]
    uns = tf.fill([1, tf.size(serial_class)], 1)[0]

    class_ = tf.where(class_, zeros, uns)
    pred_ = tf.where(pred_, zeros, uns)

    if batch_counter == 0:
        all_class = class_
        all_pred = pred_
        batch_counter += 1
    elif batch_counter < batch:
        all_class = tf.concat([all_class, class_], 0)
        all_pred = tf.concat([all_pred, pred_], 0)
        batch_counter += 1

    if batch_counter == batch or i == len(images_class)-1:
        batch_counter = 0
        accuracy.append(tf.gather_nd(tf.metrics.accuracy(all_class, all_pred), [1]))

        if (tf.math.reduce_sum(all_class)).eval(session=sess) > 0:
            precision.append(tf.gather_nd(tf.metrics.precision(all_class, all_pred), [1]))
            recall.append(tf.gather_nd(tf.metrics.recall(all_class, all_pred), [1]))
            #fscore.append(tf.gather_nd(tf.contrib.metrics.f1_score(all_class, all_pred), [1]))

print('>> calculated')

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
sess.run(global_init)
sess.run(local_init)

accuracy = sum(accuracy)/len(accuracy)
precision = sum(precision)/len(precision)
recall = sum(recall)/len(recall)
#fscore = sum(fscore)/len(fscore)
fscore = 2 *((precision*recall)/(precision+recall))

# sess.run(tf.print(accuracy))
# sess.run(tf.print(precision))
# sess.run(tf.print(recall))
# sess.run(tf.print(fscore))


accuracy = tf.strings.join(['accuracy ',tf.dtypes.as_string(accuracy)])
precision = tf.strings.join(['precision ',tf.dtypes.as_string(precision)])
recall = tf.strings.join(['recall ',tf.dtypes.as_string(recall)])
fscore = tf.strings.join(['fscore ',tf.dtypes.as_string(fscore)])

output = tf.strings.join([accuracy, '\n',
                          precision, '\n',
                          recall, '\n',
                          fscore
                         ])

sess.run(tf.io.write_file(
    tf.strings.join(['analysis/',pred__, '_measures.txt']),
     output
))