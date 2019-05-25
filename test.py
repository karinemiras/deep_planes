# coding:utf-8
# Bin GAO

import os
import cv2
import glob
import tensorflow as tf
import numpy as np
import argparse

w =  200 #400
h =  150 #300

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',
                    type=str,
                    default='dataset/Plane_train')
parser.add_argument('--model_dir',
                    type=str,
                    default='./models/model_plane_040905')
parser.add_argument('--save_dir',
                    type=str,
                    default='./result1/planetrain_model_plane_040905')
parser.add_argument('--gpu',
                    type=int,
                    default=0)
parser.add_argument('--with_batch',
                    type=int,
                    default=0)
flags = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_model():
    file_meta = os.path.join(flags.model_dir, 'model.ckpt.meta')
    file_ckpt = os.path.join(flags.model_dir, 'model.ckpt')

    saver = tf.train.import_meta_graph(file_meta)
    # tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

    sess = tf.InteractiveSession()
    saver.restore(sess, file_ckpt)
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
    return sess


def read_image(image_path, gray=False):
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_and_resize(imageName):
    inputname = os.path.join(flags.input_dir, imageName)
    image = read_image(inputname)
    return cv2.resize(image, (w, h))


def main(flags):
    sess = load_model()
    X, mode = tf.get_collection('inputs')
    pred = tf.get_collection('upscore_fuse')[0]

    os.mkdir(flags.save_dir)
    names = os.listdir(flags.input_dir)
 
    # names.remove('.DS_Store')

    for name in names:
        # add here, if png or jpg in name!
        inputname = os.path.join(flags.input_dir, name)
        print(inputname)
        image = read_image(inputname)
        image = cv2.resize(image, (w, h))
        # sess=tf.InteractiveSession()

        label_pred = sess.run(pred, feed_dict={X: np.expand_dims(image, 0), mode: False})
        merged = np.squeeze(label_pred) * 255
        _, merged = cv2.threshold(merged, 127, 255, cv2.THRESH_BINARY)
        save_name = os.path.join(flags.save_dir, name)
        cv2.imwrite(save_name, merged)


def main_with_batch_size(flags):
    sess = load_model()
    X, mode = tf.get_collection('inputs')
    pred = tf.get_collection('upscore_fuse')[0]


    names = os.listdir(flags.input_dir)
    # here, extract name and make it png
    print(names)
    
    names = os.listdir(flags.input_dir)
    # names.remove('.DS_Store')

    names = names[:16]
    images = [read_and_resize(n) for n in names]

    label_preds = sess.run(pred, feed_dict={X: np.array(images), mode: False})
    images = np.array(label_preds * 255)

    for i in range(images.shape[0]):
        image = images[i]
        _, merged = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        save_name = os.path.join(flags.save_dir, names[i])
        cv2.imwrite(save_name, merged)
        print('Pred saved')


if __name__ == '__main__':
    if flags.with_batch == 0:
        main(flags)
    else:
        main_with_batch_size(flags)
