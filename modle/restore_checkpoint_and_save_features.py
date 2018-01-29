from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from my_image_dataset import image_dataset
import pickle


channels = 3  # original depth of input images
DATA_SET_DIR = "/home/charlesxu/Images/"
files_num = 2146  # number of train dataset
height = 120
width = 160
depth = 1  # depth of train_dataset

num_layers = 5

CONV_1_W = [0] * num_layers
CONV_1_b = [0] * num_layers
DE_W = [0] * num_layers
DE_b = [0] * num_layers
for i in range(num_layers):
    # encode layer W&b

    CONV_1_W[i] = tf.Variable(tf.random_normal(
        [3, 3, 1, 32]), dtype=tf.float32)
    CONV_1_b[i] = tf.Variable(tf.zeros([32]), dtype=tf.float32)  # 16
    # decode layer W&b
    DE_W[i] = tf.Variable(tf.random_normal([3, 3, 32, 1]), dtype=tf.float32)
    DE_b[i] = tf.Variable(tf.zeros([1]), dtype=tf.float32)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    saver.restore(sess, "/home/charlesxu/enough/stacked_CAE_model.ckpt")
    filenames = os.listdir(DATA_SET_DIR)
    features_list = []
    for image in filenames:
        print("image:", image)
        im_test = Image.open(DATA_SET_DIR + image)
        im_test = im_test.convert("L")  # uint8
        im_test = im_test.resize((160, 120), Image.ANTIALIAS)
        im_test = np.reshape(im_test, (-1, height, width, 1))
        im_test = im_test.astype(np.float32)

        input_image = np.multiply(im_test, 1.0 / 255.0)

        input_image = tf.identity(input_image)
        input_x = tf.pad(input_image,
                         [[0, 0], [1, 1], [1, 1], [0, 0]])

        for i in range(num_layers):
            print("layer:", i)
            conv1 = tf.nn.bias_add(tf.nn.conv2d(input_x, CONV_1_W[i], [
                                   1, 1, 1, 1], 'VALID'), CONV_1_b[i])  # batch,480,640,16
            conv1 = tf.nn.tanh(conv1)
            #print("conv1 is:", conv1)
            # only apply this when training

            pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [
                1, 2, 2, 1], 'VALID')  # batch,240,320,16
            #print("max_pool is:", pool1)

            # padding to make it the same size of input_image(batch,480,640,32)
            pad_for_decode = tf.pad(pool1, [[0, 0], [31, 31], [
                41, 41], [0, 0]])  # batch,482,642,16
            #print("pool_pad is :", pad_for_decode)

            # decode layer
            decode_result = tf.nn.bias_add(tf.nn.conv2d(
                pad_for_decode, DE_W[i], [1, 1, 1, 1], 'VALID'), DE_b[i])  # batch,480,640,1
            decode = tf.nn.tanh(decode_result)
            #print("decode is:", decode)
            #print("input_x is:", input_x)
            input_x = tf.stop_gradient(decode)
            input_x = tf.pad(input_x,
                             [[0, 0], [1, 1], [1, 1], [0, 0]])

        features_list.append(sess.run(pool1))  # append an array to list
    with open("stacked_CAE_features.txt", "wb") as fp:
        pickle.dump(features_list, fp)
