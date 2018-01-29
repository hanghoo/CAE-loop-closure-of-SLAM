# -*- coding: utf-8 -*-
# import modules
import sys
import tensorflow as tf
import math
import numpy as np
from PIL import Image, ImageFilter

# 强制打印整个数组
np.set_printoptions(threshold='nan')

# Parameters
learning_rate = 0.00001
training_epochs = 1000

height = 480
width = 640
depth = 1
filter_side = 3
stride = 1
filters_number = 32
amount = filter_side - 1

channels = 3  # original depth of input images
files = "/home/plz/slam/Images_12/*.jpg"
files_num = 12

input_x = tf.placeholder(tf.float32, [None, height, width, depth])
pad_input = tf.pad(input_x,
                   [[0, 0], [amount, amount], [amount, amount], [0, 0]])

# encode layer W&b
shape1 = [filter_side, filter_side, pad_input.get_shape()[3].value,
          filters_number]
W1 = tf.Variable(tf.random_normal(shape1), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([shape1[3]]), dtype=tf.float32)

# decode layer W&b
shape2 = [filter_side, filter_side, filters_number, pad_input.get_shape()[
    3].value]
W2 = tf.Variable(tf.random_normal(shape2), dtype=tf.float32)
b2 = tf.Variable(tf.zeros([shape2[3]]), dtype=tf.float32)

# encoder layer
encode_result = tf.nn.bias_add(
    tf.nn.conv2d(pad_input, W1, [1, stride, stride, 1], 'VALID'), b1)
encode = tf.nn.tanh(encode_result)

# decode layer
decode_result = tf.nn.bias_add(
    tf.nn.conv2d(encode, W2, [1, stride, stride, 1], 'VALID'), b2)
decode = tf.nn.tanh(decode_result)

reconstructions = decode

mse_error = tf.div(tf.reduce_mean(
    tf.square(reconstructions - input_x)), 2)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(mse_error)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

# image preprocessing


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im_test = Image.open(argv).convert("L")  # uint8
    # im_test = im_test.resize((80,60), Image.ANTIALIAS)
    #print(im_test)
    im_test = np.reshape(im_test, (-1, height, width, depth))
    im_test = im_test.astype(np.float32)
    # normalize pixels to 0 and 1. 0—->0, 1-->255.
    im_test = np.multiply(im_test, 1.0 / 255.0)
    return im_test


def get_feature(imvalue):
    x = tf.placeholder(tf.float32, [None, height, width, depth])
    pad_x = tf.pad(x, [[0, 0], [amount, amount], [amount, amount], [0, 0]])
    # encode layer
    encode_result = tf.nn.bias_add(tf.nn.conv2d(
        pad_x, W1, [1, stride, stride, 1], 'VALID'), b1)
    encode = tf.nn.tanh(encode_result)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    # load the model file
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "CAE_model.ckpt")
        print("model restored")
        return encode.eval(feed_dict={x: imvalue}, session=sess)


def similarity_compute(x, y):  # x,y均为list格式，如[1,2,3,4]
    if(len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    x_np=np.asarray(x)
    y_np=np.asarray(y)
    for i in range(len(x)):
        result1 += x_np[i] * y_np[i]
        result2 += x_np[i] * x_np[i]
        result3 += y_np[i] * y_np[i]
    print("result is:" + str(result1 / ((result2 * result3)**0.5)))
    return result1 / ((result2 * result3)**0.5)

# main function


def main(argv):
    imvalue = imageprepare(argv)
    feature = get_feature(imvalue)
    return feature

if __name__ == "__main__":
    feature1 = main(sys.argv[1])
    feature2 = main(sys.argv[2])
    feature3 = main(sys.argv[3])
    feature1 = feature1.tolist()
    feature2 = feature2.tolist()
    feature3 = feature3.tolist()

    simi_12 = similarity_compute(feature1, feature2)
    simi_13 = similarity_compute(feature1, feature3)
    simi_23 = similarity_compute(feature2, feature3)

    print(simi_12)
    print(simi_13)
    print(simi_23)
