from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import tempfile
import time
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from my_image_dataset import image_dataset
import pickle


def convert_to_array_queue(files, files_num, height, width, channels):
        # convert files containing multple images in png format to
   # gray_scale images and pack them in
   # image batch[batch_num,height,width]

    filenames = tf.train.match_filenames_once(files)
    filename_queue = tf.train.string_input_producer(filenames)

    # step 3: read, decode and resize images
    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    image = tf.image.decode_jpeg(content, channels=channels)
    # for python 3
    resized_image = tf.image.resize_images(image, [height, width])
    # for python 2
    #resized_image = tf.image.resize_images(image, height, width)
    gray_images = tf.image.rgb_to_grayscale(resized_image)

    # step 4: batching
    image_batch = tf.train.batch([gray_images], batch_size=1)
    batch_size = 1
    batch_num = int(files_num / batch_size)

    with tf.Session() as sess_1:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess_1, coord=coord)
        image_total = []
        for i in range(batch_num):
            image_tensor = image_batch.eval()  # (1,height,width,1)
            image_array = np.asarray(image_tensor[0])  # (height,width,1)
            image_total.append(image_array)

        # convert list to array
        image_total = np.array(image_total)  # (batch_num,height,W1idth,1))
        image_total = np.multiply(image_total, 1.0 / 255.0)
        #image_total = np.reshape(image_total, (batch_num, height, W1idth))
        num_examples = image_total.shape[0]

        coord.request_stop()
        coord.join(threads)
    return image_total, num_examples

learning_rate = 0.00001
training_epochs = 100000

height = 480
width = 640
depth = 1
filter_side = 3
stride = 1
filters_number = 32
amount = filter_side - 1

channels = 3  # original depth of input images
files = "/home/charlesxu/Images/*.jpg"
files_num = 2146

train_batch_size = 29

# Images. 4D tensor of[b1atch_size, self._image_height, self._image_W1idth,
# self._image_depth] size
images, num_examples = convert_to_array_queue(
    files, files_num, height, width, channels)

print("num_examples is:", num_examples)

input_x = tf.placeholder(tf.float32, [None, height, width, depth])
pad_input = tf.pad(input_x,
                   [[0, 0], [amount, amount], [amount, amount], [0, 0]])

# encode layer W&b
shape1 = [filter_side, filter_side, pad_input.get_shape()[3].value,
          filters_number]
W1 = tf.Variable(tf.random_normal(shape1), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([shape1[3]]), dtype=tf.float32)

# decode layer
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
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    images_reshape = np.reshape(images, (files_num, height * width))
    image_batches = image_dataset(images_reshape, images_reshape)
    print("num_examples =", image_batches._num_examples)

    # train cycle

    total_batch = int(files_num / train_batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs = image_batches.next_batch(train_batch_size)
            batch_xs_reshape = np.reshape(
                batch_xs, (train_batch_size, height, width, depth))
            _, c = sess.run([optimizer, mse_error], feed_dict={
                            input_x: batch_xs_reshape})

        print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c))

    print("W1 is:", W1.eval(session=sess))
    print("b1 is:", b1.eval(session=sess))
    coord.request_stop()
    coord.join(threads)
    print("Optimization Finished!")
    saver = tf.train.Saver()
    save_path = saver.save(sess, "CAE_model.ckpt")
    print("Model saved in file: ", save_path)

    filenames = os.listdir("/home/charlesxu/Images_12")

    features_list = []
    for image in filenames:
        im_test = Image.open("/home/charlesxu/Images_12/"+image)
        im_test = im_test.convert("L")  # uint8
        im_test = np.reshape(im_test, (-1, height, width, 1))
        im_test = im_test.astype(np.float32)

        im_test = np.multiply(im_test, 1.0 / 255.0)
        pad_im_test = tf.pad(im_test,
                             [[0, 0], [amount, amount], [amount, amount], [0, 0]])

        encode_result = tf.nn.bias_add(tf.nn.conv2d(
            pad_im_test, W1, [1, stride, stride, 1], 'VALID'), b1)
        encode = tf.nn.tanh(encode_result)
        features_list.append(encode.eval())
        # print(sess.run(encode))
    print(features_list)
    #print("feature0 is: ",features_list[0].eval())
    #print("feature0 shape is:",(features_list[0].eval()).shape)
    
    with open("test.txt","wb") as fp:
        pickle.dump(features_list,fp)

