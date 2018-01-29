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
        # image_total = np.reshape(image_total, (batch_num, height, Width))
        num_examples = image_total.shape[0]

        coord.request_stop()
        coord.join(threads)
    return image_total, num_examples


# train parameters
learning_rate = 0.00001
training_epochs = 10000
train_batch_size = 29
# image parameters
channels = 3  # original depth of input images
DATA_SET_DIR = "/home/charlesxu/Images/"
files_num = 2146  # number of train dataset
height = 120  # resize input_images to this size
width = 160
depth = 1  # depth of train_dataset

# convolution parameters
conv_stride = 1  # convolution stride
filters_number = 32
filter_side = 3
num_layers = 5

files = DATA_SET_DIR + "*.jpg"


# Images:4D tensor of[batch_size, height, width,
# depth size
images, num_examples = convert_to_array_queue(
    files, files_num, height, width, channels)

print("num_examples is:", num_examples)

input_images = tf.placeholder(
    tf.float32, [None, height, width, depth])  # batch,480,640,1
input_x = tf.pad(input_images, [[0, 0], [1, 1], [1, 1], [0, 0]])
input_x = tf.identity(input_x)

CONV_1_W = [0] * num_layers
CONV_1_b = [0] * num_layers
DE_W = [0] * num_layers
DE_b = [0] * num_layers
# initialize  the list containing num_layers weight matrix
for i in range(num_layers):
    # encode layer W&b
    shape1 = [filter_side, filter_side, input_x.get_shape()[3].value, filters_number]  # 3,3,1,32
    CONV_1_W[i] = tf.Variable(tf.random_normal(
        shape1), dtype=tf.float32)
    CONV_1_b[i] = tf.Variable(tf.zeros([filters_number]), dtype=tf.float32)  # 32

    # decode layer W&b
    shape2 = [filter_side, filter_side, filters_number, input_x.get_shape()[3].value]  # 3,3,32,1
    DE_W[i] = tf.Variable(tf.random_normal(shape2), dtype=tf.float32)
    DE_b[i] = tf.Variable(tf.zeros([input_x.get_shape()[3].value]), dtype=tf.float32)

for i in range(num_layers):
    # encoder layer
    conv1 = tf.nn.bias_add(tf.nn.conv2d(
        input_x, CONV_1_W[i], [1, conv_stride, conv_stride, 1], 'VALID'), CONV_1_b[i])  # batch,height,weight,32
    conv1 = tf.nn.tanh(conv1)
    print("conv1 is:", conv1)

    pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [
        1, 2, 2, 1], 'VALID')  # batch,0.5*height,0.5*width,32
    print("max_pool is:", pool1)

    # only apply this when training
    encode_dropout = tf.nn.dropout(pool1, 0.5)

    # padding to make it the same size of input_image(batch,height,width,32)
    pad_height = int(0.25 * height + 1)
    pad_width = int(0.25 * width + 1)
    pad_for_decode = tf.pad(encode_dropout, [[0, 0], [pad_height, pad_height], [
        pad_width, pad_width], [0, 0]])  # batch,122,162,32
    print("pool_pad is :", pad_for_decode)

    # decode layer
    decode_result = tf.nn.bias_add(tf.nn.conv2d(
        pad_for_decode, DE_W[i], [1, conv_stride, conv_stride, 1], 'VALID'), DE_b[i])  # batch,120,160,1
    decode = tf.nn.tanh(decode_result)
    print("decode is:", decode)
    #print("input_x is:", input_x)
    print("input_images is:", input_images)

    mse = tf.div(tf.reduce_mean(tf.square(decode - input_images)), 2)
    tf.add_to_collection('losses', mse)
    input_x = tf.stop_gradient(decode)
    input_x = tf.pad(input_x,
                     [[0, 0], [1, 1], [1, 1], [0, 0]])

total_error = tf.add_n(tf.get_collection('losses'))
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(total_error)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(mse)  # minimize final_layer_error can get a better result

init = tf.initialize_all_variables()
# enable device memory grows gradually
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
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
            _, c = sess.run([optimizer, mse], feed_dict={
                            input_images: batch_xs_reshape})

        print("Epoch:", '%04d' % (epoch), "final_error=", "{:.9f}".format(c))

    # print("CONV_1[1] is:", CONV_1_W[1].eval(session=sess))
    # print("CONV_1_b[1] is:", CONV_1_b[1].eval(session=sess))
    coord.request_stop()
    coord.join(threads)
    print("Optimization Finished!")
    saver = tf.train.Saver()
    save_path = saver.save(sess, "stacked_CAE_model.ckpt")
    print("Model saved in file: ", save_path)
