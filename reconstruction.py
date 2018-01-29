# -*- coding: utf-8 -*-
#import modules
import sys
import tensorflow as tf
import math
import numpy as np
from PIL import Image, ImageFilter

#强制打印整个数组
np.set_printoptions(threshold='nan')

# Parameters
learning_rate = 0.00001
training_epochs = 1000

height = 480
width = 640
depth = 1
filter_side = 3
stride=1
filters_number = 32
amount = filter_side - 1

channels = 3  #original depth of input images
files = "/home/plz/slam/Images_12/*.jpg"
files_num = 12

input_x = tf.placeholder(tf.float32, [None, height, width, depth])
pad_input = tf.pad(input_x,
                   [[0, 0], [amount, amount], [amount, amount], [0, 0]])

#encode layer W&b
shape1 = [filter_side, filter_side, pad_input.get_shape()[3].value,
          filters_number]
W1 = tf.Variable(tf.random_normal(shape1), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([shape1[3]]), dtype=tf.float32)

#decode layer W&b
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
    tf.square(reconstructions-input_x)), 2)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(mse_error)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

#image preprocessing
def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im_test = Image.open(argv).convert("L")    #uint8
    #im_test = im_test.resize((80,60), Image.ANTIALIAS)
    print(im_test)	
    im_test=np.reshape(im_test,(-1,height,width,depth))
    im_test=im_test.astype(np.float32)
    #normalize pixels to 0 and 1. 0—->0, 1-->255.
    im_test=np.multiply(im_test,1.0/255.0)
    return im_test

def predictint(imvalue):
	x=tf.placeholder(tf.float32, [None, height, width, depth])
	pad_x = tf.pad(x,[[0, 0], [amount, amount], [amount, amount], [0, 0]])
	#encode layer
	encode_result = tf.nn.bias_add(tf.nn.conv2d(pad_x, W1, [1, stride, stride, 1], 'VALID'), b1)
	encode = tf.nn.tanh(encode_result)
    
    
	decode_result = tf.nn.bias_add(tf.nn.conv2d(encode, W2, [1, stride, stride, 1], 'VALID'), b2)
	decode = tf.nn.tanh(decode_result)
	
	init_op=tf.initialize_all_variables()
	saver=tf.train.Saver()
	
	#load the model file
	with tf.Session() as sess:
		sess.run(init_op)
		saver.restore(sess,"CAE_model.ckpt")
		print ("model restored")
		return decode.eval(feed_dict={x: imvalue},session=sess)
		
#main function
def main(argv):
	imvalue=imageprepare(argv)
	pred_img=predictint(imvalue)
	return pred_img
	
if __name__=="__main__":
	pred_img=main(sys.argv[1])
	print(pred_img.dtype)
	print(pred_img)
	
	pred_img=np.multiply(pred_img,255.0)
	pred_img = np.reshape(pred_img, (480, 640))
	#print(pred_img)
	pred_img=pred_img.astype(np.uint8)   #convert into "L"
	
	#print(pred_img)
	pred_img=Image.fromarray(pred_img)
	print(pred_img.mode)
	pred_img.show()

