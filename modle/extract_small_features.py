import pickle
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold='nan')


# load the file containing features_list
with open("stacked_CAE_features.txt", "rb") as fp:
    feature_array_list = pickle.load(fp)  # 2146,60,80,32
    print(feature_array_list[0].shape)
'''
image = [0][0][0][0] * batch_num * 32
small_feature = [0][0][0] * batch_num * 4 * 6
for i in range(batch_num):  # for every image's feature[1,60,80,32]
    image[i] = tf.reshape(feature_array_list[i], [60, 80, 32])
    image[i] = tf.transpose(image[i], [2, 0, 1])  # [32,60,80]
    for j in range(32):
        for m in range(28, 32):  # 28,29,30,31
            p = 0
            for n in range(37, 43):  # 37 38 39 40 41 42
                q = 0
                small_feature[i][p][q] = image[i][j][m][n]
                q = q + 1
            p = p + 1
print(small_feature.shape)
'''
'''
total_feature=[]
for i in range(batch_num):
    x = []
    feature_array_list[i] = tf.reshape(feature_array_list[i], [60, 80, 32])
    feature_array_list[i] = tf.transpose(feature_array_list[i], [2, 0, 1])
    for j in range(32):
        for m in range(28, 32):
            for n in range(37, 43):
                x.append(feature_array_list[i][j][m][n])
    print(len(x))
    total_feature.append(x)
print(len(total_feature))
print(total_feature[0].eval)
'''

total_feature=[]
for i in range(0,2146,2):
    x = []
    feature_array_list[i] = np.reshape(feature_array_list[i], [60, 80, 32])
    feature_array_list[i] = np.transpose(feature_array_list[i], [2, 0, 1])
    for j in range(32):
        for m in range(28, 32):
            for n in range(37, 43):
                x.append(feature_array_list[i][j][m][n])
    print("image",i)
    total_feature.append(x)
print(len(total_feature))
with open("small_CAE_features.txt", "wb") as fp_small:
    pickle.dump(total_feature, fp_small)
