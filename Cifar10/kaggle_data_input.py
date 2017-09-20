# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:07:43 2017

@author: xiaoyi
"""
import tensorflow as tf
import numpy as np
import pandas as pd


img_width = 32
img_height = 32
file_dir = '/home/xiaoyi/data/Cifar10/train/'

def get_files(file_dir):
    img_labels = pd.read_csv('/home/xiaoyi/data/Cifar10/trainLabels.csv')
    img_labels = img_labels.values
    image = []
    label = []
    for j in range(50000):

        image.append(file_dir+str(j+1)+'.png')
    for i in range(50000):
        label.append(img_labels[i,1])

    print('there is %d image' %(len(image)))

    temp = np.array([image,label])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    return image_list,label_list



def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents,channels=3)


    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
#    image = tf.image.per_image_standardization(image)


    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size=batch_size,
                                             num_threads=64,
                                             capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch





#import matplotlib.pyplot as plt
#BATCH_SIZE =2
#CAPACITY =256
#IMG_W = 32
#IMG_H = 32
#train_dir = '/home/xiaoyi/data/Cifar10/train/'
#
#image_list,label_list = get_files(train_dir)
#image_batch,label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
#
#with tf.Session() as sess:
#   i=0
#   coord= tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#
#   try:
#       while not coord.should_stop() and i<1:
#           img,label = sess.run([image_batch,label_batch])
#           for j in np.arange(BATCH_SIZE):
#               print('label: %d'%label[j])
#               plt.imshow(img[j,:,:,:])
#               plt.show()
#               i+=1
#   except tf.errors.OutOfRangeError:
#       print('done!')
#   finally:
#       coord.request_stop()
#   coord.join(threads)

















