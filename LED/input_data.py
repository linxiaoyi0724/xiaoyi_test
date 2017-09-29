# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:48:35 2017

@author: xiaoyi
"""
import tensorflow as tf
import numpy as np
import os 


#%%

def get_files(file_dir1,file_dir2):
    larges = []
    label_larges = []
    smalls =[]
    label_smalls = []
    for file in os.listdir(file_dir1):
        larges.append(file_dir1+file)
        label_larges.append(0)
    for file in os.listdir(file_dir2):
        smalls.append(file_dir2+file)
        label_smalls.append(1)
    
    print('There are %d cats\n There are %d dogs' %(len(larges),len(smalls)))
    
    image_list = np.hstack((larges,smalls))
    label_list = np.hstack((label_larges,label_smalls))
    
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    
    return image_list,label_list
        
#%%
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    
    
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.image.per_image_standardization(image)
    
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size=batch_size,
                                             num_threads=64,
                                             capacity=capacity)
    
    #one-hot                                         
    n_classes = 2
    label_batch = tf.one_hot(label_batch,depth=n_classes)
    label_batch = tf.cast(label_batch,dtype=tf.int32)
    label_batch = tf.reshape(label_batch,[batch_size,n_classes])
    
    
#    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
    
#%% test
#
#import matplotlib.pyplot as plt
#BATCH_SIZE =2
#CAPACITY =256
#IMG_W = 400
#IMG_H = 400
#large_dir = '/home/xiaoyi/data/LED/data/train/train_large_crop/'
#small_dir = '/home/xiaoyi/data/LED/data/train/train_small_crop/'
#
#image_list,label_list = get_files(large_dir,small_dir)
#image_batch,label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
#
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    try:
#        while not coord.should_stop() and i<1:
#            img,label = sess.run([image_batch,label_batch])
#            for j in np.arange(BATCH_SIZE):
#                print('laebl: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#                i+=1
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
                    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
