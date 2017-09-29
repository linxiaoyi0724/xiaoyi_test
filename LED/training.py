# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:14:32 2017

@author: xiaoyi
"""
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import input_data
import VGG
import tools

#%%
IMG_W = 100
IMG_H = 100
N_CLASSES = 2
BATCH_SIZE = 6
CAPACITY = 2000
learning_rate = 0.001
MAX_STEP = 5000
IS_PRETRAIN = True

#%%
def train():
    pre_trained_weights = '/home/xiaoyi/data/LED/VGG16_pretrained/vgg16.npy'
    large_dir = '/home/xiaoyi/data/LED/data/train/train_large_crop/'
    small_dir = '/home/xiaoyi/data/LED/data/train/train_small_crop/'
    val_large_dir = '/home/xiaoyi/data/LED/test/test_large/'
    val_small_dir = '/home/xiaoyi/data/LED/test/test_small/'
    train_log_dir = '/home/xiaoyi/data/LED/logs1/train/'
    val_log_dir = '/home/xiaoyi/data/LED/logs1/val/'
    
    with tf.name_scope('input'):
        train,train_laebl = input_data.get_files(large_dir,small_dir)
        train_batch,train_label_batch = input_data.get_batch(train,
                                                             train_laebl,
                                                             IMG_W,
                                                             IMG_H,
                                                             BATCH_SIZE,
                                                             CAPACITY)
        val,val_label = input_data.get_files(val_large_dir,val_small_dir)
        val_batch,val_label_batch = input_data.get_batch(val,
                                                         val_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)
        
    logits = VGG.VGG16N(train_batch,N_CLASSES,IS_PRETRAIN)
    loss = tools.loss(logits,train_label_batch)
    accuracy = tools.accuracy(logits,train_label_batch)
    my_global_step = tf.Variable(0,name='global_step',trainable=False)
    train_op = tools.optimize(loss,learning_rate,my_global_step)
    
    x = tf.placeholder(tf.float32,shape=[BATCH_SIZE,IMG_W,IMG_H,3])
    y_ = tf.placeholder(tf.int16,shape=[BATCH_SIZE,N_CLASSES])
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    tools.load_with_skip(pre_trained_weights,sess,['fc6','fc7','fc8'])
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir,sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir,sess.graph)
    
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            
            tra_images,tra_labels =sess.run([train_batch,train_label_batch])
            _,tra_loss,tra_acc = sess.run([train_op,loss,accuracy],
                                          feed_dict = {x:tra_images,y_:tra_labels})
                                         
            if step % 50 ==0 or (step + 1) == MAX_STEP:
                print('Step: %d,loss:%.4f,accuracy:%.4f%%' % (step,tra_loss,tra_acc))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str,step)
                
                
            if step % 200 == 0 or (step + 1 ) == MAX_STEP:
                val_images,val_labels =sess.run([val_batch,val_label_batch])
                val_loss,val_acc =sess.run([loss,accuracy],
                                           feed_dict = {x:val_images,y_:val_labels})
                print('** Step %d,val loss = %.2f,val accuracy = %.2f%% **' %(step,val_loss,val_acc))
                
                summary_str = sess.run(summary_op)
                val_summary_writer.add_summary(summary_str,step)
                
            if step % 2000 == 0 or (step + 1 ) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
            
            
            
        
        
        
    
 































