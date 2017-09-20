# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:01:17 2017

@author: xiaoyi
"""
#%%
import os 
import os.path
import math
import numpy as np
import tensorflow as tf
import kaggle_data_input

#%%
BATCH_SIZE = 1
learning_rate = 0.005
MAX_STEP = 20000
TRAIN = True
IMG_W=32
IMG_H=32
CAPACITY = 2000

#%%
img_width = 32
img_height = 32
img_depth = 3
img_pexel = img_width*img_height*img_depth

#%%
def inference(images):
    #conv1
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,3,96],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[96],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name=scope.name)
        
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1= tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')
       
       
       
       
       
       
       
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,96,64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name='conv2')
    
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME',name='pooling2')       
        
        
        
        
        
        
        

    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2,shape=[BATCH_SIZE,-1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        
        
        
        
        
        
        
        
        
        
        
        
    #local4    
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384,192],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name='local4')
        
        
        
        
        
    #softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[192,10],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[10],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name='softmax_linear')
        
    return softmax_linear
        
        
        
        



def losses(logits,labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels,tf.int64)
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name='loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss
        
        
  

      
        
#%% 

def train():
    my_global_step = tf.Variable(0,name='global_step',trainable=False)
    data_dir = '/home/xiaoyi/data/Cifar10/train/'
    log_dir = '/home/xiaoyi/data/Cifar10/kaggle_logs/'
    
    images,labels = kaggle_data_input.get_files(data_dir)
    images_batch,label_batch = kaggle_data_input.get_batch(images,
                                                           labels,
                                                           IMG_W,
                                                           IMG_H,
                                                           BATCH_SIZE,
                                                           CAPACITY)
    
    images_batch = tf.cast(images_batch,tf.float32)
    logits = inference(images_batch)
    loss  = losses(logits,label_batch)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss,global_step=my_global_step)
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    summary_writer = tf.summary.FileWriter(log_dir,sess.graph)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,loss_value = sess.run([train_op,loss])
            
            if step % 50 ==0:
                print('Step: %d,loss: %.4f' % (step,loss_value))
            
            if step % 100 ==0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str,step)
                
            if step %2000 == 0 or (step+1) == MAX_STEP:
                checkpoint_path = os.path.join(log_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
    
    
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()     

#%%    


from PIL import Image
import pandas as pd
def get_one_image(img_dir):
    image = Image.open(img_dir)
   # plt.imshow(image)
    image = image.resize([32,32])
    image = np.array(image)
    return image

def evaluate_one_image():
    img_dir = '/home/xiaoyi/data/Cifar10/test/'
    path = '/home/xiaoyi/data/Cifar10/'
    a = []
    n = 3
    for i in xrange(n):
        print(i)
        image_array = get_one_image(img_dir+str(i+1)+'.png')
        
        with tf.Graph().as_default():
            image = tf.cast(image_array,tf.float32)
            image = tf.reshape(image,[1,32,32,3])
            logit = inference(image)
            logit = tf.nn.softmax(logit)
            
            x = tf.placeholder(tf.float32,shape=[32,32,3])
            log_dir = '/home/xiaoyi/data/Cifar10/kaggle_logs/'
            saver = tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                print('Reading checkpoints...')
                ckpt = tf.train.get_checkpoint_state(log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    print('loading seccess,global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')
                prediction = sess.run(logit,feed_dict={x:image_array})
                max_index = np.argmax(prediction)
                if max_index == 0:
                    a.append('airplane')
                if max_index == 1:
                    a.append('automobile')
                if max_index == 2:
                    a.append('bird')
                if max_index == 3:
                    a.append('cat')
                if max_index == 4:
                    a.append('deer')
                if max_index == 5:
                    a.append('dog')
                if max_index == 6:
                    a.append('frog')
                if max_index == 7:
                    a.append('horse')
                if max_index == 8:
                    a.append('ship')
                if max_index == 9:
                    a.append('truck')
    ImageId = [i+1 for i in range(n)]
    submit_pd = pd.DataFrame({'id':ImageId,'label':a})
    submit_pd.to_csv(path+'model-test.csv',index=False)



















