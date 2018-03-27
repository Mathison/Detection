
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from dataloader import Loader
import importlib
from subpixel import subpix_conv2d
from convert_mat_to_img import convert_nyu
import os
import h5py
import dataloader_tf as dtf


# In[3]:


current_directory = os.getcwd()
nyu_path = './nyu_depth_v2_labeled.mat'
convert_nyu(nyu_path)


# In[2]:


default_activation = tf.nn.leaky_relu
def redefine_loss(logits, depths):
    logits_flat = tf.reshape(logits, [tf.shape(logits)[0],-1])
    depths_flat = tf.reshape(depths, [tf.shape(depths)[0],-1])
    predict=logits_flat
    target=depths_flat
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / tf.cast(tf.shape(logits_flat)[1],'float32') - 0.5*sqare_sum_d / tf.cast(tf.shape(logits_flat)[1],'float32')**2)
    return cost 
def accuracy(logits, depths, delta=1.25):
    return tf.reduce_mean(tf.cast(tf.maximum(tf.subtract(logits, depths), tf.subtract(depths, logits))<tf.log(delta), tf.float32))


# In[3]:


def conv_layer(image):
    # Hidden layer with 96 neurons
    layer_1 = tf.layers.conv2d(image,filters=96,kernel_size=[11,11],strides=4,padding='VALID',activation=default_activation,name='CoarseConv1')
    layer_1 = tf.layers.max_pooling2d(layer_1,pool_size=2,strides=2,name='CoarseMax1')
    #layer_1 = tf.layers.batch_normalization(layer_1, training=is_training, name='CoarseConvBN1')
    
    # Hidden layer with 256 neurons
    layer_2 = tf.layers.conv2d(layer_1,filters=256,kernel_size=[5,5],strides=1,padding='SAME',activation=default_activation,name='CoarseConv2')
    layer_2 = tf.layers.max_pooling2d(layer_2,pool_size=[2,2],strides=2,name='CoarseMax2')
    #layer_2 = tf.layers.batch_normalization(layer_2, training=is_training, name='CoarseConvBN2')
    
    layer_3 = tf.layers.conv2d(layer_2,filters=384,kernel_size=[3,3],strides=1,padding='SAME',activation=default_activation,name='CoarseConv3')
    #layer_3 = tf.layers.batch_normalization(layer_3, training=is_training, name='CoarseConvBN3')
    
    layer_4 = tf.layers.conv2d(layer_3,filters=384,kernel_size=[3,3],strides=1,padding='SAME',activation=default_activation,name='CoarseConv4')
    #layer_4 = tf.layers.batch_normalization(layer_4, training=is_training, name='CoarseConvBN4')
    layer_5 = tf.layers.conv2d(layer_4,filters=128,kernel_size=[3,3],strides=2,padding='VALID',activation=default_activation,name='CoarseConv5')
    #layer_5 = tf.layers.batch_normalization(layer_5, training=is_training, name='CoarseConvBN5')
    return layer_5

def fully_connect_layer(conv_data,dropout,is_training):
    conv_data = tf.reshape(conv_data,[-1,conv_data.shape[1]*conv_data.shape[2]*conv_data.shape[3]])
    print(conv_data.shape)
    layer_1 = tf.layers.dense(conv_data,units=4096,activation=default_activation,name='CoarseFC1')
    #layer_1 = tf.layers.batch_normalization(layer_1, training=is_training, name='CoarseFCBN1')
    layer_1 = tf.layers.dropout(layer_1,rate=dropout,training=is_training,name='CoarseFCDrop1')
    
    layer_2 = tf.layers.dense(layer_1,units=4070,activation=None,name='CoarseFC2')
    #layer_2 = tf.layers.batch_normalization(layer_2, training=is_training, name='CoarseFCBN2')
    out_layer = tf.reshape(layer_2,[-1,55,74,1])
    return out_layer


# In[4]:


def coarse(image, dropout, is_training):
    conv_data = conv_layer(image) #6x8x256
    coarse = fully_connect_layer(conv_data,dropout,is_training)
#     coarse = subpix_conv2d(conv_data, 128, (5,5), 3)
#     coarse = subpix_conv2d(coarse, 64, (5,5), 3)
#     coarse = tf.layers.conv2d_transpose(coarse, 1, (2,3), 1)
    return coarse #74x55x1


# In[5]:


def fine(image, coarse, fine_gate, dropout, is_training):
    coarse = tf.cond(fine_gate, lambda : tf.stop_gradient(coarse), lambda : coarse)
    layer_1 = tf.layers.conv2d(image,filters=64,kernel_size=[9,9],strides=2,padding='VALID',activation=default_activation,name='FineConv1')
    layer_1_d = tf.nn.dropout(layer_1, dropout)
    #print(layer_1.shape)
    #layer_1 = tf.layers.max_pooling2d(layer_1,pool_size=[2,2],strides=2,data_format='channels_last',name='FineMax1')
    #layer_1 = tf.layers.batch_normalization(layer_1, training=is_training, name='FineBN1')
    layer_2 = tf.layers.conv2d(layer_1_d,filters=63,kernel_size=[1,1],strides=2,padding='SAME',activation=default_activation,name='FineConv2')
    layer_2_d = tf.nn.dropout(layer_2, dropout)
    #print(layer_2.shape)
    #print(coarse.shape)
    catted = tf.concat([layer_2_d,coarse],axis=3, name='Fine1to2')
    #print(catted.shape)
    
    #tf.image.resize_images(catted, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    '''
    layer_3 = tf.layers.conv2d(catted,filters=64,kernel_size=[5,5],strides=1,padding='SAME',activation=default_activation,name='FineConv3')
    print(layer_3.shape)
    #layer_3 = tf.layers.batch_normalization(layer_3, training=is_training, name='FineBN2')
    layer_4 = tf.layers.conv2d(layer_3,filters=1,kernel_size=[5,5],strides=1,padding='SAME',activation=default_activation,name='FineConv4')
    print(layer_4.shape)
    #layer_4 = tf.layers.batch_normalization(layer_4, training=is_training, name='FineBN3')
    '''
    '''
    layer_3 = tf.layers.conv2d_transpose(catted,filters=64, kernel_size=[5,5],strides=(2,2),padding='same')
    print(layer_3.shape)
    layer_4 = tf.layers.conv2d_transpose(layer_3,filters=1, kernel_size=[10,10],strides=(2,2),padding='VALID')
    print(layer_4.shape)
    ##############get the extra layer from coarse that directly connect to output
    extra_coarse_1 = tf.layers.conv2d_transpose(coarse,filters=1, kernel_size=[5,5],strides=(2,2),padding='same')
    print(extra_coarse_1.shape)
    extra_coarse_2 = tf.layers.conv2d_transpose(extra_coarse_1,filters=1, kernel_size=[10,10],strides=(2,2),padding='VALID')
    print(extra_coarse_2.shape)
    '''
    
    layer_3 = subpix_conv2d(catted, 64, 5, 2, name='Fineconv3')
    #print(layer_3.shape)
    layer_4 = subpix_conv2d(layer_3, 1, 5, 2, name='Fineconv4')
    #print(layer_4.shape)
    ##############get the extra layer from coarse that directly connect to output
    extra_coarse_1 = subpix_conv2d(coarse, 64, 5, 2, name='coarse2out1')
    #print(extra_coarse_1.shape)
    extra_coarse_2 = subpix_conv2d(extra_coarse_1, 1, 5, 2, name='coarse2out2')
    #print(extra_coarse_2.shape)
    
    out = tf.cond(is_training,
                  lambda : tf.cond(fine_gate, lambda : layer_4, lambda : 0.1*layer_4+0.9*extra_coarse_2),#coarse
                  lambda : layer_4)
    return out


# In[6]:


def augment(images,depths,crop_size,scale_range,rot_range,flip):
    catted = tf.concat([images, depths], axis=3)
    s = tf.random_uniform([], *scale_range)
    scaled = tf.image.resize_images(catted, tf.cast(tf.cast(tf.shape(catted)[1:3], tf.float32)*s, tf.int32))
    scaled = tf.concat([scaled[...,:-1], tf.divide(scaled[...,-1:], s)], axis=3)
    #rotated = tf.contrib.image.rotate(scaled, tf.random_uniform([tf.shape(scaled)[0],], *rot_range)/180*3.14159265)
    rotated = scaled
    cropped = tf.map_fn(lambda img: tf.random_crop(img, (crop_size[0], crop_size[1], catted.shape[3])), rotated)
    if flip:
        out = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
    else:
        out = cropped
    return out[...,0:3], out[...,3:]
'''
def preprocess(images, depths, crop_size, depth_size=(220,296), scale_range=(0.5,1.5), rot_range=(-5,5), flip=True):
    image,depth = augment(images, depths, crop_size, scale_range, rot_range, flip)
    depth = tf.image.resize_images(depth, depth_size)
    return image*2/255-1, depth*2/255-1
'''
def preprocess(images, depths, crop_size, depth_size=(220,296), scale_range=(0.5,1.5), rot_range=(-5,5), flip=True):
    image,depth = augment(images, depths, crop_size, scale_range, rot_range, flip)
    depth = tf.image.resize_images(depth, depth_size)
    return image/255, tf.log(depth)

def preprocess_test(images, depths, crop_size, depth_size=(220,296), scale_range=(0.5,1.5), rot_range=(-5,5), flip=True):
    #image,depth = augment(images, depths, crop_size, scale_range, rot_range, flip)
    depth = tf.image.resize_images(depths, depth_size)
    image = tf.image.resize_images(images, crop_size)
    #depth = tf.image.resize_images(depth, depth_size)
    return image/255, tf.log(depth)


# In[7]:


INIT_LR = 0.0001
DECAY_SPEED = 1/20

tf.reset_default_graph() 

global_step = tf.Variable(0.0, trainable=False)
'''
with tf.name_scope('Reader'):
    reader = Loader(imagepath='./nyu_data/*.jpg', depthpath='./nyu_data/*.png', batch_size=4, seed=0)
    train_img_reader, train_depth_reader = reader.get_train_reader()
    test_img_reader, test_depth_reader = reader.get_test_reader()
'''
with tf.name_scope('Reader'):
    train_reader, test_reader = dtf.load_data(traindata='./NYUv2_train.tfrecords',testdata = './NYUv2_test.tfrecords')
    #reader = Loader(imagepath='./data/nyu_datasets/*.jpg', depthpath='./data/nyu_datasets/*.png', batch_size=4, seed=0)
    train_img_reader, train_depth_reader = train_reader
    test_img_reader, test_depth_reader = test_reader
    train_img_reader,test_img_reader = tf.cast(train_img_reader,tf.float32),tf.cast(test_img_reader,tf.float32)
with tf.name_scope('Preprocess'):
    train_imgs, train_depths = preprocess(train_img_reader, train_depth_reader, crop_size=[228,304], depth_size=[220,296])#[55,74])
    test_imgs, test_depths = preprocess(test_img_reader, test_depth_reader, crop_size=[228,304], depth_size=[220,296], scale_range=(0.475,0.475), rot_range=(0,0), flip=False)
with tf.name_scope('Parameters'):
    fine_gate = tf.placeholder_with_default(True,shape=[])
    learning_rate = INIT_LR/(1+DECAY_SPEED*global_step)
    is_training = tf.placeholder_with_default(False, [])
with tf.name_scope('Model'):
    img_input = tf.placeholder(tf.float32,[None,228,304,3])
    depth_input = tf.placeholder(tf.float32,[None,220,296,1])
    coarse_output = coarse(img_input, 0.3, is_training)
    refined_output = fine(img_input, coarse_output, fine_gate, 0.3, is_training)
with tf.name_scope('Loss'):
    loss_op = redefine_loss(refined_output, depth_input)
with tf.name_scope('Adam'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op)
#print(refined_output.shape,depth_input.shape)
with tf.name_scope('Accuracy'):
    accuracy_op = accuracy(refined_output, depth_input, 1.25)

running_loss_sum = tf.placeholder(tf.float32, [])
running_accuracy_sum = tf.placeholder(tf.float32, [])
input_image_sum = tf.placeholder(tf.float32, [1,228,304,3])
input_depth_sum = tf.placeholder(tf.float32, [1,220,296,1])
coarse_sum = tf.placeholder(tf.float32, [1,55,74,1])
refined_sum = tf.placeholder(tf.float32, [1,220,296,1])

train_summary = tf.summary.merge([
    tf.summary.scalar('train_running_loss', running_loss_sum),
    tf.summary.scalar('train_running_accuracy', running_accuracy_sum),
    tf.summary.scalar('learning_rate', learning_rate),
    tf.summary.image('train_input', input_image_sum),
    tf.summary.image('train_target', input_depth_sum),
    tf.summary.image('train_coarse', coarse_sum),
    tf.summary.image('train_output', refined_sum)
])
test_summary = tf.summary.merge([
    tf.summary.scalar('test_loss', running_loss_sum),
    tf.summary.scalar('test_accuracy', running_accuracy_sum),
    tf.summary.image('test_input', input_image_sum),
    tf.summary.image('test_target', input_depth_sum),
    tf.summary.image('test_coarse', coarse_sum),
    tf.summary.image('test_output', refined_sum)
])


# In[8]:


def normalize(image):
    img = image-np.min(image)
    return (img)/np.max(img)


# In[9]:


from datetime import datetime
from PIL import Image
now = datetime.now()

Nth_RUN = 'NOBN_linear_augment_0003'
training_epochs=1000
logs_path = './logs/{}'.format(Nth_RUN)
saver = tf.train.Saver()


Train_loss=[]
Train_acc=[]
Test_loss=[]
Test_acc=[]


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    tf.global_variables_initializer().run(session=sess)
    tf.local_variables_initializer().run(session=sess)
    '''
    latest_ckpt = tf.train.latest_checkpoint(logs_path)
    if latest_ckpt:
        saver.restore(sess, latest_ckpt)
        print('Checkpoint recovered', latest_ckpt)
    '''
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # start queue loader (for data)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    # Training cycle
    training_coarse = 2
    while sess.run(tf.assign_add(global_step, 1)) < training_epochs:
        print(sess.run(global_step), end=' ')
        train_loss = 0
        train_accu = 0
        for s in range(reader.n_batches()[0]):
            batch_img, batch_depth = sess.run([train_imgs, train_depths])
            _, cur_loss, cur_accu = sess.run([train_op, loss_op, accuracy_op], feed_dict={img_input:batch_img, 
                                                           depth_input:batch_depth, 
                                                           fine_gate:not training_coarse, 
                                                           is_training:True})
            #print(cur_loss)
            #print(cur_accu)
            training_coarse = training_coarse-1
            if training_coarse<0:
                training_coarse = 2
            #print(reader.size())
            train_loss += cur_loss/reader.size()[0]*batch_img.shape[0]#/reader.n_batches()[0]
            train_accu += cur_accu/reader.size()[0]*batch_img.shape[0]#/reader.n_batches()[0]
            #print(cur_loss)
            #print(cur_accu)
        #print(train_loss)
        #print(train_accu)
        Train_acc.append(train_accu)
        Train_loss.append(train_loss)
        I,D = sess.run([train_imgs, train_depths])
        C,O = sess.run([coarse_output, refined_output], feed_dict={img_input:I, depth_input:D, is_training:False})
        
        summary_writer.add_summary(
            sess.run(train_summary, 
                     feed_dict={
                         running_loss_sum:train_loss,
                         running_accuracy_sum:train_accu,
                         input_image_sum:I[-1:,...], 
                         input_depth_sum:D[-1:,...], 
                         coarse_sum:C[-1:,...], 
                         refined_sum:O[-1:,...]}), global_step=sess.run(global_step))
        
        
        test_loss = 0
        test_accu = 0
        for s in range(reader.n_batches()[1]):
            tbatch_img, tbatch_depth = sess.run([test_imgs, test_depths])
            cur_loss, cur_accu = sess.run([loss_op, accuracy_op], 
                                  feed_dict={
                                      img_input:tbatch_img,
                                      depth_input:tbatch_depth,
                                      is_training:False})
            test_loss += cur_loss/reader.size()[1]*tbatch_img.shape[0]#/reader.n_batches()[1]
            test_accu += cur_accu/reader.size()[1]*tbatch_img.shape[0]#/reader.n_batches()[1]
            #print(cur_accu)
        print(test_loss)
        print(test_accu)
        TI, TD = sess.run([test_imgs, test_depths])
        TC, TO = sess.run([coarse_output, refined_output], feed_dict={img_input:TI,depth_input:TD, is_training:False})
        plt.figure(figsize=(8, 6))
        plt.subplot(131)
        test_in = normalize(TI[1])
        plt.imshow(np.reshape(test_in,[test_in.shape[0],test_in.shape[1],3]))
        plt.subplot(132)
        pred = normalize(TO[1])
        plt.imshow(np.reshape(pred,[pred.shape[0],pred.shape[1]]))
        plt.subplot(133)
        target = normalize(TD[1])
        plt.imshow(np.reshape(target,[target.shape[0],target.shape[1]]))
        plt.show()
        #print(TO.min(),TO.max())
        #print(TD.min(),TD.max())
        #print()
        #print(test_loss.shape)
        #print(cur_accu.shape)
        Test_acc.append(test_accu)
        Test_loss.append(test_loss)
        
        summary_writer.add_summary(
            sess.run(test_summary, 
                     feed_dict={
                         running_loss_sum:test_loss, 
                         running_accuracy_sum:cur_accu,
                         input_image_sum:TI[-1:,...], 
                         input_depth_sum:TD[-1:,...], 
                         coarse_sum:TC[-1:,...], 
                         refined_sum:TO[-1:,...]}), global_step=sess.run(global_step))
        
        if sess.run(global_step)%10==0:
            save_path = saver.save(sess, logs_path+"/CSE291FinalModel_test1.ckpt", global_step=global_step)  #save model for second part
    # end queue loader
    coord.request_stop()
    coord.join(threads)


# In[ ]:


from datetime import datetime
from PIL import Image
now = datetime.now()

Nth_RUN = 'NOBN_linear_augment_0001'
training_epochs=1000
logs_path = './logs/{}'.format(Nth_RUN)
saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    tf.global_variables_initializer().run(session=sess)
    tf.local_variables_initializer().run(session=sess)
    '''
    latest_ckpt = tf.train.latest_checkpoint(logs_path)
    if latest_ckpt:
        saver.restore(sess, latest_ckpt)
        print('Checkpoint recovered', latest_ckpt)
    '''
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # start queue loader (for data)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    # Training cycle
    training_coarse = 2
    while sess.run(tf.assign_add(global_step, 1)) < training_epochs:
        print(sess.run(global_step), end=' ')
        train_loss = 0
        train_accu = 0
        for s in range(reader.n_batches()[0]):
            batch_img, batch_depth = sess.run([train_imgs, train_depths])
            _, cur_loss, cur_accu = sess.run([train_op, loss_op, accuracy_op], feed_dict={img_input:batch_img, 
                                                           depth_input:batch_depth, 
                                                           fine_gate:not training_coarse, 
                                                           is_training:True})
            #print(cur_loss)
            #print(cur_accu)
            training_coarse = training_coarse-1
            if training_coarse<0:
                training_coarse = 2
            #print(reader.size())
            train_loss += cur_loss/reader.size()[0]*batch_img.shape[0]#/reader.n_batches()[0]
            train_accu += cur_accu/reader.size()[0]*batch_img.shape[0]#/reader.n_batches()[0]
            #print(cur_loss)
            #print(cur_accu)
        #print(train_loss)
        #print(train_accu)
        I,D = sess.run([train_imgs, train_depths])
        C,O = sess.run([coarse_output, refined_output], feed_dict={img_input:I, depth_input:D, is_training:False})
        
        summary_writer.add_summary(
            sess.run(train_summary, 
                     feed_dict={
                         running_loss_sum:train_loss,
                         running_accuracy_sum:train_accu,
                         input_image_sum:I[-1:,...], 
                         input_depth_sum:D[-1:,...], 
                         coarse_sum:C[-1:,...], 
                         refined_sum:O[-1:,...]}), global_step=sess.run(global_step))
        
        
        test_loss = 0
        test_accu = 0
        for s in range(reader.n_batches()[1]):
            tbatch_img, tbatch_depth = sess.run([test_imgs, test_depths])
            cur_loss, cur_accu = sess.run([loss_op, accuracy_op], 
                                  feed_dict={
                                      img_input:tbatch_img,
                                      depth_input:tbatch_depth,
                                      is_training:False})
            test_loss += cur_loss/reader.size()[1]*tbatch_img.shape[0]#/reader.n_batches()[1]
            test_accu += cur_accu/reader.size()[1]*tbatch_img.shape[0]#/reader.n_batches()[1]
            #print(cur_accu)
        print(test_loss)
        print(test_accu)
        TI, TD = sess.run([test_imgs, test_depths])
        TC, TO = sess.run([coarse_output, refined_output], feed_dict={img_input:TI,depth_input:TD, is_training:False})
        plt.figure(figsize=(8, 6))
        plt.subplot(131)
        test_in = normalize(TI[1])
        plt.imshow(np.reshape(test_in,[test_in.shape[0],test_in.shape[1],3]))
        plt.subplot(132)
        pred = normalize(TO[1])
        plt.imshow(np.reshape(pred,[pred.shape[0],pred.shape[1]]))
        plt.subplot(133)
        target = normalize(TD[1])
        plt.imshow(np.reshape(target,[target.shape[0],target.shape[1]]))
        plt.show()
        #print(TO.min(),TO.max())
        #print(TD.min(),TD.max())
        #print()
        #print(test_loss.shape)
        #print(cur_accu.shape)
        summary_writer.add_summary(
            sess.run(test_summary, 
                     feed_dict={
                         running_loss_sum:test_loss, 
                         running_accuracy_sum:cur_accu,
                         input_image_sum:TI[-1:,...], 
                         input_depth_sum:TD[-1:,...], 
                         coarse_sum:TC[-1:,...], 
                         refined_sum:TO[-1:,...]}), global_step=sess.run(global_step))
        
        if sess.run(global_step)%10==0:
            save_path = saver.save(sess, logs_path+"/CSE291FinalModel_test1.ckpt", global_step=global_step)  #save model for second part
    # end queue loader
    coord.request_stop()
    coord.join(threads)


# In[8]:


compile()


# In[25]:


from PIL import Image
import matplotlib.image as mpimg
path = './nyu_data/'
for filename in os.listdir(path):
    print(filename)
    img=mpimg.imread(path+filename)#.convert("RGB")
    
    plt.imshow(img)
    plt.show()


# In[26]:


img


# In[22]:


img.load()


# In[ ]:


from tqdm import tqdm
import numpy as np
import tensorflow as tf
path = './nyu_depth_v2_labeled.mat'
f = h5py.File(path)
writer = tf.python_io.TFRecordWriter("NYU.tfrecords")
for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
        example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
          # Features contains a map of string to Feature proto objects
          feature={
            # A Feature contains one of either a int64_list,
            # float_list, or bytes_list
            'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=depth)),
            'image': tf.train.Feature(
                int64_list=tf.train.Int64List(value=image)),
    }))
        serialized = example.SerializeToString()
        writer.write(serialized)

