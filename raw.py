# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 00:53:25 2017

@author: viveksagar
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def create_attributes(calcium, spikes, num_history = 45, num_fut = 1, offset_step=1, num_der = 2, integ_kernel = 100):
# num_history = number of points in the past used as attribute, 
# num_fut = number of points in the future taken as attribute, offset_step = distance between history/future points
# num_der = maximum order of derivative, integ_kernel  = window size of running cumsum 
    print("Creating additional attributes...")
    cal_past = np.asarray([np.roll(np.squeeze(calcium),(ii+1)*offset_step) for ii in range(num_history)]).T # History
    cal_fut = np.asarray([np.roll(np.squeeze(calcium),-(ii+1)*offset_step) for ii in range(num_fut)]).T # Future
    cal_grad = np.asarray([np.append(np.diff(np.squeeze(calcium),ii+1),np.zeros(ii+1)) for ii in range(num_der)]).T # n-Gradients upto n-order      
    cal_integ =  np.cumsum(calcium) # Cum-sum
    cal_integ = (cal_integ[integ_kernel:]-cal_integ[:-integ_kernel])/integ_kernel # Moving Avg
    cal_avg =  np.asarray([np.lib.pad(cal_integ, (len(calcium)-len(cal_integ),0), 'edge')]).T # Padded with additional values to maintain original size   
    att_data_raw = np.concatenate((calcium,cal_past,cal_fut,cal_grad, cal_avg), axis = 1)     
    mask = np.ones(len(att_data_raw),dtype=bool) # Clip the edges with inaccurate attributes
    mask[:integ_kernel], mask[-num_fut*offset_step:] = False,False
    att_data = att_data_raw[mask]
    clipped_spikes = spikes[mask].astype(np.float64)    
    return att_data, clipped_spikes
    
def one_hot_label(label):
# One hot representation of labels for cross entropy measurement
    print("Using onehot representation")
    label_onehot = np.concatenate((label,1-label),axis=1)
    return label_onehot
    
def pre_train(att_data,clipped_spikes,batch_size):
#     Basic preprocessing prior to training: scaling, crossvalidation split, clipping the size to match batch size.
    att_data = preprocessing.scale(att_data)
    train_x, test_x, train_y, test_y = train_test_split(att_data, clipped_spikes, test_size=0.2)
    split_tr = (len(train_x)-np.mod(len(train_x),batch_size)).astype(np.int64)
    split_test = (len(test_x)-np.mod(len(test_x),batch_size)).astype(np.int64) 
    mask_tr = np.ones(len(train_x),dtype=bool) # Clip the edges to have the len of data a multiple of batch_size
    mask_test = np.ones(len(test_x),dtype=bool)
    mask_tr[split_tr:], mask_test[split_test:] = False,False
    train_x = train_x[mask_tr]
    train_y = train_y[mask_tr]
    train_y = one_hot_label(train_y)
    test_x = test_x[mask_test]
    test_y = test_y[mask_test]
    test_y = one_hot_label(test_y)
    return train_x, test_x, train_y, test_y
    
# RNN
    
hm_epochs = 10
n_classes = 2
batch_size = 128
chunk_size = 5
n_chunks = 10
rnn_size = 64

x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}
    x = tf.unstack(x, n_chunks, 1)
    lstm_cell = rnn.BasicLSTMCell(rnn_size,forget_bias=1.0)
    
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    return output
    
def train_neural_network(att_data, clipped_spikes):
    [train_x, test_x, train_y, test_y]=pre_train(att_data,clipped_spikes,batch_size)
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch+1, 'completed out of ', hm_epochs,'loss:',epoch_loss)
            test_x = test_x.reshape((-1,n_chunks,chunk_size))
            spike_predict= prediction.eval({x:test_x})
    tf.reset_default_graph()
    return test_y, spike_predict            

def main(calcium,spikes):
#    Provide inputs as 1D column vector of shape (N,1)
    [att_data,clipped_spikes] = create_attributes(calcium,spikes)
    [test_spikes, predicted_spikes]= train_neural_network(att_data,clipped_spikes)
    return test_spikes, predicted_spikes
    
