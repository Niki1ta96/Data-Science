import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import losses

os.chdir("//home//tensorflow//")
# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

# Learning some basics of neural networks
#########################################

# one-hot-encoding in action
x1 = np.array([0,1,2,1,1,1,1,0])
temp = tf.one_hot(x1,3,1,0) # one-hot(target, #classes, existence, non-existence)
tf.Session().run(temp)

# sigmoid output. sigmoid(x) = 1/ [1 + e^(-x)]. Range is (0,1)
x2 = np.array([10,100,0,-10,-100],dtype=np.float32)
sigmoid = tf.sigmoid(x2, name ='X2')
tf.Session().run(sigmoid)
#array([  9.99954581e-01,   1.00000000e+00,   5.00000000e-01,
#        4.53978719e-05,   0.00000000e+00], dtype=float32)

# softmax output. softmax(aj) = e^aj/ (sigma(k=1 to k) e^ak)
# transforms output of 4 perceptrons into probability distribution
x3 = np.array([0.6,0.7,0.9,0.9], dtype=np.float32)
softmax = tf.nn.softmax(x3, name ='SoftMax')
tf.Session().run(softmax)
#array([ 0.20812137,  0.23000968,  0.28093445,  0.28093445], dtype=float32)

# Fully Connected:-  all inputs are connected to every perceptron
# default activation function is 'Linear'
# output is (4,8)
# 4 = 1(1) + 2(1) + 1;
#8 = 3(1) + 4(1) + 1

features = np.array([[1,2],[3,4]], dtype = np.float)
nn_fully_connected = layers.fully_connected(inputs=features,
                                            weights_initializer=tf.constant_initializer([1.0]),
                                            biases_initializer=tf.constant_initializer([1.0]),
                                            num_outputs=1,
                                            activation_fn=None)

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nn_fully_connected)
#array([[ 4.],
#       [ 8.]])

# output with SIGMOID FUNCTION: sigmoid(4) and sigmoid(8)
nn_fully_connected_sigmoid = layers.fully_connected(inputs= features, # [[1,2], [3,4]]
                                                    weights_initializer= tf.constant_initializer([1.0]),
                                                    biases_initializer=tf.constant_initializer([1.0]),
                                                    num_outputs=1,
                                                    activation_fn=tf.sigmoid)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nn_fully_connected_sigmoid)
#array([[ 0.98201379],
#      [ 0.99966465]])

# layer of 2 perceptrons. Both of them give the same output of (4,8) and (4,8)
nn_fully_connected_2 = layers.fully_connected(inputs=features, # [[1,2], [3,4]]
                                              weights_initializer=tf.constant_initializer([1.0]),
                                              biases_initializer=tf.constant_initializer([1.0]),
                                              num_outputs=2,
                                              activation_fn=None)

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nn_fully_connected_2)
#array([[ 4.,  4.],
#       [ 8.,  8.]])

# individual weights to each perceptron. Perceptron1 (w11=1,w21=2,bias=1) Perceptron2 (w12=1,w22=2,bias=2)
# output is [6,7] and [12,13]
# Input: 1,2  -->    6 = 1(1) + 2(2) + 1;  7 = 1(1) + 2(2) + 2
# Input: 3,4  -->   12 = 3(1) + 4(2) + 1; 13 = 3(1) + 4(2) + 2

nn_fully_connected_3 = layers.fully_connected(inputs= features, ## [[1,2], [3,4]]
                                               # [[w11,w12],[w21,w22]]; 
                                               #wij = weight from ith input to jth perceptron
                                              weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
                                              biases_initializer=tf.constant_initializer([1.0,2.0]),
                                              num_outputs=2,
                                              activation_fn=None)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nn_fully_connected_3)
#array([[  6.,   7.],
#       [ 12.,  13.]])

## SAME Presceptron with Sigmoid Activation function
nn_fully_connected_4 = layers.fully_connected(inputs=features,
                                              weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
                                              biases_initializer=tf.constant_initializer([1.0,2.0]),
                                              num_outputs=2,
                                              activation_fn=tf.sigmoid)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nn_fully_connected_4)
#array([[ 0.99752738,  0.99908895],
#       [ 0.99999386,  0.99999774]])

# softmax output. Gives global meaning to both the perceptrons combined
# output is [0.26,0.73], [0.26,0.73].
# [e^6/(e^6+e^7),e^7/(e^6 + e^7) ] = [1/1+e, 1/1+e^-1] = [0.26,0.73]
# [e^12/(e^12+e^13),e^13/(e^12 + e^13) ] = [1/1+e, 1/1+e^-1] = [0.26,0.73]
# Note how the result is very different from that obtained with sigmoid activation function.

nn_fully_connected_5 = layers.fully_connected(inputs= features,
                                              weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
                                              biases_initializer=tf.constant_initializer([1.0,2.0]),
                                              num_outputs=2,
                                              activation_fn=tf.nn.softmax)

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nn_fully_connected_5)
#array([[ 0.26894142,  0.73105858],
#     [ 0.26894142,  0.73105858]])

#Losses Sq_loss, log_loss
outputs = tf.constant([0,1,0,1])
targets = tf.constant([1,1,1,0])
sq_loss = losses.mean_squared_error(outputs,targets)
log_loss = losses.log_loss(outputs,targets)

outputs_2 = tf.constant([[100.0,-100.0,-100.0],
                        [-100.0,100.0,-100.0],
                        [-100.0,-100.0,100.0]])

targets_2 = tf.constant([[0,0,1],
                         [1,0,0],
                         [0,1,0]])

sq_loss2 = losses.mean_squared_error(outputs_2,targets_2)

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(sq_loss)   #0.75  = [(0-1)^2 + (1-1)^2 + (0-1)^2 + (1-0)^2] / 4
session.run(log_loss)  #12.088572 = (6*100^2 + 3*101^2)/9
session.run(sq_loss2)  #10067.0 = sigma(-y_i*log(y_i))

##############################################################################################




