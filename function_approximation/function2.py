#!usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import math, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Joseph McGill
## Artificial Neural Network
## Function approximation f(x, y)
## Data was generated using generate_data.py
##
## This program uses artificial neural networks to approximate the funcion used
## to generate the data passed in as the command-argument. The net has 1 hidden
## layer with 20 nodes using the sigmoid activation function.

#### Some of this code was taken from a tutorial for TensorFlow and modified
#### Credit goes to Delip Rao
#### https://github.com/delip/blog-stuff/blob/master/tensorflow_ufp.ipy

# data file
infile = './data/xyz_data.csv'

# use the data file from the command line argument (if it exists)
if len(sys.argv) > 1:
    infile = sys.argv[1]

# constants used for the ANN
NUM_HIDDEN = 20
NUM_EPOCHS = 1000
NUM_EXAMPLES = 1000
LEARNING_RATE = 0.01
TRAIN_SPLIT = .6
TEST_SPLIT = .2

#### Approximate f(x, y) without noise ####
# read in the values for x and y
all_xy = np.genfromtxt(infile, dtype=np.float32, delimiter=',', usecols=(0, 1))
all_xy = np.reshape(all_xy, (len(all_xy), 2))

# read in the values for z
all_z = np.genfromtxt(infile, dtype=np.float32, delimiter=',', usecols=(2))
all_z = np.reshape(all_z, (len(all_z), 1))

# split the data sets into train/validation/test sets
train_size = int(NUM_EXAMPLES * TRAIN_SPLIT)
test_size = int(NUM_EXAMPLES * TEST_SPLIT)

trainxy = all_xy[:train_size]
validxy = all_xy[train_size:train_size + test_size]
testxy = all_xy[train_size + test_size:]

trainz = all_z[:train_size]
validz = all_z[train_size:train_size + test_size]
testz = all_z [train_size + test_size:]

# declare X and Z placeholders
X = tf.placeholder(tf.float32, [None, 2], name="X")
Z = tf.placeholder(tf.float32, [None, 1], name="Z")

# initialize the input weights (for x and y) and the bias weights
# random input weights, zero bias weight
w_h = tf.Variable(tf.random_normal([2, NUM_HIDDEN], stddev=0.01,
                  dtype=tf.float32))
b_h = tf.Variable(tf.zeros([1, NUM_HIDDEN], dtype=tf.float32))

# the activation function of each hidden node (sigmoid)
h = tf.nn.sigmoid(tf.matmul(X, w_h)  + b_h)

# initialize the output weights and bias weights
# random output weight, zero bias weight
w_o = tf.Variable(tf.random_normal([NUM_HIDDEN, 1], stddev=0.01,
                  dtype=tf.float32))
b_o = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))

# the activation function of the output node (linear)
zhat = tf.matmul(h, w_o) + b_o

# function to calculate the mse
mean_sq_err = lambda x, xhat: tf.reduce_mean(tf.square(tf.sub(x, xhat)))

# operation used for training the net (gradient descent)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
                                                         mean_sq_err(Z, zhat))

# create the session and initialize the tf variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# run the input data through the ANN some number of times
errors = []
for i in range(NUM_EPOCHS):

    # train the ANN using the input and output data
    sess.run(train_op, feed_dict={X:trainxy, Z:trainz})

    # calculate the validation MSE
    mse = sess.run(tf.reduce_mean(tf.square(tf.sub(validz, zhat))),
                                            feed_dict={X:validxy})
    errors.append(mse)

# print the mean squared error for the
# training, validation, and test set
mse = sess.run(mean_sq_err(trainz, zhat), feed_dict={X:trainxy})
print("MSE (train): %f") % mse

mse = sess.run(mean_sq_err(validz, zhat), feed_dict={X:validxy})
print("MSE (validation): %f") % mse

mse = sess.run(mean_sq_err(testz, zhat), feed_dict={X:testxy})
print("MSE (test): %f") % mse

# plot the mse of the validation set over the epochs
plt.plot(errors, 'ro')
plt.xlabel('Number of Epochs')
plt.ylabel('MSE (validation)')
plt.title('Number of Epochs vs MSE (validation)')
plt.show()

# plot the wireframe of the test set
output = sess.run(zhat, feed_dict={X:testxy})
cols = np.split(testxy, 2, axis=1)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Test set input vs Predicted output')
surf = ax.scatter(cols[0], cols[1], output, zdir='z')
plt.show()
