#!usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import math, random
import matplotlib.pyplot as plt

## Joseph McGill
## Artificial Neural Network
## Function approximation f(x)
## Data was generated using generate_data.py
##
## This program uses artificial neural networks to approximate the funcion used
## to generate the data passed in as the command-argument. The net has 1 hidden
## layer with 10 nodes using the sigmoid activation function.

#### Some of this code was taken from a tutorial for TensorFlow and modified
#### Credit goes to Delip Rao
#### https://github.com/delip/blog-stuff/blob/master/tensorflow_ufp.ipynb


# data file
infile = './data/xy_data.csv'

# use the data file from the command line argument (if it exists)
if len(sys.argv) > 1:
    infile = sys.argv[1]

# constants used for the ANN
NUM_HIDDEN = 10
NUM_EPOCHS = 1000
NUM_EXAMPLES = 1000
LEARNING_RATE = 0.1
TRAIN_SPLIT = .6
TEST_SPLIT = .2

#### Approximate f(x) without noise ####
# read in the values for x
all_x = np.genfromtxt(infile, dtype=np.float32, delimiter=',',
                      usecols=(0))

all_x = np.reshape(all_x, (len(all_x), 1))

# read in the values for y
all_y = np.genfromtxt(infile, dtype=np.float32, delimiter=',',
                      usecols=(1))
all_y = np.reshape(all_y, (len(all_y), 1))

# split the data sets into train/validation/test sets
train_size = int(NUM_EXAMPLES * TRAIN_SPLIT)
test_size = int(NUM_EXAMPLES * TEST_SPLIT)

trainx = all_x[:train_size]
validx = all_x[train_size:train_size + test_size]
testx = all_x[train_size + test_size:]

trainy = all_y[:train_size]
validy = all_y[train_size:train_size + test_size]
testy = all_y[train_size + test_size:]

# declare X and Y placeholders
X = tf.placeholder(tf.float32, [None, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

# initialize the input weights and bias weights
# random input weight, zero bias weight
w_h = tf.Variable(tf.random_normal([1, NUM_HIDDEN], stddev=0.01,
                  dtype=tf.float32))

b_h = tf.Variable(tf.zeros([1, NUM_HIDDEN], dtype=tf.float32))

# the activation function of each hidden neuron (sigmoid)
h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)

# initialize the output weights and bias weights
# random output weight, zero bias weight
w_o = tf.Variable(tf.random_normal([NUM_HIDDEN, 1], stddev=0.01,
                  dtype=tf.float32))
b_o = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))

# the activation function of the output node (linear)
yhat = tf.matmul(h, w_o) + b_o

# function to calculate the mse
mean_sq_err = lambda x, xhat: tf.reduce_mean(tf.square(tf.sub(x, xhat)))

# operation used for training the net (gradient descent)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
                                                        mean_sq_err(Y, yhat))

# create the session and initialize the tf variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# run the input data through the ANN some number of times
errors = []
for i in range(NUM_EPOCHS):

    # train the ANN using the input and output data
    sess.run(train_op, feed_dict={X: trainx, Y: trainy})

    # calculate the validation MSE
    mse = sess.run(tf.reduce_mean(tf.square(tf.sub(validy, yhat))),
                                  feed_dict={X:validx})
    errors.append(mse)

# print the mean squared error for the
# training, validation, and test set
mse = sess.run(mean_sq_err(trainy, yhat), feed_dict={X:trainx})
print("MSE (train): %f") % mse

mse = sess.run(mean_sq_err(validy, yhat), feed_dict={X:validx})
print("MSE (validation): %f") % mse

mse = sess.run(mean_sq_err(testy, yhat), feed_dict={X:testx})
print("MSE (test): %f") % mse

# plot the mse of the validation set over the epochs
plt.plot(errors, 'ro')
plt.xlabel('Number of Epochs')
plt.ylabel('MSE (validation)')
plt.title('Number of Epochs vs MSE (validation)')
plt.show()

# plot the output of the test set
output = sess.run(yhat, feed_dict={X:testx})
plt.plot(testx, output, 'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test set input vs Predicted output')
plt.show()
