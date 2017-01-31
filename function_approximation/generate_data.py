#!usr/bin/env pthon

import csv
import numpy as np

# set the RNG seed
np.random.seed(2016)

NUM_EXAMPLES = 1000
Y_NOISE_MEAN = 0
Y_NOISE_VAR = 5
Z_NOISE_MEAN = 0
Z_NOISE_VAR = 50

# function we want to learn
function_to_learn = lambda x: 10*np.sin(x) + x**2

# generate x and y (without noise) and save them to a csv
x = np.random.uniform(-5, 5, (1, NUM_EXAMPLES)).T
np.random.shuffle(x)
y = function_to_learn(x)
vals = np.concatenate((x, y), axis=1)
np.savetxt("data/xy_data.csv", vals, delimiter=",")

# generate x and y (with noise) and save them to a csv
x = np.random.uniform(-5, 5, (1, NUM_EXAMPLES)).T
np.random.shuffle(x)
y = function_to_learn(x)
y = y.T + np.random.normal(Y_NOISE_MEAN, np.sqrt(Y_NOISE_VAR), len(y))
y = y.T
vals = np.concatenate((x, y), axis=1)
np.savetxt("data/xy_noise_data.csv", vals, delimiter=",")


# function we want to learn
function_to_learn = lambda x,y: 10*np.sin(x) + y**2

# generate x, y, and z (without noise) and save them to a csv
x = np.random.uniform(-5, 5, (1, NUM_EXAMPLES)).T
y = np.random.uniform(-10, 10, (1, NUM_EXAMPLES)).T
np.random.shuffle(x)
np.random.shuffle(y)
z = function_to_learn(x, y)
vals = np.concatenate((x, y), axis=1)
vals = np.concatenate((vals, z), axis=1)
np.savetxt("data/xyz_data.csv", vals, delimiter=",")

# generate x, y, and z (with noise) and save them to a csv
x = np.random.uniform(-5, 5, (1, NUM_EXAMPLES)).T
y = np.random.uniform(-10, 10, (1, NUM_EXAMPLES)).T
np.random.shuffle(x)
np.random.shuffle(y)
z = function_to_learn(x, y)
z = z.T + np.random.normal(Z_NOISE_MEAN, np.sqrt(Z_NOISE_VAR), len(z))
z = z.T
vals = np.concatenate((x, y), axis=1)
vals = np.concatenate((vals, z), axis=1)
np.savetxt("data/xyz_noise_data.csv", vals, delimiter=",")
