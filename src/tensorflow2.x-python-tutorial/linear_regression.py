'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

tf.compat.v1.disable_eager_execution()

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 10

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

if True:
    # tf Graph Input
    X = tf.compat.v1.placeholder("float")
    Y = tf.compat.v1.placeholder("float")

    # Set model weights
    W = tf.Variable(-0.06, name="weight")
    b = tf.Variable(-0.73, name="bias")

    # Construct a linear model
    mul = tf.multiply(X, W)
    pred = tf.add(mul, b)

    # Mean squared error
    sub = pred - Y
    pow = tf.pow(sub, 2)
    reduce = tf.reduce_sum(pow)
    cost = reduce / (2 * n_samples)
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    grad = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    optimizer = grad.minimize(cost)
    # tf.train.export_meta_graph(filename='save_model.meta');
else:
    # tf Graph Input
    new_saver = tf.train.import_meta_graph("linear_regression.meta")
    nodes = tf.get_default_graph()._nodes_by_name;
    optimizer = nodes["GradientDescent"]
    cost = nodes["truediv"].outputs[0]
    X = nodes["Placeholder"].outputs[0]
    Y = nodes["Placeholder_1"].outputs[0]
    W = nodes["weight"].outputs[0]
    b = nodes["bias"].outputs[0]
    pred = nodes["Add"].outputs[0]

# Initialize the variables (i.e. assign their default value)
init = tf.compat.v1.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()