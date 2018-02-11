import numpy as np
import scipy.io as sio
import tensorflow as tf
from cd1 import cd1
from showRbm import showWeights

dataSet = sio.loadmat("/home/student/Documents/Neural networks for Machine Learning/assignment4/data_set.mat")

learningRate = 0.9
momentumSpeedConstant = 0.9
miniBatchSize = 100
numberOfHiddenUnits = 50

inputs = dataSet["data"]["training"][0, 0]["inputs"][0, 0].astype("float32")
# inputs = np.reshape(dataSet["data"]["training"][0, 0]["inputs"][0, 0].astype("float32")[:, 0:2], (256, 2))

targets = dataSet["data"]["training"][0, 0]["targets"][0, 0].astype("float32")

inputsConstant = tf.placeholder(dtype="float32", shape=inputs.shape, name="inputs")
weights = tf.Variable(initial_value=np.random.random_sample((numberOfHiddenUnits, inputs.shape[0])).astype("float32"),
                      name="weights")

weightSummary = tf.summary.image('input', tf.reshape(weights[0, :], [-1, 16, 16, 1]), 1)
writer = tf.summary.FileWriter("../model_output/logs2")

momentumSpeed = tf.Variable(np.zeros(weights.shape).astype("float32"), dtype="float32")
startOfNextMiniBatch = 0

miniBatch = tf.placeholder(dtype="float32", shape=[inputs.shape[0], miniBatchSize], name="miniBatch")
gradient = cd1(miniBatch, weights)

momentumSpeedUpdated = tf.add(tf.multiply(tf.constant(momentumSpeedConstant, dtype="float32"), momentumSpeed), gradient)
weightsUpdated = tf.add(weights, tf.multiply(momentumSpeedUpdated, tf.constant(learningRate, dtype="float32")))

with tf.Session() as sess:
    momentumSpeed.initializer.run()
    weights.initializer.run()

    for i in range(1000):
        miniBatchValues = inputs[:, startOfNextMiniBatch: (startOfNextMiniBatch + miniBatchSize)]
        startOfNextMiniBatch = np.mod(startOfNextMiniBatch + miniBatchSize, inputs.shape[1])

        updatedSpeed, updatedWeights = sess.run([momentumSpeedUpdated, weightsUpdated], feed_dict={miniBatch: miniBatchValues})

        updateMomentumSpeed = momentumSpeed.assign(updatedSpeed)
        updateWeights = weights.assign(updatedWeights)

        sess.run([updateMomentumSpeed, updateWeights])

        # print("i: ", i)

        summary_str = sess.run(weightSummary)

        writer.add_summary(summary_str, i)


    finalWeights = sess.run(weights)
    showWeights(finalWeights)

