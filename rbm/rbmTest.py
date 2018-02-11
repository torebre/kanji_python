import numpy as np
import scipy.io as sio
import tensorflow as tf
from cd1 import cd1

dataSet = sio.loadmat("/home/student/Documents/Neural networks for Machine Learning/assignment4/data_set.mat")


inputs = dataSet["data"]["training"][0, 0]["inputs"][0, 0].astype("float32")
targets = dataSet["data"]["training"][0, 0]["targets"][0, 0].astype("float32")

numberOfHiddenUnits = 50
rbmWeights = np.random.random_sample((numberOfHiddenUnits, inputs.shape[0])).astype("float32")


inputsConstant = tf.placeholder(dtype="float32", shape = inputs.shape, name="inputs")
weights = tf.Variable(initial_value=rbmWeights, name="weights")

# outputTensor = cd1(inputs, rbmWeights)


# weightSummary = tf.summary.image("weights", weights)
weightSummary = tf.summary.image('input', tf.reshape(weights, [-1, 16, 16, 1]), 1)

writer = tf.summary.FileWriter("./logs")

learningRate = 0.9

momentumSpeed = tf.Variable(np.random.random_sample(weights.shape).astype("float32"), dtype="float32")
miniBatchSize = 100
startOfNextMiniBatch = 0



miniBatch = tf.placeholder(dtype="float32", shape = [inputs.shape[0], miniBatchSize], name="miniBatch")
gradient = cd1(miniBatch, weights)
# gradient = tf.placeholder(dtype="float32", shape = rbmWeights.shape, name="gradient")

momentumSpeedUpdate = tf.add(tf.multiply(tf.constant(0.9, dtype="float32"), momentumSpeed), gradient)
modelUpdate = tf.add(rbmWeights, tf.multiply(momentumSpeedUpdate, tf.constant(0.9, dtype="float32")))

with tf.Session() as sess:
    momentumSpeed.initializer.run()
    weights.initializer.run()

    for i in range(10000):
        miniBatchValues = inputs[:, startOfNextMiniBatch: (startOfNextMiniBatch + miniBatchSize)]

        startOfNextMiniBatch = np.mod(startOfNextMiniBatch + miniBatchSize, inputs.shape[1])

        # gradient = sess.run(gradientComputation, feed_dict={miniBatch: miniBatchValues})
        # sess.run([gradient, momentumSpeedUpdate], feed_dict={miniBatch: miniBatchValues})
        momentumSpeed, weights= sess.run([momentumSpeedUpdate, modelUpdate], feed_dict={miniBatch: miniBatchValues})

        print("weights sample: ", weights[10, 10])

        summary_str = sess.run(weightSummary)

        # momentumSpeed = sess.run(momentumSpeedUpdate)
        # weights = learnedWeights

        # _, _, summary_str = sess.run([momentumSpeedUpdate, modelUpdate, weightSummary], feed_dict={miniBatch: miniBatchValues})

        writer.add_summary(summary_str, i)

        # momentumSpeed = 0.9 * momentumSpeed + gradient
        # model = model + momentumSpeed * learningRate

        # print("i: ", i, "weights: ", weights)


