import numpy as np
import scipy.io as sio
import tensorflow as tf
from cd1 import cd1

dataSet = sio.loadmat("/home/student/Documents/Neural networks for Machine Learning/assignment4/data_set.mat")


inputs = dataSet["data"]["training"][0, 0]["inputs"][0, 0].astype("float32")
targets = dataSet["data"]["training"][0, 0]["targets"][0, 0].astype("float32")

numberOfHiddenUnits = 50
rbmWeights = np.random.random_sample((numberOfHiddenUnits, inputs.shape[0])).astype("float32")


inputsConstant = tf.placeholder(dtype="float32", shape = inputs.shape)
weights = tf.Variable(initial_value=rbmWeights, name="weights")

outputTensor = cd1(inputs, rbmWeights)


# weightSummary = tf.summary.image("weights", weights)
weightSummary = tf.summary.image('input', tf.reshape(rbmWeights, [-1, 16, 16, 1]), 1)

writer = tf.summary.FileWriter("./logs")

learningRate = 0.9

model = np.random.random_sample(modelShape)
momentumSpeed = np.zeros(modelShape)
miniBatchSize = 100
startOfNextMiniBatch = 1

with tf.Session() as sess:
    for i in range(1000):
        miniBatch = inputs[:, startOfNextMiniBatch: (startOfNextMiniBatch + miniBatchSize - 1)]
        startOfNextMiniBatch = np.mod(startOfNextMiniBatch + miniBatchSize, inputs.shape[1])
        gradient = cd1(model, miniBatch)
        momentumSpeed = 0.9 * momentumSpeed + gradient
        model = model + momentumSpeed * learningRate



        result, summary_str= sess.run([outputTensor, weightSummary], feed_dict={weights: rbmWeights})
        writer.add_summary(summary_str, i)

        print("i: ", i, ". Result: ", result)


