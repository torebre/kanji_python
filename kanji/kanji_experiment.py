import numpy as np
import tensorflow as tf
from rbm.showRbm import showWeights
from rbm.cd1 import cd1

import kanji.load_data as ld




learningRate = 0.9
momentumSpeedConstant = 0.9
miniBatchSize = 100
numberOfHiddenUnits = 50
numberOfIterations = 100000
numberType = "float32"




images = ld.loadFilesInDirectory("/home/student/workspace/testEncodings/fragments")

# ld.plotImages(images)

loadedImages = []
imageShape = None

for image in images:
    loadedImages.append(np.reshape(image.astype("float32"), (image.shape[0] * image.shape[1])))

    if imageShape is None:
        imageShape = image.shape

inputs = np.asarray(loadedImages).transpose()
shape = inputs.shape

initialWeights = ((np.random.random_sample((numberOfHiddenUnits, inputs.shape[0])) * 2 - 1) * 0.).astype(numberType)
weights = tf.Variable(initial_value=initialWeights, name="weights")

writer = tf.summary.FileWriter("../model_output/kanji_logs")

momentumSpeed = tf.Variable(np.zeros(weights.shape).astype(numberType), dtype=numberType)

weightSummary = tf.summary.image('input', tf.reshape(weights[0, :], [-1, imageShape[0], imageShape[1], 1]), 1)

startOfNextMiniBatch = 0

miniBatch = tf.placeholder(dtype=numberType, shape=[shape[0], miniBatchSize], name="miniBatch")
gradient = cd1(miniBatch, weights)

momentumSpeedUpdated = tf.add(tf.multiply(tf.constant(momentumSpeedConstant, dtype=numberType), momentumSpeed), gradient)
weightsUpdated = tf.add(weights, tf.multiply(momentumSpeedUpdated, tf.constant(learningRate, dtype=numberType)))

updateMomentumSpeed = momentumSpeed.assign(momentumSpeedUpdated)
updateWeights = weights.assign(weightsUpdated)

for i in range(numberOfHiddenUnits):
    tf.summary.image('weights' + str(i), tf.reshape(updateWeights[i, :], [-1, 16, 16, 1]), 1)


tf.summary.histogram('momentumSpeedUpdated', momentumSpeedUpdated[0])
tf.summary.histogram('weightsUpdated',weightsUpdated[0])

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    momentumSpeed.initializer.run()
    weights.initializer.run()

    for i in range(numberOfIterations):
        miniBatchValues = inputs[:, startOfNextMiniBatch: (startOfNextMiniBatch + miniBatchSize)]
        startOfNextMiniBatch = np.mod(startOfNextMiniBatch + miniBatchSize, inputs.shape[1])

        _, _, summary = sess.run([updateWeights, updateMomentumSpeed, merged_summary_op], feed_dict={miniBatch: miniBatchValues})

        if i % 1000 == 0:
            print("Iteration: ", i)
            writer.add_summary(summary, i)

    finalWeights = sess.run(weights)
    showWeights(finalWeights, imageShape)
