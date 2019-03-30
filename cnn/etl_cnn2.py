import random

import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from skimage.transform import resize
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K

from tensorflow.python.framework import graph_util
import tensorflow as tf

from etl.readEtl_9 import readall

random.seed(123)
np.random.seed(123)


allKanji = readall()

numberOfKanji = len(allKanji)
permutation = np.random.permutation(numberOfKanji)

kanjiImages = np.empty((numberOfKanji, 32, 32, 1), dtype=int)
kanjiClasses = np.empty(numberOfKanji, dtype=int)

counter = 0
for kanjiData in allKanji:
    # skeletonizedImage = skeletonize(image)
    kanjiImages[counter] = np.asarray(resize(kanjiData['image'], (32, 32)) > 0, dtype=int).reshape(1, 32, 32, 1)
    kanjiClasses[counter] = kanjiData['kanjiCode']
    counter = counter + 1

trainingSetStart = numberOfKanji - (numberOfKanji / 5)


xTrain = kanjiImages[permutation[0:trainingSetStart]]
xTest = kanjiImages[permutation[trainingSetStart:numberOfKanji]]


classesMap = {}
kanjiClassesInList = []
kanjiTargets = []
counter = 0
for kanjiCode in kanjiClasses:
    if kanjiCode not in classesMap:
        classesMap[kanjiCode] = counter
        kanjiClassesInList.append(kanjiCode)
        kanjiTargets.append(counter)
        counter = counter + 1
    else:
        kanjiTargets.append(classesMap[kanjiCode])



allClasses = to_categorical(kanjiTargets)
yTrain = allClasses[permutation[0:trainingSetStart]]
yTest = allClasses[permutation[trainingSetStart:numberOfKanji]]

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32, 32, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(counter, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])



model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10)

# predictFirst = model.predict(xTest[:1])
# np.argmax(predictFirst)



output_node_names = 'dense_1/Softmax'
output_graph_def = graph_util.convert_variables_to_constants(
    K.get_session(),
    tf.get_default_graph().as_graph_def(),
    output_node_names.split(",")
)
model_file = "./saved_model_2.pb"
with tf.gfile.GFile(model_file, "wb") as f:
    f.write(output_graph_def.SerializeToString())
