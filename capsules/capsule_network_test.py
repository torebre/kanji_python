import numpy as np
import tensorflow as tf

import kanji.load_data as ld



images = ld.loadFilesInDirectory("/home/student/workspace/testEncodings/fragments2")

# ld.plotImages(images)

loadedImages = []
imageShape = None

for image in images:
    loadedImages.append(np.reshape(image.astype("float32"), (image.shape[0] * image.shape[1])))

    if imageShape is None:
        imageShape = image.shape

inputs = np.asarray(loadedImages).transpose()
