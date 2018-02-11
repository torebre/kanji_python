import matplotlib.pyplot as plt
import numpy as np



def showWeights(weights):
    numberOfRows = weights.shape[0] // 4 + 1
    plt.gray()

    print("number of rows:", numberOfRows)

    for i in range(0, weights.shape[0]):
        plt.subplot(numberOfRows, 4, i + 1)
        plt.axis('off')
        plt.imshow(np.reshape(weights[i, :], (16, 16)))

    plt.show()
