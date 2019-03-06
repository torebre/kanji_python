import struct
import numpy as np
from matplotlib import pylab as plt
from PIL import Image


structFormat = 'hh4s504s64s'
structLength = struct.calcsize(structFormat)
unpackFunction = struct.Struct(structFormat).unpack_from

kanjiData = []
images = []


with open('/home/student/Downloads/ETL/ETL9B/ETL9B_2', mode = 'rb') as file:
    while True:
        record = file.read(structLength)
        if not record:
            break

        (serialSheetNumber, kanjiCode, typicalReading, imageData, uncertain) = unpackFunction(record)
        image = np.unpackbits(np.fromstring(imageData, dtype=np.uint8)).reshape((63, 64))

        images.append(Image.frombuffer('1', (64, 63), imageData, 'raw'))

        kanjiData.append(tuple([serialSheetNumber, kanjiCode, typicalReading, image]))


print "Kanji data length: ", len(kanjiData)


# for x in range(10000, len(kanjiData), 1000):
for x in range(10000, 10050):
    # sm.imsave('kanji_' + str(x) + '.png', kanjiData[x][3])
    # sm.toimage(kanjiData[x][3], low=0, high=1).save('kanji_' + str(x) + '.png')
    plt.imsave('../etl_sample_pictures/kanji_' + str(x) + '.png', kanjiData[x][3])



for x in range(10000, 10050):
    images[x].save('../etl_sample_pictures/etl8b_v2_kanji_' + str(x) + '.png')
