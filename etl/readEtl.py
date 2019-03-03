import struct
import numpy as np
from matplotlib import pylab as plt


structFormat = 'hh4s504s64s'
structLength = struct.calcsize(structFormat)
unpackFunction = struct.Struct(structFormat).unpack_from

kanjiData = []

with open('/home/student/Downloads/ETL/ETL9B/ETL9B_1', mode = 'rb') as file:
    while True:
        record = file.read(structLength)
        if not record:
            break

        (serialSheetNumber, kanjiCode, typicalReading, imageData, uncertain) = unpackFunction(record)
        image = np.unpackbits(np.fromstring(imageData, dtype=np.uint8)).reshape(64, 63)

        kanjiData.append(tuple([serialSheetNumber, kanjiCode, typicalReading, image]))


print "Kanji data length: ", len(kanjiData)

