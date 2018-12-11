import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from scipy import ndimage

lines = []
with open('/opt/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    # current_path = '../data/IMG/' + filename
    current_path = '/opt/data/IMG/' + filename
    # image = cv2.imread(current_path)
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda

model = Sequential()
# Basic architecture
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('model.h5')

# python clone.py
"""
# LeNet architecture
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160, 320, 3)))
"""