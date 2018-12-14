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
    steering_center = float(line[3])
    correction = 0.2
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        # current_path = '../data/IMG/' + filename
        current_path = '/opt/data/IMG/' + filename
        image = cv2.imread(current_path)
        #image = ndimage.imread(current_path)
        images.append(image)
     
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

"""
model = Sequential()
# Basic architecture
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('model.h5')
"""

# LeNet architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(3,160,320)))

#Layer 1
#Conv Layer 1
model.add(Conv2D(filters = 6, 
                 kernel_size = 5, 
                 strides = 1, 
                 activation = 'relu', 
                 input_shape = (70,320,3)))
#Pooling layer 1
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters = 16, 
                 kernel_size = 5,
                 strides = 1,
                 activation = 'relu',
                 input_shape = (14,14,6)))
#Pooling Layer 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Flatten
model.add(Flatten())
#Layer 3
#Fully connected layer 1
model.add(Dense(units = 120))
#Layer 4
#Fully connected layer 2
model.add(Dense(units = 84))
#Layer 5
#Output Layer
model.add(Dense(units = 1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')