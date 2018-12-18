import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from scipy import ndimage
import sklearn
import matplotlib.pyplot as plt

samples = []
with open('/opt/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            # use center, left, right images with correction
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = '/opt/data/IMG/' + filename
                    image = cv2.imread(current_path)
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
            Y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, Y_train)

# compile and train model using generator function
train_generator= generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Convolution2D

# NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(filters = 24, kernel_size = 5, strides=2, activation="relu"))
model.add(Conv2D(filters = 36, kernel_size = 5, strides=2, activation="relu"))
model.add(Conv2D(filters = 48, kernel_size = 5, strides=2, activation="relu"))
model.add(Conv2D(filters = 64, kernel_size = 3, strides=1, activation="relu"))
model.add(Conv2D(filters = 64, kernel_size = 3, strides=1, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print(model.summary())

model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
history_object = model.fit_generator(
            train_generator, samples_per_epoch = len(train_samples),
            validation_data=validation_generator,  nb_val_samples=len(validation_samples),
            nb_epoch=5, verbose = 1)

model.save('model.h5')


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()