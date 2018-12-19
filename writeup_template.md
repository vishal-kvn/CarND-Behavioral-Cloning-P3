# Behavioral Cloning - Writeup

## Goals

The goals/steps of this project are the following:

- Use the simulator to collect data of good driving behavior.
- Build, a convolution neural network in [Keras](https://keras.io/) that predicts steering angles from images.
- Train and validate the model with a training and validation set.
- Test that the model successfully drives around track one without leaving the road.
- Summarize the results with a written report.

## Rubric points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:

- **model.py** : Containing the script to create and train the model
- **drive.py** : For driving the car in autonomous mode in the simulator (This is provided [Udacity](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/drive.py), my only modification was to increase the car speed on line 47 from 9 to 15)
- **model.h5** : Containing a trained convolution neural network.
- **writeup_report.md** : Summarizing the results

Node:

On my first iteration, I tried [LeNet](http://yann.lecun.com/exdb/lenet/) model and [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model.

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file; the car can be driven autonomously around the track by executing

```
Python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My initial approach was to use [LeNet](http://yann.lecun.com/exdb/lenet/), but it was hard to have the car inside the street with three epochs. After this, I decided to try the [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model, and the car drove the complete first track after just three training epochs (this model could be found [here]()).

A model summary from keras is as follows:

```
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
```

(More details about this bellow.)

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting, I trained the model for a low number of epochs. Since the number of epochs is low, I did not use regularization techniques like Dropout and Maxpooling.

In addition to that, I split my sample data into training and validation data. Using 80% as training and 20% as validation. This can be seen at [this part of the code](model.py#L17).

#### 3. Model parameter tuning

I did not modify the nVidea architecture. The only difference is the size of the input. It is (65, 320, 3) after cropping. I initially trained the model for 2 epochs and saw that the car was having touble on the bridge. After increasing the epochs to 3 the car was able to drive much better on the bridge. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road which I downloaded from [https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip](wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). In addition to the center, left and right images from the simulator I used the flipped version of these images to increase the training data. A correction factor of +0.2 was applied for the left steering measurement and a correction factor of -0.2 was applied for the right steering measurement.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As mentioned in the lesson, my first step was to try the LeNet](http://yann.lecun.com/exdb/lenet/) model with three epochs and the training data provided by Udacity. On the first track, the car went straight to the lake. After pre-processing the images by normalizing the inputs to zero mean and cropping the top 70 pixels and bottom 25 pixels the car made it further along but went off the track.

The second step was to use a more powerfull model: [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) The only modification was to add a new layer at the end to have a single output as it was required. This time the car did its first complete track, but it was having trouble near the bridge. 
To provide the model with more data, I augmented the images by flipping them and multiplying their corresponding measurements with -1. This improved the performance of the car.

#### 2. Final Model Architecture

The final model architecture is shown in the following image:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Fully connected: neurons: 100, activation: RELU
- Fully connected: neurons:  50, activation: RELU
- Fully connected: neurons:  10, activation: RELU
- Fully connected: neurons:   1 (output)

#### 3. Creation of the Training Set & Training Process

The training data was obtained from [https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). This data was shuffled and used train the model. As the size of the data increased due to data augmentation I switched to using generators.

After this training, the car was driving down the road all the time on the [first](video.mp4).

## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim