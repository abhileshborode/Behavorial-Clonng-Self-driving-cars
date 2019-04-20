import csv
import cv2
import numpy as np

from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


lines=[]

firstline = True
car_images=[]
steering_angles=[]
with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if firstline:    #skip first line
            firstline = False
            continue
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        source_path1 = row[0]
        source_path2 = row[1]
        source_path3 = row[2]
        filename1 = source_path1.split('/')[-1]
        filename2 = source_path2.split('/')[-1]
        filename3 = source_path3.split('/')[-1]
        path1 = 'data/IMG/' + filename1 
        path2 = 'data/IMG/' + filename2 
        path3 = 'data/IMG/' + filename3 

        img_center = np.asarray(Image.open(path1))
        img_left = np.asarray(Image.open(path2))
        img_right = np.asarray(Image.open(path3))



        car_images.extend([img_center, img_left, img_right])

        steering_angles.extend([steering_center, steering_left, steering_right])

#####----Data Augmentation-----######

'''
The foloowing code augments data by flipping every frame in the video by 180 degrees which 
is also equivalent to driving the same track in reverse direction.
'''
'''

aug_images,aug_mens= [],[]

for aug_image,aug_men in zip(car_images,steering_angles):
    aug_images.append(aug_image)
    aug_mens.append(aug_men)
    aug_images.append(cv2.flip(aug_image,1))
    aug_mens.append(aug_men*-1.0)
    
    
y_train = np.array(aug_mens)
X_train=np.array(aug_images)    
'''    
#####----Data Augmentation-----######


y_train = np.array(steering_angles) # training labels
X_train=np.array(car_images)   # training image pixels


def preprocess(image):  # preprocess image
    import tensorflow as tf
    return tf.image.resize_images(image, (200, 66))

# NVIDIA's End to end deep learning network architecture

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(preprocess))
model.add(Lambda(lambda x: (x/ 127.0 - 1.0)))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
#model.add(Dense(units=1164, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))
print(model.summary())


model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=6)
model.save('model.h5')


    