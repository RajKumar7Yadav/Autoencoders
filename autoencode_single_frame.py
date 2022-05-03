# -*- coding: utf-8 -*-
"""
Created on Tue May  3 20:20:42 2022

@author: raj.yadav
"""

from matplotlib.pyplot import imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.models import Sequential


SIZE=256
img_arr=[]
img=cv2.imread('477_1092778_image.jpg')
img=cv2.resize(img,(SIZE,SIZE))
img=img.reshape(1,SIZE,SIZE,3)
#img_arr=np.array(img,copy=False, dtype=np.float32)
#img_array = np.reshape(img, (len(img), SIZE, SIZE, 3))
img_array = img.astype('float32') / 255.

#Defining the model architecture, it is very necessary that the dimension of input and output should be same
# hence we MaxPool and Upsample accordingly
model=Sequential()
# 1st convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 

model.add(MaxPool2D((2, 2), padding='same'))
     
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

model.fit(img_array, img_array,epochs=50,shuffle=True)


print("Neural network output")
pred = model.predict(img_array)



imshow(pred[0].reshape(SIZE,SIZE,3), cmap="gray")
