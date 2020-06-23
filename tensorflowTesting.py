# Import the required packages and libraries

import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np 
import cv2

model = tf.keras.models.load_model('mnist.h5')

def testing():
	img = cv2.imread('image.png', 0)
	img = cv2.bitwise_not(img)
	img = cv2.resize(img, (28, 28))
	img = img.reshape(1, 28, 28, 1)
	img = img.astype('float32')
	img = img/255

	pred = model.predict(img)
	return pred 