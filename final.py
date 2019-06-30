import numpy as np
import matplotlib.pyplot as plt
import os
import cv2



DATA_DIR = "/Users/krishrana/Python/Dataset" #path to the dataset your 

CATEGORIES = ["Car", "Auto", "Motorcycle"] #categories of vehicles

img_size=70

training_data = []
def create_training_data():
	#iterating through different categories
	for category in CATEGORIES:
		path = os.path.join(DATA_DIR,category) #creating the path to the images
		print(path)
		class_num = CATEGORIES.index(category) #creating classification label for the dataset (indexing the categories)
		print(class_num) 
		for img in os.listdir(path): #loading each image in the directory
			try:
				img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array,(img_size,img_size)) #resizeing the all the images to same size(70,70)
				training_data.append([new_array,class_num])
			except Exception as e:
				pass


			

create_training_data()
print(len(training_data))

import random
random.shuffle(training_data) #randomizing the data

X=[]
y=[]

for features,label in training_data: #new_array is features and class_num is label
	X.append(features)
	y.append(label)


X=np.array(X).reshape(-1,img_size,img_size,1) #transforming X into numpy array
print(X.shape)
print(y)


# training the data set using tensorflow.keras library
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

#tf.keras is tensorflow's implementation of keras
X=X/255.0
model=Sequential()
model.add(Conv2D(256,(3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())  #makes it 1d array
model.add(Dense(64))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
y_final = to_categorical(y, num_classes=3) #changing the classification indices to binary for loss=categorical
print(y_final.shape)
print(y_final)


model.fit(X, y_final, batch_size=32, epochs=3, validation_split=0.3) #30% of the dataset is used for testing
model.save('257x32.CNN.model') #saving the model for future use

#prediction of new image is done in testfinal.py
