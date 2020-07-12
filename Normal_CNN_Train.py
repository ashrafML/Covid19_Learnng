# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:58:10 2020

@author: a.ragab
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
import os
import cv2
from imutils import paths
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
#loadind data


print("[INFO] loading images...")
imagePaths = list(paths.list_images("dataset"))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# Deffining the Training and Testing Datasets

data = np.array(data) / 255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
#inail data augmentaion
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# Initialising the CNN
CovdModel = Sequential()

# Step1 - Convolution
# Input Layer/dimensions
# Step-1 Convolution
# 64 is number of output filters in the convolution
# 3,3 is filter matrix that will multiply to input_shape=(64,64,3)
# 64,64 is image size we provide
# 3 is rgb

CovdModel.add(Convolution2D(64,3,3, input_shape=(224,224,3), activation='relu'))

# Step2 - Pooling
#Processing
# Hidden Layer 1
# 2,2 matrix rotates, tilts, etc to all the images
CovdModel.add(MaxPooling2D(pool_size=(4,4)))

# Adding a second convolution layer
# Hidden Layer 2
# relu turns negative images to 0
CovdModel.add(Convolution2D(64,3,3, activation='relu'))
CovdModel.add(MaxPooling2D(pool_size=(4,4)))

# step3 - Flattening
# converts the matrix in a singe array
CovdModel.add(Flatten())

# Step4 - Full COnnection
# 128 is the final layer of outputs & from that 1 will be considered ie dog or cat

CovdModel.add(Dense(output_dim=128, activation='relu'))
CovdModel.add(Dropout(0.5))
CovdModel.add(Dense(output_dim=2, activation='softmax'))

# Compiling the CNN
INIT_LR = 1e-3
EPOCHS = 8
BS = 15
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
CovdModel.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])


# nb_epochs how much times you want to back propogate
# steps_per_epoch it will transfer that many images at 1 time
# & epochs means 'steps_per_epoch' will repeat that many times
CovdModel.fit_generator(
     train_datagen.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,epochs=EPOCHS
    )
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = CovdModel.predict(testX)

predIdxsN = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxsN))
#conf matrix
cm = confusion_matrix(testY.argmax(axis=1), predIdxsN)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# make predsiction

hist=CovdModel.history
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.savefig("plot1 for cnn")
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("plot2 for cnn")
#
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")
from PIL import Image
imagePathsPred = list(paths.list_images("check"))
newdata=[]
for imagePath in imagePathsPred:


	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update the data and labels lists, respectively
	newdata.append(image)
newdata = np.array(newdata) / 255.0   
 
result = CovdModel.predict(newdata)
