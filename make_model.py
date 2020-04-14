import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

train_folder = './asl_alphabet_train/asl_alphabet_train/'
test_folder = './asl_alphabet_test/asl_alphabet_test/'

values={0:'A',1:'B',2:'C',3:'D',4:'delete',5:'E',6:'F',7:'G',8:'H',9:'I',10:'J',11:'K',12:'L',13:'M',14:'N',15:'nothing',16:'O',17:'P',18:'Q',19:'R',20:'S',21:'space',22:'T',23:'U',24:'V',25:'W',26:'X',27:'Y',28:'Z',29:'0',30:'1',31:'2',32:'3',33:'4',34:'5',35:'6',36:'7',37:'8',38:'9'}
valList=list(values.values())

batch_size = 128
epochs = 15

train_data = []
labels = []

print("Creating input data...")
for foldername in os.listdir(train_folder):
	for filename in os.listdir(train_folder + foldername):
		img = cv2.imread(train_folder + foldername + "/" + filename, cv2.IMREAD_GRAYSCALE)
		currLabel=valList.index(foldername)
		resized_img = cv2.resize(img, (70,70))
		img_data = resized_img.flatten() / 255 # flatten to 784 and normalize values
		train_data.append(img_data)
		labels.append(currLabel)

train_data = np.asarray(train_data)
labels = np.asarray(labels)

print("Created input data with shape: %s" % (train_data.shape,))
print("Created label data with shape: %s" % (labels.shape,))

x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size = 0.2, random_state = 101)
x_train = x_train.reshape(x_train.shape[0], 70, 70,1)
x_test = x_test.reshape(x_test.shape[0], 70, 70,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = tf.keras.utils.to_categorical(y_train, 39)
y_test = tf.keras.utils.to_categorical(y_test, 39)

print("Creating the model...")

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(70, 70 , 1) ))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dropout(0.20))

model.add(keras.layers.Dense(39, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


# Train model
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
model.save('model.h5')
print("Model saved as model.h5")