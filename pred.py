import cv2
import numpy as np
import keras
import os
import matplotlib.pyplot as plt

test_folder = './asl-alphabet/asl_alphabet_test/asl_alphabet_test/'

model = keras.models.load_model('model.h5') # edit this if your model is in a different directory
values={0:'A',1:'B',2:'C',3:'D',4:'delete',5:'E',6:'F',7:'G',8:'H',9:'I',10:'J',11:'K',12:'L',13:'M',14:'N',15:'nothing',16:'O',17:'P',18:'Q',19:'R',20:'S',21:'space',22:'T',23:'U',24:'V',25:'W',26:'X',27:'Y',28:'Z',29:'0',30:'1',31:'2',32:'3',33:'4',34:'5',35:'6',36:'7',37:'8',38:'9'}

for filename in os.listdir(test_folder):
    img = cv2.imread(test_folder+"/" + filename, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (70,70))
    img_data = resized_img.flatten() / 255
    img_data = img_data.reshape(1, 70, 70, 1)
    img_data = img_data.astype('float32')
    result=model.predict(img_data)
    result=values[np.argmax(result)]
    print("Should be "+filename)
    print("Actual "+result)
        
# Read in an image
img = cv2.imread('IMG_7050.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocessing
img_data = []
resized_img = cv2.resize(img, (70,70))
img_data.append(resized_img.flatten() / 255) # flatten to 784 and normalize values
img_data = np.asarray(img_data)
img_data = img_data.reshape(img_data.shape[0], 70, 70, 1)
img_data = img_data.astype('float32')

result = model.predict(img_data)
final=values[np.argmax(result)]
print(final)