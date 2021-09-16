import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf

from model import *

from tensorflow.keras.optimizers import Adam
import plotly.offline as pyo
pyo.init_notebook_mode()
import os
import string
from sklearn.model_selection import train_test_split
from utils import *
import matplotlib.pyplot as plt

np.random.seed(1)
tf.random.set_seed(1) 


#Load samples
datapath= "samples"
symbols = string.ascii_lowercase + '0123456789'


model = myModel()
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])

def preprocessing(path):

	print("LOADING. . . .")
	n_samples= len(os.listdir(path))
	

	# variables for data and labels 
	X = np.zeros((n_samples , 50 , 200 ,1 ))  # (samples , height , width , channel)
	y = np.zeros((n_samples,5, 36 ))       #(samples , captcha characters , ascii char + numbers)

	for i , image in enumerate(os.listdir(path)):
		img = cv.imread(os.path.join(path, image) , cv.IMREAD_GRAYSCALE)

		targets = image.split('.')[0]

		if len(targets)<6:

			img = img/255.0
			img = np.reshape(img , (50,200,1))

			#find the char and one hot encode it to the target
			targ = np.zeros((5,36))

			for l , char in enumerate(targets):

				idx = symbols.find(char)
				targ[l , idx] = 1

			X[i] = img
			y[i,: ,:] = targ

	print("PROCESSING. . . .")

	return X,y

X, y = preprocessing(datapath)

trainX , testX , trainY , testY = train_test_split(X, y , test_size=0.2 , random_state=42)

labels = {'char_1': trainY[:,0,:], 
         'char_2': trainY[:,1,:],
         'char_3': trainY[:,2,:],
         'char_4': trainY[:,3,:],
         'char_5': trainY[:,4,:]}

test_labels = {'char_1': testY[:,0,:], 
         'char_2': testY[:,1,:],
         'char_3': testY[:,2,:],
         'char_4': testY[:,3,:],
         'char_5': testY[:,4,:]}

history = model.fit(trainX ,labels , epochs=32, batch_size=64 , validation_split=0.2)

#score = model.evaluate(testX , test_labels , batch_size=32)

def predictions(image):
    
    image = np.reshape(image , (50,200))
    result = model.predict(np.reshape(image , (1,50,200,1)))
    result = np.reshape(result ,(5,36))
    indexes =[]
    for i in result:
        indexes.append(np.argmax(i))
       
    label=''
    for i in indexes:
        label += symbols[i]
        
    plt.imshow(image)
    plt.title(label)
    plt.show()
    print("PREDICTION: ", label)

#img = cv.imread('captchatest1.png')[:,:,0]

#test out accuracy
predictions(testX[158])
predictions(testX[200])
predictions(testX[23])
predictions(testX[199])

model.save('captcha.model')