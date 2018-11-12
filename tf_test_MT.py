import pickle
import numpy as np
import PIL.Image
import itertools
from IPython.core.display import Image, display
import scipy.ndimage
import random
import numpy as np
import matplotlib.pylab as plt
import tqdm
import math
import tensorflow as tf
from tensorflow import keras
from sklearn.neural_network import MLPClassifier
#%matplotlib inline


with open('./hw_1_train.pickle', 'rb') as f:
    train = pickle.load(f)

with open('./hw_1_test_no_lables.pickle', 'rb') as f:
    test_no_lables = pickle.load(f)

model = keras.Sequential()

model.add(keras.layers.Dense(784, activation='linear',input_shape=(784,)))
model.add(keras.layers.Dense(60, activation='linear'))
model.add(keras.layers.Dense(60, activation='linear'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.sparse_categorical_crossentropy)#,
              #metrics=['accuracy'])

xtrain = np.array([np.array(i,dtype='float32') for i in train['data'][:5000]])
xtest = np.array([np.array(i,dtype='float32') for i in train['data'][5000:]])
ytrain = np.array([np.array(i,dtype='float32') for i in train['labels'][:5000]])
ytest = np.array([np.array(i,dtype='float32') for i in train['labels'][5000:]])

#print(ytest)

model.fit(xtrain,ytrain, epochs=1, steps_per_epoch=100,
          validation_data=(xtest,ytest),
          validation_steps=10)

xpred = np.array([np.array(i,dtype='float32') for i in test_no_lables['data']])
predict_results = model.predict(xpred, steps=30)
predictions = list(itertools.islice(predict_results,len(xpred)))

for k,val in enumerate(predictions):
  print(val)
