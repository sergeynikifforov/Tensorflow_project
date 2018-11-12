import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
#building a model

def func(x):
    return (x**3)+math.exp(x)

model = keras.Sequential()

model.add(keras.layers.Dense(60, activation='elu',input_shape=(1,)))
#model.add(keras.layers.Dense(1, activation='elu'))
model.add(keras.layers.Dense(60, activation='softmax'))
#model.add(keras.layers.Dense(1, activation='elu'))
model.add(keras.layers.Dense(60, activation='elu'))
#model.add(keras.layers.Dense(64, activation='sigmoid'))
#model.add(keras.layers.Dense(1, activation='sigmoid'))
model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l1(0.01)))

#layers.Dense(64, activation='sigmoid')


model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='MSLE',
              metrics=['mae'])

#input dataset

xtrain = np.array([np.array([i/10.],dtype='float32') for i in range(1,100,2)])
#print(xtrain.shape)
#ytrain = np.array([np.array([math.exp(i)],dtype='float32') for i in xtrain])
ytrain = np.array([np.array([func(i[0])],dtype='float32') for i in xtrain])
#print([func(i[0]) for i in xtrain])
xtest = np.array([np.array([i/10.],dtype='float32') for i in range(3,100,4)])
#ytest = np.array([np.array([math.exp(i)],dtype='float32') for i in xtest])
ytest = np.array([np.array([func(i[0])],dtype='float32') for i in xtest])

dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
dataset = dataset.batch(64).repeat()
#print(dataset.shape)
val_dataset = tf.data.Dataset.from_tensor_slices((xtest, ytest))
val_dataset = val_dataset.batch(64).repeat()

model.fit(dataset, epochs=300, steps_per_epoch=100,
          validation_data=val_dataset,
          validation_steps=10)

#model.fit(xtrain,ytrain, epochs=10, steps_per_epoch=30,
#          validation_data=test_data
#)


xpred = np.array([np.array([i/10.],dtype = 'float32') for i in range(2,100,4)],dtype = 'float32')

model.evaluate(dataset, steps=300)

predict_results = model.predict(xpred, steps=30)

predictions = list(itertools.islice(predict_results,len(xpred)))

massive_x = []
massive_y = []
#massive_y_ist = [np.array([math.exp(i)],dtype='float32') for i in xpred]
massive_y_ist = [np.array([func(i[0])],dtype='float32') for i in xpred]
for k,val in enumerate(predictions):
  print(str(xpred[k]) + ' ' + str(val))
  massive_x.append(xpred[k])
  massive_y.append(val)


model.save_weights('my_model_weights_old.h5')

plt.plot(massive_x, massive_y,label = 'pred')
plt.plot(massive_x, massive_y_ist,label = 'ist')
plt.yscale('log')
plt.legend()
plt.show()
