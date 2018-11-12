import sys
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Activation
import cantera as ct
from scipy import optimize as opt
import numpy as np
import itertools
import math
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K





class custom_activation_class_r6(Activation):

    def __init__(self, activation, **kwargs):
        super(custom_activation_class_r6, self).__init__(activation, **kwargs)
        self.__name__ = 'custom_activation_r6'


class custom_activation_class_r12(Activation):

    def __init__(self, activation, **kwargs):
        super(custom_activation_class_r12, self).__init__(activation, **kwargs)
        self.__name__ = 'custom_activation_r12'

class custom_activation_class_ln(Activation):

    def __init__(self, activation, **kwargs):
        super(custom_activation_class_ln, self).__init__(activation, **kwargs)
        self.__name__ = 'custom_activation_ln'

class custom_activation_class_exp(Activation):

    def __init__(self, activation, **kwargs):
        super(custom_activation_class_exp, self).__init__(activation, **kwargs)
        self.__name__ = 'custom_activation_exp'

class custom_activation_class_sigm_der(Activation):

    def __init__(self, activation, **kwargs):
        super(custom_activation_class_sigm_der, self).__init__(activation, **kwargs)
        self.__name__ = 'custom_activation_sigm_der'



def custom_activation_r6(x):
    return  x**(-6)

def custom_activation_ln(x):
    return  tf.math.log(tf.abs(x)+0.0001)

def custom_activation_exp(x):
    return  tf.math.exp(x)

def custom_activation_r12(x):
    return  x**(-12)
def custom_activation_sigm_der(x):
    return tf.math.exp(-x)/(1+tf.math.exp(-x))**2

def states_new_init(A):
    R_new = ct.Reaction.fromCti('''reaction('O2 + 2 H2 => 2 H2O',
            [%e, 0.0, 0.0])'''%(A))
    #print(type(R_new))
    #print(type(gas.reactions()))
    gas2 = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
    species=gas.species(), reactions=[R_new])
    gas2.TPX = initial_state
    r_new = ct.IdealGasConstPressureReactor(gas2, energy = 'off')
    t_new = 0.0
    states_new = ct.SolutionArray(gas2, extra=['t'])
    sim_new = ct.ReactorNet([r_new])
    tt = []
    TT = []
    for n in range(100):
        '''t_new += 1.e-5
        sim_new.advance(t_new)
        #print(t_new)
        tt.append(1000 * t_new*1000)
        TT.append(r_new.T)'''
        t_new += 1.e-5
        sim_new.advance(t_new)
        states_new.append(r_new.thermo.state, t=t_new*1e3)
    return states_new, gas2



def obj_func(A,states_ref):
    ret = 0.
    states_new,gas2 = states_new_init(A)
    for n in range(100):
        ret += (states_new.X[n,gas2.species_index('H2')] - states_ref.X[n,gas2.species_index('H2')])**2/100
    return ret
#return abs(a-b)


gas = ct.Solution('gri30.xml')
initial_state = 1500, ct.one_atm, 'H2:2,O2:1'
gas.TPX = 1500.0, ct.one_atm, 'H2:2,O2:1'
r = ct.IdealGasConstPressureReactor(gas,energy = 'off')
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])

#
i = (j for j in range(0,10))
N = (j for j in range(1,11))
#
for n in range(100):
    time += 1.e-5
    sim.advance(time)
    states.append(r.thermo.state, t=time*1e3)
    #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,
    #                                       r.thermo.P, r.thermo.u))

#min = opt.minimize(lambda x: obj_func(x,states),1000.,method='Nelder-Mead')
#print(min)

min = opt.minimize(lambda x: obj_func(x,states),1000.,method='Nelder-Mead')

states_new,gas2 = states_new_init(min.x[0])




#model init
model = keras.Sequential()

get_custom_objects().update({'custom_activation_r6': custom_activation_class_r6(custom_activation_r6)})
get_custom_objects().update({'custom_activation_r12': custom_activation_class_r12(custom_activation_r12)})
get_custom_objects().update({'custom_activation_ln': custom_activation_class_ln(custom_activation_ln)})
get_custom_objects().update({'custom_activation_exp': custom_activation_class_exp(custom_activation_exp)})
get_custom_objects().update({'custom_activation_sigm_der': custom_activation_class_sigm_der(custom_activation_sigm_der)})

model.add(keras.layers.Dense(1,custom_activation_class_sigm_der(custom_activation_sigm_der),input_shape=(1,)))
model.add(keras.layers.Dense(1,custom_activation_class_ln(custom_activation_ln)))
#model.add(keras.layers.Dense(1,custom_activation_class_ln(custom_activation_ln),input_shape=(1,)))
model.add(keras.layers.Dense(2,custom_activation_class_exp(custom_activation_exp)))
model.add(keras.layers.Dense(2,custom_activation_class_sigm_der(custom_activation_sigm_der)))
model.add(keras.layers.Dense(1,custom_activation_class_ln(custom_activation_ln)))
model.add(keras.layers.Dense(1, activation='linear'))
#model.add(keras.layers.Dense(1, activation='sigmoid',input_shape=(1,)))
#model.add(keras.layers.Dense(64, activation='elu'))
#model.add(keras.layers.Dense(64, activation='softmax'))
#model.add(keras.layers.Dense(1, activation='elu'))
#model.add(keras.layers.Dense(1,custom_activation_class_r12(custom_activation_r12)))
#model.add(keras.layers.Dense(20,activation='relu'))
#model.add(keras.layers.Dense(1,Activation(custom_activation)))
#model.add(keras.layers.Dense(1, activation='linear'))
#model.add(keras.layers.Dense(20, activation='softmax'))
#model.add(keras.layers.Dense(1, activation='elu'))
#model.add(keras.layers.Dense(1, activation='selu'))
#model.add(keras.layers.Dense(64, activation='sigmoid'))
#model.add(keras.layers.Dense(1, activation='sigmoid'))
#model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l1(0.01)))



#add parameters of model

model.compile(optimizer=tf.train.MomentumOptimizer(0.005,0.8,use_locking=True),
              loss='MSE',
              metrics=['mae'])


#obtain_train_data
file_train_x = open('output_train_x.txt')
file_train_y = open('output_train_y.txt')
train_x_arr = [line.strip() for line in file_train_x]
train_y_arr = [line.strip() for line in file_train_y]
file_train_x.close()
file_train_y.close()

#set_train_data
xtrain = np.array([np.array([math.log(float(i))],dtype='float32') for i in train_x_arr])
xtrain_norm_mean = np.mean(xtrain)
xtrain_norm_max = np.max(xtrain)
xtrain_norm_min = np.min(xtrain)
xtrain_norm = (xtrain - xtrain_norm_mean)/(xtrain_norm_max-xtrain_norm_min)+2
ytrain = np.array([np.array([math.log(float(i))],dtype='float32') for i in train_y_arr])

#obtain_test_data
file_test_x = open('output_test_x.txt')
file_test_y = open('output_test_y.txt')
test_x_arr = [line.strip() for line in file_test_x]
test_y_arr = [line.strip() for line in file_test_y]
file_test_x.close()
file_test_y.close()

#set_test_data
xtest = np.array([np.array([math.log(float(i))],dtype='float32') for i in test_x_arr])
xtest_norm_mean = np.mean(xtest)
xtest_norm_max = np.max(xtest)
xtest_norm_min = np.min(xtest)
xtest_norm = (xtest - xtest_norm_mean)/(xtest_norm_max-xtest_norm_min)+2
ytest = np.array([np.array([math.log(float(i))],dtype='float32') for i in test_y_arr])

#create_train_dataset
dataset = tf.data.Dataset.from_tensor_slices((xtrain_norm, ytrain))
dataset = dataset.batch(64).repeat()

#create_test_dataset
val_dataset = tf.data.Dataset.from_tensor_slices((xtest_norm, ytest))
val_dataset = val_dataset.batch(64).repeat()

#train_the_model
model.fit(dataset, epochs=300, callbacks = [tf.keras.callbacks.TerminateOnNaN()], steps_per_epoch=100,
          validation_data=val_dataset,
          validation_steps=15)

#obtain_prediction_data
file_pred_x = open('output_pred_x.txt')
pred_x_arr = [line.strip() for line in file_pred_x]
file_pred_x.close()

xpred = np.array([np.array([math.log(float(i))],dtype='float32') for i in pred_x_arr])
xpred_norm_mean = np.mean(xpred)
xpred_norm_max = np.max(xpred)
xpred_norm_min = np.min(xpred)
xpred_norm = (xpred - xpred_norm_mean)/(xpred_norm_max-xpred_norm_min)+2
xpred_test = np.array([np.array([math.log(float(i))],dtype='float32') for i in test_x_arr])
xpred_test_norm_mean = np.mean(xpred_test)
xpred_test_norm_max = np.max(xpred_test)
xpred_test_norm_min = np.min(xpred_test)
xpred_test_norm = (xpred_test - xpred_test_norm_mean)/(xpred_test_norm_max-xpred_test_norm_min)+2

predict_results = model.predict(xpred_norm, steps=30)
predict_results_train = model.predict(xpred_test_norm, steps=30)

predictions = list(itertools.islice(predict_results,len(xpred_norm)))
predictions_train = list(itertools.islice(predict_results_train,len(xpred_test_norm)))
res = []
res_test = []
#output predictions
for k,val in enumerate(predictions):
  print(str(xpred_norm[k]) + ' ' + str(val))
  res.append(val)

print('#################################################################################')

for k,val in enumerate(predictions_train):
  print(str(xpred_test_norm[k]) + ' ' + str(val))
  res_test.append(val)


model.save_weights('my_model_weights.h5')
json_string = model.to_json()
file_1 = open('json_model_output.txt', 'w')
file_1.write(json_string)
file_1.close()


plt.clf()
plt.subplot(3, 1, 1)
plt.scatter(xpred_norm, res ,color='red')
plt.xlim((1.5,2.2))
#plt.ylim(
plt.subplot(3, 1, 2)
plt.scatter(xtest_norm,ytest,color='red')
plt.xlim((1.5,2.2))
plt.subplot(3, 1, 3)
plt.scatter(xtrain_norm,ytrain,color='red')
plt.xlim((1.5,2.2))
#plt.tight_layout()
plt.show()

'''
C = plt.cm.winter(np.linspace(0,1,10))

















plt.clf()
plt.subplot(2, 2, 1)
plt.plot(states.t, states.T,color='red')
plt.plot(states_new.t, states_new.T,color='green')
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')
plt.subplot(2, 2, 2)
plt.plot(states.t, states.X[:,gas.species_index('O2')],color='red')
plt.plot(states_new.t, states_new.X[:,gas2.species_index('O2')],color='green')
plt.xlabel('Time (ms)')
plt.ylabel('O2 Mole Fraction')
plt.subplot(2, 2, 3)
plt.plot(states.t, states.X[:,gas.species_index('H2O')],color='red')
plt.plot(states_new.t, states_new.X[:,gas2.species_index('H2O')],color='green')
plt.xlabel('Time (ms)')
plt.ylabel('H2O Mole Fraction')
plt.subplot(2, 2, 4)
plt.plot(states.t, states.X[:,gas.species_index('H2')],color='red')
plt.plot(states_new.t, states_new.X[:,gas2.species_index('H2')],color='green')
plt.xlabel('Time (ms)')
plt.ylabel('H2 Mole Fraction')
plt.tight_layout()
plt.show()
'''
