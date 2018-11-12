import sys
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import cantera as ct
from scipy import optimize as opt
import numpy as np
import itertools

### ~INFO
### ~CANTERA SCRIPT FOR WRITTING ONLY

def states_new_init(A):
    R_new = ct.Reaction.fromCti('''reaction('O2 + 2 H2 => 2 H2O',
            [%e, 0.0, 0.0])'''%(A))
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



gas = ct.Solution('gri30.xml')
initial_state = 1500, ct.one_atm, 'H2:2,O2:1'
gas.TPX = 1500.0, ct.one_atm, 'H2:2,O2:1'
r = ct.IdealGasConstPressureReactor(gas,energy = 'off')
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])
i = (j for j in range(0,10))
N = (j for j in range(1,11))
for n in range(100):
    time += 1.e-5
    sim.advance(time)
    states.append(r.thermo.state, t=time*1e3)


min = opt.minimize(lambda x: obj_func(x,states),1000.,method='Nelder-Mead')
states_new,gas2 = states_new_init(min.x[0])


arren_array = abs(np.random.normal(min.x[0],10e10,500))
states_new_arr = []
gas_new =  []

for i in range(len(arren_array)):
    value_1, value_2 = states_new_init(arren_array[i])
    states_new_arr.append(value_1)
    gas_new.append(value_2)

#train dataset
xtrain = [i for i in arren_array[:250]]
ytrain = [obj_func(i,states) for i in arren_array[:250]]
#test Dataset
xtest = [i for i in arren_array[250:]]
ytest = [obj_func(i,states) for i in arren_array[250:]]

#predict dataset

arren_array_pred = abs(np.random.normal(min.x[0],10e10,100))

xpred = [i for i in arren_array_pred]








file_1 = open('output_train_x.txt', 'w')
file_2 = open('output_train_y.txt', 'w')
file_3 = open('output_test_x.txt', 'w')
file_4 = open('output_test_y.txt', 'w')
file_5 = open('output_pred_x.txt', 'w')

file_1.writelines("%s\n" % i for i in xtrain)
file_2.writelines("%s\n" % i for i in ytrain)
file_3.writelines("%s\n" % i for i in xtest)
file_4.writelines("%s\n" % i for i in ytest)
file_5.writelines("%s\n" % i for i in xpred)

file_1.close()
file_2.close()
file_3.close()
file_4.close()
file_5.close()
