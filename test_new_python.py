import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cantera as ct
from scipy import optimize as opt

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

min = opt.minimize(obj_func(x,states),1000.,method='Nelder-Mead')
print(min)

states_new,gas2 = states_new_init(min.x[0])





C = plt.cm.winter(np.linspace(0,1,10))
'''
plt.plot(tt,TT, lw=2, color=C[0],
        label='K={0}, R={1}'.format(gas2.n_species, N))
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')
plt.legend(loc='upper left')
plt.title('Reduced mechanism ignition delay times\n'
          'K: number of species; R: number of reactions')
plt.xlim(0, 200)
plt.tight_layout()
plt.show()
'''

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
