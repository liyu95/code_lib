#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# plt.style.use('ggplot')
plt.style.use('seaborn-paper')

width = 0.25

plt.figure(dpi=350)



a = [0.936738672, 0.919129385, 0.789914356, 0.760632167, 0.769681696]
b = [0.883349865, 0.845950942, 0.792786653, 0.713368304, 0.73874345]
c = [0.543962651, 0.315914379, 0.437488156, 0.258426155, 0.282291543]
d = [0.621827254, 0.459619358, 0.505443876, 0.328574136, 0.360654284]
e = [0.855348991, 0.802314199, 0.834380686, 0.682032907, 0.735927322]
f = [0.897931314, 0.866961844, 0.66418035, 0.619124406, 0.631909399]

x = np.arange(len(a))*2.5

bar1 = plt.bar(x, a, width, label='DEEPre')
bar2 = plt.bar(x+width, b, width, label='EzyPred')
bar3 = plt.bar(x+2*width, c, width, label='SVMProt')
bar4 = plt.bar(x+3*width, d, width, label='SVM-Raw')
bar5 = plt.bar(x+4*width, e, width, label='SVM-PrePssm')
bar6 = plt.bar(x+5*width, f, width, label='NN-Raw')

plt.xticks(x+width*(0.5*5), ['Accuracy', 'Cohen\'s Kappa score', 
	'Macro-Precison', 'Macro-Recall', 'Macro-F1'], fontsize=7)
axes = plt.gca()
axes.set_ylim([0.3,1])

# plt.xlabel('Critiria')
# plt.ylabel('Score')
plt.title('NEW data level 2 performance comparison')
legend = plt.legend(#bbox_to_anchor=(0.9,0.9), 
	loc=0, shadow=True, fontsize=5,fancybox=True)
legend.get_frame().set_facecolor('#eeefff')
plt.grid(True, color='k', linestyle=':', linewidth=0.5, alpha=0.2)
# plt.savefig('test.svg')
plt.savefig('nd_level_2.jpg')
# plt.show()
