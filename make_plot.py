import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
L = []
with open('./results/CNP_MLP_Mean_30_uniform_5.txt') as f:
    x = []
    y = []
    lines = f.readlines()
    for line in lines:
        x.append(float(line.strip().split(';')[0]))
        y.append(float(line.strip().split(';')[1]))
    X.append(x)
    Y.append(y)
    L.append('CNP')
with open('./results/BiLSTM_30_uniform_5.txt') as f:
    x = []
    y = []
    lines = f.readlines()
    for line in lines:
        x.append(float(line.strip().split(';')[0]))
        y.append(float(line.strip().split(';')[1]))
    X.append(x)
    Y.append(y)
    L.append('BiLSTM')

plt.figure()
for x, y, l in zip(X,Y,L):
    plt.plot(x, y, '-', linewidth=2, label=l)
plt.plot([30,30], [0,np.max(Y)*1.02], '--', linewidth=2, color='grey')
plt.xlabel('Resolution (2K+1)', fontsize=16)
plt.xticks(fontsize=12)
plt.ylabel('Representation Equivalence Error (mean(||uTf-TTuTf||))', fontsize=16)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.savefig('results.svg', bbox_inches='tight')
plt.close()
     