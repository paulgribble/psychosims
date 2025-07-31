# python3 -m venv .venv
# source .venv/bin/activate
# pip install scipy tqdm matplotlib

# simulate a psychophysical experiment
# estimate psychophysical function pre and post learning
# 7 tested limb positions: -20.0, -10.0, -5.0, 0.0, 5.0,  10.0, 20.0
# threshold shift of E mm
# assume no change in slope
# N trials per position
# S subjects
# repeat experiment X times, do a t-test each time
# how many times out of X do we detect threshold change?

import os
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm import trange
from itertools import product

S = np.arange(5, 21, 3)        # number of subjects
N = np.arange(5, 16, 2)        # trials per position
E = np.arange(0.1, 3.1, 0.1)   # threshold shift (mm)
X = 10000;                     # number of experiments per config

nsims = len(S)*len(N)*len(E)*X
i_count = 0

D05 = np.zeros((len(S),len(N),len(E)))
D01 = np.zeros((len(S),len(N),len(E)))

# Define the grid of parameter combinations
param_grid = list(product(range(len(S)), range(len(N)), range(len(E))))

# Total number of innermost simulations
total_steps = len(param_grid) * X

with tqdm(total=total_steps, desc="Total Simulations") as pbar:
    for i_s, i_n, i_e in param_grid:
        b1 = 0.4
        b0 = -b1 * E[i_e]
        sim1 = format("./sims %d %d %f %f > pre" % (S[i_s]*X, N[i_n], 0.0, b1))
        sim2 = format("./sims %d %d %f %f > post" % (S[i_s]*X, N[i_n], b0, b1))
        os.system(sim1)
        os.system(sim2)
        pre = np.genfromtxt('pre')
        post = np.genfromtxt('post')

        for i_x in range(X):
            i1, i2 = i_x * S[i_s], (i_x + 1) * S[i_s]
            t, pval = ttest_rel(post[i1:i2, 2], pre[i1:i2, 2])
            D05[i_s, i_n, i_e] += int((pval < .05) and (t > 0))
            D01[i_s, i_n, i_e] += int((pval < .01) and (t > 0))
            pbar.update(1)
            #       (i_count, nsims, S[i_s], N[i_n], E[i_e],
            #        D05[i_s,i_n,i_e], D01[i_s,i_n,i_e]) ) )
            # print( " " )
D05 = D05 / float(X)
D01 = D01 / float(X)
print( "done: simulated %d subjects" % (nsims) )

plt.figure(figsize=(16,8))
for i in range(len(S)):
    plt.subplot(2,3,i+1)
    for j in range(len(N)):
        plt.plot(E,D05[i,j,:],'-')
    plt.ylim(0,1)
    plt.xlim(min(E),max(E))
    plt.title("%d Subjects" % S[i])
    plt.grid(1)
    if (i>2):
        plt.xlabel("Threshold shift (mm)")
    if (i==0 or i==3):
        plt.ylabel("Power at p<0.05")
    if (i==5):
        plt.legend(N, loc=4, title="Trials")
plt.savefig("psychosims.pdf")
