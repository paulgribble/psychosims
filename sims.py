# ipython --pylab

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

S = arange(5, 21, 3)        # number of subjects
N = arange(5, 16, 2)        # trials per position
E = arange(0.1, 3.1, 0.1)   # threshold shift (mm)
X = 10000;                  # number of experiments per config

nsims = len(S)*len(N)*len(E)*X
i_count = 0

D05 = zeros((len(S),len(N),len(E)))
D01 = zeros((len(S),len(N),len(E)))

for i_s in range(len(S)):
    for i_n in range(len(N)):
        for i_e in range(len(E)):
            b1 = 0.4  # slope
            b0 = -b1 * E[i_e]
            sim1 = format("./sims %d %d %f %f > pre" %
                          (S[i_s]*X, N[i_n], 0.0, b1) )
            sim2 = format("./sims %d %d %f %f > post" %
                          (S[i_s]*X, N[i_n], b0, b1) )
            print sim1
            print sim2
            tmp = os.system(sim1)
            tmp = os.system(sim2)
            pre = genfromtxt('pre')
            post = genfromtxt('post')
            for i_x in range(X):
                i1, i2 = i_x*S[i_s], (i_x+1)*S[i_s]
                t,p = ttest_rel(post[i1:i2,2],pre[i1:i2,2])
                D05[i_s,i_n,i_e] += int((p<.05) and (t>0))
                D01[i_s,i_n,i_e] += int((p<.01) and (t>0))
                i_count += 1
            print("%6d/%6d : S=%2d N=%2d E=%4.2f D05=%3d D01=%3d" %
                  (i_count, nsims, S[i_s], N[i_n], E[i_e],
                   D05[i_s,i_n,i_e], D01[i_s,i_n,i_e]) )
            print " "
D05 = D05 / float(X)
D01 = D01 / float(X)
print "done: simulated %d subjects" % (nsims)

figure(figsize=(16,8))
for i in range(len(S)):
    subplot(2,3,i+1)
    for j in range(len(N)):
        plot(E,D05[i,j,:],'-')
    ylim(0,1)
    xlim(min(E),max(E))
    title("%d Subjects" % S[i])
    grid(1)
    if (i>2):
        xlabel("Threshold shift (mm)")
    if (i==0 or i==3):
        ylabel("Power at p<0.05")
    if (i==5):
        legend(N, loc=4, title="Trials")
savefig("psychosims.pdf")
