from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.sparse.linalg import spsolve
import pandas as pd
from matplotlib import *

from inputABF import *
from P05fit import *


current, length, voltage, gain, conc_KCL= impt()


'''
### 02 -- S ###
# Smoothing data #
'''


lengthSmoothed = length
for i in range(0, 2):
    lengthSmoothed = signal.savgol_filter(x=lengthSmoothed, window_length=115, polyorder=2, deriv=0, mode='nearest')

plt.plot(length)
plt.plot(lengthSmoothed, color='r')
plt.show()

b = signal.get_window('triang', 1000, False)
print len(b)

lengthDeriv = signal.savgol_filter(length, 285, 4, 1)
lengthDeriv = signal.savgol_filter(lengthDeriv, 255, 1, mode='nearest')

flagus = 0
pnts = []
for i in range(len(lengthDeriv)):
    if lengthDeriv[i] < 0 and flagus == 0:
        flagus = 1
        tmpList = []
        tmpList.append(i)
    if lengthDeriv[i] > 0 and flagus == 1:
        flagus = 0
        if lengthSmoothed[tmpList[0]] - lengthSmoothed[i] >= 0.02:
            tmpList.append(i)
            pnts.append(tmpList)

for pnt in pnts:
    plt.plot(pnt[0], lengthDeriv[pnt[0]], 'ro')
    plt.plot(pnt[1], lengthDeriv[pnt[1]], 'ro', color = 'g')
plt.plot(lengthDeriv)
plt.show()

for pnt in pnts:
    plt.plot(pnt[0], length[pnt[0]], 'ro')
    plt.plot(pnt[1], length[pnt[1]], 'ro', color = 'g')
plt.plot(length)
plt.show()


'''
### 03 -- B ###
# Baseline correction #
'''

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in xrange(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

lengthDecimated = lengthSmoothed

for i in range(0, 11):
    lengthDecimated = signal.decimate(lengthDecimated, 2, zero_phase=True)

baseline4correction = baseline_als(lengthDecimated, 10000, 0.01)
print 'base'
#baseline4correction = signal.resample(baseline4correction, len(length))

#baseline4correction = signal.resample_poly(baseline4correction, len(lengthSmoothed), len(baseline4correction))
#baseline4correction = signal.savgol_filter(length, 121, 1)
print'resamp'
#length4peaks = lengthSmoothed - baseline4correction
length4peaks = lengthDecimated - baseline4correction


print np.mean(length4peaks)
print np.median(length4peaks)
print np.max(length4peaks)
print np.min(length4peaks)

plt.plot(lengthDecimated)
plt.plot(baseline4correction, color='r')
plt.plot(length4peaks, color='g')
plt.show()

minimal_length = np.mean(length4peaks)-np.median(length4peaks)
minimal_length2 = (np.max(length4peaks)-np.min(length4peaks))/2
print minimal_length, minimal_length2

points = pnts

for i in range(len(points)):
    points[i][0] = points[i][0] + int((points[i][1] - points[i][0]) * 0.05)
    #points[i][1] = points[i][1] + int((points[i][1] - points[i][0]) * 0.01)

for point in points:
    plt.plot(point[0], length[point[0]], 'ro')
    plt.plot(point[1], length[point[1]], 'ro', color = 'g')
plt.plot(length)
plt.show()

finalTable = fit(points=points, current=current, length=length, voltage=voltage, conc_KCL=conc_KCL, gain=gain)

print pd.DataFrame(finalTable)
