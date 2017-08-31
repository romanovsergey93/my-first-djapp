import argparse
import sys
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.sparse.linalg import spsolve
import pandas as pd
from matplotlib import *
from scipy.optimize import curve_fit
from neo import io


parser = argparse.ArgumentParser(description="link to the ABF file")
parser.add_argument('--link', help='link to the ABF file', type=str, default='F:\\abfs\\17208000.abf')
parser.add_argument('--gain', help='gain', type=int, default=20)
parser.add_argument('--ionicforce', help='ionicforce', type=int, default=500)

def impt():

	args = parser.parse_args()

	read_abf = io.AxonIO(filename=args.link)
	read_blocks = read_abf.read_block(lazy=False, cascade=True)
	reader_data = read_blocks.segments[0].analogsignals

	print("INFO #1")
	print(read_abf.name, read_abf.description, read_abf.extensions, \
	        read_abf.extentions, read_abf.filename, read_abf.has_header, \
	        read_abf.is_readable, read_abf.is_streameable, \
	        read_abf.is_writable, read_abf.logger, read_abf.mode)

	print("INFO #2")
	print(read_blocks.annotations, read_abf.description, \
	        read_blocks.file_datetime, read_blocks.file_origin, \
	        read_blocks.index, read_blocks.name, read_blocks.rec_datetime)

	print("INFO #3")
	print(read_blocks, read_blocks.segments, read_blocks.segments, \
	        read_blocks.segments[0].analogsignals)

	current = np.array(reader_data[0], float)
	length = np.array(reader_data[1], float)
	voltage = np.array(reader_data[2], float)

	current = np.reshape(current, len(current))
	length = np.reshape(length, len(length))
	voltage = np.reshape(voltage, len(voltage))

	gain = args.gain
	conc_KCL = args.ionicforce

	return current, length, voltage, gain, conc_KCL



def fit(points, current, length, voltage, conc_KCL, gain):

    d = {'radius, nm': [], 'error, nm': [], 'voltage, mV': []}

    for point in points:
        l = length[point[0]: point[1]]
        v = voltage[point[0]: point[1]]
        c = current[point[0]: point[1]]
        c = c * 1000 / gain
        l = l * 100 / 1.49
        v = v * 100
        L = (3.95056 - 0.7495 * l + 0.04611 * l ** 2 - 8.04926E-4 * l ** 3 + 6.75729E-6 * l ** 4 - 2.78498E-8 * l ** 5 + 4.52289E-11 * l ** 6)
        G = c * 1000 / v

        y = G
        x = L


        def func(x, p1, p2, p3):
            return p1 / (p2 + x) + p3

        try:
            popt, pcov = curve_fit(func, x, y, p0=[0, 0, G[0]])
            perr = np.sqrt(np.diag(pcov))
            print("p1 = %s + %s, p2 = %s + %s, p3 = %s + %s" % (popt[0], perr[0], popt[1], perr[1], popt[2], perr[2]))

            error = perr[0] / popt[0]
            rho = 1.1 * (conc_KCL / 100)
            radius = np.sqrt (popt[0]/(np.pi*rho))
            error = 0.5 * error * radius

            d['radius, nm'].append(radius)
            d['error, nm'].append(error)
            d['voltage, mV'].append(np.mean(v))

            plt.plot(range(point[0], point[1]), length[point[0]:point[1]], color='r')
            plt.plot(range(point[0]-1000, point[0]), length[point[0]-1000 : point[0]], color='b')
            plt.plot(range(point[1], point[1]+1000), length[point[1] : point[1]+1000], color='b')
            plt.show()

            plt.plot(x, y, 'ro', label="Original Data")
            plt.plot(x, func(x, *popt), label="Fitted Curve")
            #plt.legend(loc='upper left')
            plt.show()

        except RuntimeError:
            print("no fit")
            d['radius, nm'].append('no fit')
            d['error, nm'].append('no fit')
            d['voltage, mV'].append(np.mean(v))

    #print pd.DataFrame(d)

    return d


def smooth():

	lengthSmoothed = length
	for i in range(0, 2):
	    lengthSmoothed = signal.savgol_filter(x=lengthSmoothed, window_length=115, polyorder=2, deriv=0, mode='nearest')

	plt.plot(length)
	plt.plot(lengthSmoothed, color='r')
	plt.show()

	b = signal.get_window('triang', 1000, False)
	print(len(b))

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
	print("base")
	#baseline4correction = signal.resample(baseline4correction, len(length))

	#baseline4correction = signal.resample_poly(baseline4correction, len(lengthSmoothed), len(baseline4correction))
	#baseline4correction = signal.savgol_filter(length, 121, 1)
	print("resamp")
	#length4peaks = lengthSmoothed - baseline4correction
	length4peaks = lengthDecimated - baseline4correction


	#print np.mean(length4peaks)
	#print np.median(length4peaks)
	#print np.max(length4peaks)
	#print np.min(length4peaks)

	plt.plot(lengthDecimated)
	plt.plot(baseline4correction, color='r')
	plt.plot(length4peaks, color='g')
	plt.show()

	minimal_length = np.mean(length4peaks)-np.median(length4peaks)
	minimal_length2 = (np.max(length4peaks)-np.min(length4peaks))/2
	#print minimal_length, minimal_length2

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

	return finalTable

current, length, voltage, gain, conc_KCL = impt()
finalTable = smooth()
print(pd.DataFrame(finalTable))
