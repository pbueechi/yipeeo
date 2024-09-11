#%%
'''
# Author: Nirajan Luintel
# Date: 2024-07-12
# Created with: Visual Studio Code
# Purpose: To apply hants to the timeseries VI because s-g filter still had some noise 
I expect HANTS to smoothen out all the noises, code from chatgpt
Mamba Environment: yipeeo
'''
#%%
import numpy as np
import scipy.fft

from scipy.optimize import curve_fit

from scipy.signal import find_peaks_cwt
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal

import pandas as pd
import math
import numpy as np
from copy import deepcopy
import warnings
from matplotlib import markers, pyplot as plt

import xarray as xr
#%%
def hants(time_series, n_harmonics=10, low_pass_filter=0.1):
    n = len(time_series)
    t = np.arange(n)

    # Fourier transform
    fft_result = scipy.fft.fft(time_series)
    fft_freqs = scipy.fft.fftfreq(n)

    # Filter out high frequencies
    filtered_fft = np.zeros_like(fft_result)
    filtered_fft[np.abs(fft_freqs) <= low_pass_filter] = fft_result[np.abs(fft_freqs) <= low_pass_filter]

    # Inverse Fourier transform to get the smoothed time series
    smoothed_series = scipy.fft.ifft(filtered_fft).real

    # Use the first `n_harmonics` frequencies for the harmonic model
    harmonics = [scipy.fft.ifft(filtered_fft * (np.abs(fft_freqs) == f)).real for f in range(1, n_harmonics + 1)]

    # Fit a curve using the harmonics
    def harmonic_model(t, *params):
        a0 = params[0]
        result = a0
        for i in range(n_harmonics):
            ai = params[2 * i + 1]
            bi = params[2 * i + 2]
            result += ai * np.cos(2 * np.pi * (i + 1) * t / n) + bi * np.sin(2 * np.pi * (i + 1) * t / n)
        return result



    # Initial guess for parameters
    initial_guess = [np.mean(time_series)] + [0] * 2 * n_harmonics
    # return t, time_series
    # Fit the model to the time series
    params, _ = curve_fit(harmonic_model, t, time_series, p0=initial_guess)

    # Construct the fitted curve
    fitted_curve = harmonic_model(t, *params)

    return smoothed_series, fitted_curve
#%%
def BISE(x, slide_period = 5, slope_threshold = 0.2):
    slope_threshold_value = 0
    days = len(x)
    x = np.concatenate((x,x,x)).flatten()
    cor_x = np.zeros((len(x),1))
    
    for i in range (2,3*days):
        if x[i] >= x[i-1]:
            cor_x[i] = x[i]
        else:
            if (i+slide_period) > 3*days-1:
                period = 3*days - i -1
            else:
                period = slide_period
            
            slope_threshold_value = x[i] +slope_threshold * (x[i-1] - x[i])
            bypassed_elems = 0
            ndvi_chosen = 0
            
            for j in range(i+1, i+period-1):
                if (x[j] > slope_threshold_value) and (x[j] > ndvi_chosen):
                    ndvi_chosen = x[j]
                    bypassed_elems = j-i
                
                if ndvi_chosen >= x[i-1]:
                    break
                
            if ndvi_chosen == 0:
                cor_x[i] = x[i]
            else:
                for j in range(0, bypassed_elems):
                    cor_x[i-1+j] = -1
                i = i + bypassed_elems
                cor_x[i] = ndvi_chosen
        
    cor_x = cor_x [days:days*2]
    
    #i excluded the last part because it will replace all the -1 with the previous values
        
    return cor_x

#%%
def HANTS(ni, nb, nf, y, ts, HiLo, low, high, fet, dod, delta, fill_val):
    '''
    This function applies the Harmonic ANalysis of Time Series (HANTS)
    algorithm originally developed by the Netherlands Aerospace Centre (NLR)
    (http://www.nlr.org/space/earth-observation/).

    This python implementation was based on two previous implementations
    available at the following links:
    https://codereview.stackexchange.com/questions/71489/harmonic-analysis-of-time-series-applied-to-arrays
    http://nl.mathworks.com/matlabcentral/fileexchange/38841-matlab-implementation-of-harmonic-analysis-of-time-series--hants-
    '''
    y[np.isnan(y)] = fill_val
    # Arrays
    mat = np.zeros((min(2*nf+1, ni), ni))
    # amp = np.zeros((nf + 1, 1))

    # phi = np.zeros((nf+1, 1))
    yr = np.zeros((ni, 1))
    y_len = len(y)
    outliers = np.zeros((1, y_len))

    # Filter
    sHiLo = 0
    if HiLo == 'Hi':
        sHiLo = -1
    elif HiLo == 'Lo':
        sHiLo = 1

    nr = min(2*nf+1, ni)
    noutmax = ni - nr - dod
    # dg = 180.0/math.pi
    mat[0, :] = 1.0

    ang = 2*math.pi*np.arange(nb)/nb
    cs = np.cos(ang)
    sn = np.sin(ang)

    i = np.arange(1, nf+1)
    for j in np.arange(ni):
        index = np.mod(i*ts[j], nb)
        mat[2 * i-1, j] = cs.take(index)
        mat[2 * i, j] = sn.take(index)

    p = np.ones_like(y)
    bool_out = (y < low) | (y > high)
    p[bool_out] = 0
    outliers[bool_out.reshape(1, y.shape[0])] = 1
    nout = np.sum(p == 0)

    if nout > noutmax:
        if np.isclose(y, fill_val).any():
            ready = np.array([True])
            yr = y
            outliers = np.zeros((y.shape[0]), dtype=int)
            outliers[:] = fill_val
        else:
            raise Exception('Not enough data points.')
    else:
        ready = np.zeros((y.shape[0]), dtype=bool)

    nloop = 0
    nloopmax = ni

    while ((not ready.all()) & (nloop < nloopmax)):

        nloop += 1
        za = np.matmul(mat, p*y)

        A = np.matmul(np.matmul(mat, np.diag(p)),
                         np.transpose(mat))
        A = A + np.identity(nr)*delta
        A[0, 0] = A[0, 0] - delta

        zr = np.linalg.solve(A, za)

        yr = np.matmul(np.transpose(mat), zr)
        diffVec = sHiLo*(yr-y)
        err = p*diffVec

        err_ls = list(err)
        err_sort = deepcopy(err)
        err_sort.sort()

        rankVec = [err_ls.index(f) for f in err_sort]

        maxerr = diffVec[rankVec[-1]]
        ready = (maxerr <= fet) | (nout == noutmax)

        if (not ready):
            i = ni - 1
            j = rankVec[i]
            while ((p[j]*diffVec[j] > 0.5*maxerr) & (nout < noutmax)):
                p[j] = 0
                outliers[0, j] = 1
                nout += 1
                i -= 1
                if i == 0:
                    j = 0
                else:
                    j = 1

    return [yr, outliers]

#%%
# Example usage
import numpy as np
time_series = np.array([1., 2, -9, -9, 5, 4, 3, 2, 1, -9, 1, 2, 4, 5, 4, 3, 2, 1, 0, 1, 2, -9, -9, -9, 4, 3, 2, 1, 0])

# time_series[time_series == -9] = np.nan
# time_series = 
#%%
smoothed_series, fitted_curve = hants(time_series, n_harmonics=5, low_pass_filter=0.1)

print("Smoothed Series:", smoothed_series)
print("Fitted Curve:", fitted_curve)

#%%
# HANTS parameters
ni = len(time_series)
nb = ni
nf = 2
low = -1
high = 1
HiLo = 'Hi'
fet = 0.05
delta = 0.1
dod = 1
fill_val = -9
out = HANTS(ni, nb, nf, time_series, np.arange(ni), HiLo, low, high, fet, dod, delta, fill_val)
plt.plot(out[0])
plt.plot(time_series)
# %%
# https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Tools/deafrica_tools/temporal.py
# https://www.sciencedirect.com/science/article/pii/S0034425709001746
#%%
# import hdstats

# from hdstats import number_peaks
# n = number_peaks(time_series)
# %%
n = find_peaks_cwt(time_series, np.arange(5,10))
# %%

time_series = gaussian_filter1d(time_series, np.std(time_series))
tmax = signal.argrelmax(time_series)
tmin = signal.argrelmin(time_series)

plt.plot(time_series)
#%%
# data =np.array([5.14,5.22,5.16,4.82,4.46,4.36,4.4,4.35,4.13,3.83,3.59,3.51,3.46,3.27,3.08,3.03,2.95,2.96,2.98,3.02,3.09,3.14,3.06,2.84,2.68,2.72,2.92,3.23,3.44,3.5,3.28,3.34,3.73,3.97,4.26,4.48,4.5,5.06,6.02,6.68,7.09,7.58,8.6,9.85,10.7,11.3,11.3,11.6,12.3,12.6,12.8,12.8,12.5,12.4,12.2,12.2,12.3,11.9,11.2,10.6,10.3,10.3,10.,9.53,8.97,8.55,8.49,8.41,8.09,7.71,7.34,7.26,7.42,7.47,7.37,7.17,7.05,7.02,7.09,7.23,7.18,7.16,7.47,7.92,8.55,8.68,8.31,8.52,9.11,9.59,9.83,9.73,10.2,11.1,11.6,11.7,11.7,12.,12.6,13.1,13.3,13.2,13.,12.6,12.3,12.2,12.3,12.,11.6,11.1,10.9,10.9,10.7,10.3,9.83,9.64,9.63,9.37,8.88,8.39,8.14,8.12,7.92,7.48,7.06,6.87,6.87,6.63,6.17,5.71,5.45,5.45,5.34,5.05,4.78,4.57,4.47,4.37,4.16,3.95,3.88,3.83,3.69,3.64,3.57,3.5,3.51,3.33,3.14,3.09,3.06,3.12,3.11,2.94,2.83,2.76,2.74,2.77,2.75,2.73,2.72,2.59,2.47,2.53,2.54,2.63,2.76,2.78,2.75,2.69,2.54,2.42,2.58,2.79,2.83,2.78,2.71,2.77,2.88,2.97,2.97,2.9,2.92,3.16,3.29,3.28,3.49,3.97,4.32,4.49,4.82,5.08,5.48,6.03,6.52,6.72,7.16,8.18,9.52,10.9,12.1,12.6,12.9,13.3,13.3,13.6,13.9,13.9,13.6,13.3,13.2,13.2,12.8,12.,11.4,11.,10.9,10.4,9.54,8.83,8.57,8.61,8.24,7.54,6.82,6.46,6.43,6.26,5.78,5.29,5.,5.08,5.14,5.,4.84,4.56,4.38,4.52,4.84,5.33,5.52,5.56,5.82,6.54,7.27,7.74,7.64,8.14,8.96,9.7,10.2,10.2,10.5,11.3,12.,12.4,12.5,12.3,12.,11.8,11.8,11.9,11.6,11.,10.3,10.,9.98,9.6,8.87,8.16,7.76,7.74,7.54,7.03,6.54,6.25,6.26,6.09,5.66,5.31,5.08,5.19,5.4,5.38,5.38,5.22,4.95,4.9,5.02,5.28,5.44,5.93,6.77,7.63,8.48,8.89,8.97,9.49,10.3,10.8,11.,11.1,11.,11.,10.9,11.1,11.1,11.,10.7,10.5,10.4,10.3,10.4,10.3,10.2,10.1,10.2,10.4,10.4,10.5,10.7,10.8,11.,11.2,11.2,11.2,11.3,11.4,11.4,11.3,11.2,11.2,11.,10.7,10.4,10.3,10.3,10.2,9.9,9.62,9.47,9.46,9.35,9.12,8.82,8.48,8.41,8.61,8.83,8.77,8.48,8.26,8.39,8.84,9.2,9.31,9.18,9.11,9.49,9.99,10.3,10.5,10.4,10.2,10.,9.91,10.,9.88,9.47,9.,8.78,8.84,8.8,8.55,8.17,8.02,8.03,7.78,7.3,6.8,6.54,6.53,6.35,5.94,5.54,5.33,5.32,5.14,4.76,4.43,4.28,4.3,4.26,4.11,4.,3.89,3.81,3.68,3.48,3.35,3.36,3.47,3.57,3.55,3.43,3.29,3.19,3.2,3.17,3.21,3.33,3.37,3.33,3.37,3.38,3.26,3.34,3.62,3.86,3.92,3.83,3.69,4.2,4.78,5.03,5.13,5.07,5.4,6.,6.42,6.5,6.45,6.48,6.55,6.66,6.79,7.06,7.33,7.53,7.9,8.17,8.29,8.6,9.05,9.35,9.51,9.69,9.88,10.2,10.6,10.8,10.6,10.7,10.9,11.2,11.3,11.3,11.4,11.5,11.6,11.8,11.7,11.3,11.1,10.9,11.,11.2,11.1,10.6,10.3,10.1,10.2,10.,9.6,9.03,8.73,8.73,8.7,8.53,8.26,8.06,8.03,8.03,7.97,7.94,7.77,7.64,7.85,8.29,8.65,8.68,8.61,9.08,9.66,9.86,9.9,9.71,10.,10.9,11.4,11.6,11.8,11.8,11.9,11.9,12.,12.,11.7,11.3,10.9,10.8,10.7,10.4,9.79,9.18,8.89,8.87,8.55,7.92,7.29,6.99,6.98,6.73,6.18,5.65,5.35,5.35,5.22,4.89,4.53,4.28,4.2,4.05,3.83,3.67,3.61,3.61,3.48,3.27,3.05,2.9,2.93,2.99,2.99,2.98,2.94,2.88,2.89,2.92,2.86,2.97,3.,3.02,3.03,3.11,3.07,3.46,3.96,4.09,4.25,4.3,4.67,5.7,6.33,6.68,6.9,7.09,7.66,8.25,8.75,8.87,8.97,9.78,10.9,11.6,11.8,11.8,11.9,12.3,12.6,12.8,12.9,12.7,12.4,12.1,12.,12.,11.9,11.5,11.1,10.9,10.9,10.7,10.5,10.1,9.91,9.84,9.63,9.28,9.,8.86,8.95,8.87,8.61,8.29,7.99,7.95,7.96,7.92,7.87,7.77,7.78,7.9,7.73,7.51,7.43,7.6,8.07,8.62,9.06,9.24,9.13,9.14,9.46,9.76,9.8,9.78,9.73,9.82,10.2,10.6,10.8,10.8,10.9,11.,10.9,11.,11.,10.9,10.9,11.,10.9,10.8,10.5,10.2,10.2,10.2,9.94,9.51,9.08,8.88,8.88,8.62,8.13,7.64,7.37,7.37,7.23,6.91,6.6,6.41,6.42,6.29,5.94,5.57,5.43,5.46,5.4,5.17,4.95,4.84,4.87,4.9,4.69,4.4,4.24,4.26,4.35,4.34,4.19,3.96,3.97,4.42,5.03,5.34,5.15,4.73,4.86,5.35,5.88,6.35,6.52,6.81,7.26,7.62,7.66,8.01,8.91,10.,10.9,11.3,11.1,10.9,10.9,10.8,10.9,11.,10.7,10.2,9.68,9.43,9.42,9.17,8.66,8.13,7.83,7.81,7.62,7.21,6.77,6.48,6.44,6.31,6.06,5.72,5.47,5.45,5.42,5.31,5.23,5.22,5.3,5.32,5.16,4.96,4.82,4.73,4.9,4.95,4.91,4.92,5.41,6.04,6.34,6.8,7.08,7.26,7.95,8.57,8.78,8.95,9.06,9.14,9.2,9.33,9.53,9.65,9.69,9.53,9.18,9.02,9.,8.82,8.42,8.05,7.85,7.84,7.79,7.58,7.28,7.09,7.07,6.94,6.68,6.35,6.09,6.2,6.27,6.24,6.16,5.91,5.86,6.02,6.19,6.45,6.92,7.35,7.82,8.4,8.87,9.,9.09,9.61,9.99,10.4,10.8,10.7,10.7,11.1,11.4,11.5,11.5,11.3,11.3,11.4,11.7,11.8,11.5,11.,10.5,10.4,10.3,9.94,9.23,8.52,8.16,8.15,7.86,7.23,6.59,6.26,6.25,6.04,5.55,5.06,4.81,4.78,4.62,4.28,3.98,3.84,3.92,3.93,3.68,3.46,3.31,3.16,3.11,3.18,3.19,3.14,3.28,3.3,3.16,3.19,3.04,3.07,3.59,3.83,3.82,3.95,4.06,4.71,5.39,5.89,6.06,6.08,6.45,6.97,7.57,8.1,8.25,8.55,8.92,9.09,9.2,9.32,9.36,9.45,9.65,9.73,9.7,9.82,9.94,9.92,9.97,9.93,9.78,9.63,9.48,9.49,9.48,9.2,8.81,8.34,8.,8.06,7.98,7.63,7.47,7.37,7.24,7.2,7.05,6.93,6.83,6.59,6.44,6.42,6.33,6.18,6.37,6.29,6.1,6.34,6.57,6.54,6.77,7.21,7.58,7.86,8.11,8.57,9.07,9.45,9.67,9.68,9.87,10.2,10.4,10.4,10.4,10.4,10.4,10.5,10.6,10.7,10.4,9.98,9.58,9.45,9.51,9.44,9.09,8.68,8.46,8.36,8.17,7.88,7.55,7.34,7.3,7.17,6.97,6.88,6.69,6.69,6.77,6.77,6.81,6.67,6.5,6.57,6.99,7.4,7.59,7.8,8.45,9.47,10.4,10.8,10.9,10.9,11.,11.4,11.8,12.,11.9,11.4,10.9,10.8,10.8,10.5,9.76,8.99,8.59,8.58,8.43,8.05,7.61,7.26,7.16,6.99,6.58,6.15,5.98,5.93,5.71,5.48,5.22,5.06,5.08,4.95,4.78,4.62,4.45,4.48,4.65,4.66,4.69])
# data = data[:200]
# dataFiltered = gaussian_filter1d(data, sigma=5)
# tMax = signal.argrelmax(dataFiltered)[0]
# tMin = signal.argrelmin(dataFiltered)[0]

# plt.plot(data, label = 'raw')
# plt.plot(dataFiltered, label = 'filtered')
# plt.plot(tMax, dataFiltered[tMax], 'o', mfc= 'none', label = 'max')
# plt.plot(tMin, dataFiltered[tMin], 'o', mfc= 'none', label = 'min')
# plt.legend()
# plt.savefig('fig.png', dpi = 300)
# %%
f = '/data/yipeeo_wd/Data/Predictors/eo_ts/s2/Spain/spain/ES_8_2784618_2.nc'

ds = xr.open_dataset(f)
ds.close()
evi = ds['evi'].data

ndvi = ds['ndvi'].data
# evi = gaussian_filter1d(evi, 1.5)
#%%
evi_peak = signal.argrelmax(evi)[0]
evi_peak = evi_peak[evi[evi_peak]>0.4]
evi_peak_diff = ds.time[evi_peak].diff(dim = 'time').dt.days
evi_peak_diff = np.insert(evi_peak_diff, 0, 0)
evi_peak = evi_peak[evi_peak_diff>60]

# #%%
# #
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(ds.time.data, evi)
# ax.plot(ds.time.data[evi_peak], [0]*len(evi_peak), '*')
#%%
from peakdet import peakdet
evi = ds['evi'].data
# evi = evi2[30:]
evig = gaussian_filter1d(evi, 2)
maxtab, mintab = peakdet(evig,.2)

# if maxtab[0, 0] < mintab[0, 0]:
#     print('it misses the first minima, so invert and replicate it')
#     maxtab_n, mintab_n = peakdet(-evig, 0.2)
#     maxtab_n[:,1] = -maxtab_n[:,1]
#     mintab_n[:,1] = -mintab_n[:,1]
#     maxtab = maxtab_n; mintab = mintab_n
plt.plot(evig)
plt.scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='blue', label = 'max')
plt.scatter(np.array(mintab)[:,0], np.array(mintab)[:,1], color='red', label = 'min')
plt.legend()
#%%
nf = len(maxtab)
ni = len(evi)
nb = ni
low = -1
high = 1
HiLo = 'HiLo'
fet = 0.05
delta = 0.1
dod = 1
fill_val = -9
out = HANTS(ni, nb, nf, evi, np.arange(ni), HiLo, low, high, fet, dod, delta, fill_val)
# plt.plot(ds.time.data, evi)
# plt.plot(ds.time.data, out[0])
plt.plot(evi)
plt.plot(out[0])
plt.show()
# %%

# %%
s1 = '/data/yipeeo_wd/Data/Predictors/eo_ts/s1/daily/ES_8_2784618_2_2020_cleaned_cr_agg_10-day.nc'
ds1 = xr.open_dataset(s1); ds1.close()
print(ds1)
# %%
datavar = list(ds1.data_vars)
vars = [i for i in datavar if 'cr' in i]
print(vars)
# %%
for var in vars:
    plt.plot(ds1['time'].data, ds1[var].data)

# %%
from peakdet_2 import peakdetect
evig = gaussian_filter1d(evi, 1.5)
maxtab, mintab = peakdetect(evig,x_axis = None, lookahead = 20, delta=0.2)

if maxtab[0][0] < mintab[0][0]:
    print('it misses the first minima, so invert and replicate it')
    maxtab_n, mintab_n = peakdetect(-evig, x_axis = None, lookahead = 20, delta=0.2)
    # maxtab_n[:,1] = -maxtab_n[:,1]
    # mintab_n[:,1] = -mintab_n[:,1]
#     maxtab = maxtab_n; mintab = mintab_n
# plt.plot(evig)
# plt.scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='blue')
# plt.scatter(np.array(mintab)[:,0], np.array(mintab)[:,1], color='red')
# %%
