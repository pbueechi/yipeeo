#%%
'''
# Author: Nirajan Luintel
# Date: 2024-05-07
# Created with: Visual Studio Code
# Purpose: To get the 10 daily data from daily data
Mamba Environment: yipeeo
Caution! Be aware of the xarray version with pandas version, 
because xarray resampling relies on pandas and I wasted two days
trying to find out bug in the code while the library was problem
'''
#%%
'''
There were too many data points in daily data with nans so I decide to resample them
by using the best observation in 10 days time period which I would then interpolate and smoothen
For that I identified what "best" actually means for each of the 4 vegetation indices
Some searches and ChatGPT (several re-questioning) hints the best observation for each index are:
    ndvi: max
    evi: max
    ndwi: min
    cloud cover: min %but the cloud cover value is not for field and is for whole scene (tile)
    nmdi: could not decide so taking median
'''
#%% import libraries
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import glob
import os
import datetime
from tqdm import tqdm
from multiprocessing import Pool
from scipy.signal import savgol_filter
#%% set up working environment
#set working directory
wd = '/data/yipeeo_wd'
os.chdir(wd)

#get parent directory for all the work
parent_directory = '/data/yipeeo_wd'
indata_parent_dir = os.path.join(parent_directory, '07_data')
s2_spain_dir = os.path.join(indata_parent_dir, 'Predictors', 'eo_ts', 's2', 'Spain')
s1_folder = os.path.join('Predictors', 'eo_ts', 's1', 'daily')
s1_dir = os.path.join(indata_parent_dir, s1_folder)
out_parent_dir = os.path.join(parent_directory, 'Data')
#%%
#define a function to get the list of all the subdirectories with netcdf data in "nc" folder
def find_directories(root_dir, target_dir):
    nc_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_dir in dirnames:
            nc_directories.append(os.path.join(dirpath, target_dir))
    return nc_directories

#now define a function to remove the path to indata_parent_dir from absolute path
#and create folders in the same structure as the input data 
# but in the output directory
def get_rel_path(abs_path, parent_path):
    rel_path = os.path.relpath(abs_path, parent_path)
    return rel_path


# %%
#define a function to resample the data based on time aggregation
#since best value differs for variables we do that separately
class resample_s2:
    """
    Get the best value composites from each variable
    """
    def __init__(self, filename, sampling_rate = 10, sgol_order = 3, sgol_len = 5):
        self.sampling_rate = sampling_rate
        self.sgol_order = sgol_order
        self.sgol_len = sgol_len
        self.ds = xr.open_dataset(filename)
        self.ds.close()
        self.ds_out = xr.Dataset()
    
    def best_val(self, var, maxmin = None, sgol_order = 3):
    #calculate either maximum of minimum based on input
        if maxmin == 'max':
            best_var = self.ds[var].resample(time = f'{self.sampling_rate}D').max(skipna = True, keep_attrs = True)
        elif maxmin == 'min':
            best_var = self.ds[var].resample(time = f'{self.sampling_rate}D').min(skipna = True, keep_attrs = True)
        elif maxmin == 'median':
            best_var = self.ds[var].resample(time = f'{self.sampling_rate}D').median(skipna = True, keep_attrs = True)
        else:
            #print(f'Unknown compositing rule for {var}: Using mean')
            best_var = self.ds[var].resample(time = f'{self.sampling_rate}D').mean(skipna = True, keep_attrs = True)
            
        #interpolate nans
        
        best_var = best_var.ffill(dim = 'time')
        best_var = best_var.bfill(dim = 'time')
        best_var = best_var.interpolate_na(dim='time', method='linear')
        # apply sgolay
        smoothed_data = savgol_filter(best_var.values, window_length=self.sgol_len, polyorder=self.sgol_order, axis=0)
        best_var.values = smoothed_data

        self.ds_out[var] = best_var
        self.ds_out[var].attrs['composite rule'] = maxmin

    def save_dsout(self, outfile):
        self.ds_out.attrs.update(self.ds.attrs)
        self.ds_out.attrs['time stamp'] = 'The start date from when the aggregation began'
        self.ds_out.attrs['aggregation'] = f'Aggregate with composite rule at {self.sampling_rate}-daily'
        self.ds_out.attrs['smoothing'] = f'Smoothing with savitzky-golay filter at {self.sgol_len} window and {self.sgol_order} polynomial on resampled daily'
        self.ds_out.to_netcdf(outfile)

# similarly for sentinel 1 data as well
#define a function to resample the data based on time aggregation
#since best value is same median for all the variables the function is shorter
class resample_s1:
    """
    Get the best value composites from each variable
    """
    def __init__(self, filename, sampling_rate = 10):
        self.sampling_rate = sampling_rate
        self.ds = xr.open_dataset(filename)
        self.ds.close()
        #self.ds_out = xr.Dataset()

    def best_val(self):
    #calculate either maximum of minimum based on input
        #print(f'Unknown compositing rule for {var}: Using mean')
        best_var = self.ds.resample(time = f'{self.sampling_rate}D').median(skipna = True, keep_attrs = True)
        #if some data missing apply linear interpolation
        best_var = best_var.ffill(dim = 'time')
        best_var = best_var.bfill(dim = 'time')
        best_var = best_var.interpolate_na(dim='time', method='linear')
        self.ds_out = best_var

    def save_dsout(self, outfile):
        self.ds_out.attrs.update(self.ds.attrs)
        self.ds_out.attrs['composite rule'] = 'median'
        self.ds_out.attrs['time stamp'] = 'The start date from when the aggregation began'
        self.ds_out.attrs['aggregation'] = f'Aggregate with composite rule at {sampling_rate}-daily'
        
        self.ds_out.to_netcdf(outfile)
#%%
#write function to feed into parallel process
def process_file_s2(file):
    basename = os.path.basename(file)
    # Construct output file path
    outfile = os.path.join(s2_out_abs_path, basename)
    # Initialize resample object
    resample_init = resample_s2(file, sampling_rate)
    # Perform resampling for each variable
    for key, value in var_maxmin_dict.items():
        resample_init.best_val(key, value)
    # Save resampled dataset
    resample_init.save_dsout(outfile)

    return outfile
#%%
#write similar function for s1 as well
def process_file_s1(file):
    basename = os.path.basename(file)
    # Construct output file path
    outfile = os.path.join(s1_out_abs_path, basename[:-8]+f'{sampling_rate}-day.nc')
    # Initialize resample object
    resample_init = resample_s1(file, sampling_rate)
    # Perform resampling for each variable
    resample_init.best_val()
    # Save resampled dataset
    resample_init.save_dsout(outfile)

#%%
#define the variables to be used for s2 sampling
sampling_rate = 10 #in days
vars = ['cloud_cover', 'ndvi', 'evi', 'ndwi', 'nmdi']
maxmins = ['min', 'max', 'max', 'min', 'median']
var_maxmin_dict = dict(zip(vars, maxmins))
# %% process s2

#get the list of directories with "nc" folder
nc_directories = find_directories(s2_spain_dir, "nc")
#print("Directories named 'nc':", nc_directories)

#loop through the nc directories
for folder in nc_directories:
    #get relative path of the subfolders
    s2_in_rel_path = get_rel_path(folder, indata_parent_dir)
    #construct path to follow the same sequence as earlier
    s2_out_abs_path = os.path.join(out_parent_dir, s2_in_rel_path)
    #make directories if not present
    if not os.path.exists(s2_out_abs_path):
        os.makedirs(s2_out_abs_path)
    #get the list of all nc file in the input folder
    filelist = sorted(glob.glob(os.path.join(folder, '*.nc')))
    print(len(filelist))
    #perform the task in parallel
    with Pool() as pool:
        list(tqdm(pool.imap(process_file_s2, filelist), total = len(filelist)))

# %% get s1 files
s1_out_abs_path = os.path.join(out_parent_dir, s1_folder)
if not os.path.exists(s1_out_abs_path):
    os.makedirs(s1_out_abs_path)
s1_filelist = sorted(glob.glob(os.path.join(s1_dir, 'ES*.nc')))
#%% Process s1
with Pool() as pool:
    list(tqdm(pool.imap(process_file_s1, s1_filelist), total = len(s1_filelist)))

# %%

# infile = '/data/yipeeo_wd/07_data/Predictors/eo_ts/s2/Spain/lleida/nc/ES_8_2784618_9.nc'

# dsin = xr.open_dataset(infile)
# dsin.close()
# evi_in = dsin.evi.values
# evi_in_nan = np.where(np.isnan(evi_in))
# #%%
# out = process_file_s2(infile)
# ds = xr.open_dataset(out)
# ds.close()
# evi_val = ds.evi.values
# evi_nans = np.where(np.isnan(evi_val))
# %%
