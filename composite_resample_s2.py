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
    cloud cover: min
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
#%% set up working environment
#set working directory
wd = '/data/yipeeo_wd'
os.chdir(wd)

#get parent directory for all the work
parent_directory = '/data/yipeeo_wd'
indata_parent_dir = os.path.join(parent_directory, '07_data')
s2_spain_dir = os.path.join(indata_parent_dir, 'Predictors', 'eo_ts', 's2', 'Spain')
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
    def __init__(self, filename, sampling_rate = 10):
        self.sampling_rate = sampling_rate
        self.ds = xr.open_dataset(filename)
        self.ds.close()
        self.ds_out = xr.Dataset()
    
    def best_val(self, var, maxmin = None):
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
        self.ds_out[var] = best_var
        self.ds_out[var].attrs['composite rule'] = maxmin

    def save_dsout(self, outfile):
        self.ds_out.attrs.update(self.ds.attrs)
        self.ds_out.attrs['time stamp'] = 'The start date from when the aggregation began'
        self.ds_out.attrs['aggregation'] = f'Aggregate with composite rule at {sampling_rate}-daily'
        self.ds_out.to_netcdf(outfile)
#%%
def process_file_s2(file):
    basename = os.path.basename(file)
    # Construct output file path
    outfile = os.path.join(out_abs_path, basename)
    # Initialize resample object
    resample_init = resample_s2(file, sampling_rate)
    # Perform resampling for each variable
    for key, value in var_maxmin_dict.items():
        resample_init.best_val(key, value)
    # Save resampled dataset
    resample_init.save_dsout(outfile)

#%%
#define the variables to be used 
sampling_rate = 10 #in days
vars = ['cloud_cover', 'ndvi', 'evi', 'ndwi', 'nmdi']
maxmins = ['min', 'max', 'max', 'min', 'unknown']
var_maxmin_dict = dict(zip(vars, maxmins))
# %%

#get the list of directories with "nc" folder
nc_directories = find_directories(s2_spain_dir, "nc")
#print("Directories named 'nc':", nc_directories)

# #get relativ path and then define the output folders retaining the same path structure
# nc_rel_path = [get_rel_path(ncpath, indata_parent_dir) for ncpath in nc_directories]
# out_abs_path = [os.path.join(out_parent_dir, ncpath) for ncpath in nc_rel_path]

#loop through the nc directories
for folder in nc_directories:
    #get relative path of the subfolders
    in_rel_path = get_rel_path(folder, indata_parent_dir)
    #construct path to follow the same sequence as earlier
    out_abs_path = os.path.join(out_parent_dir, in_rel_path)
    #make directories if not present
    if not os.path.exists(out_abs_path):
        os.makedirs(out_abs_path)
    #get the list of all nc file in the input folder
    filelist = sorted(glob.glob(os.path.join(folder, '*.nc')))
    #print(len(filelist))
    #loop through each file in the folder

    # for file in tqdm(filelist):
    #     basename = os.path.basename(file)
    #     #filename = basename[:-3] + f'_{sampling_rate}D.nc'
    #     outfile = os.path.join(out_abs_path, basename)
    #     # print(file, '\n', outfile)
    #     resample_init = resample(file, sampling_rate=10)
    #     for key, value in var_maxmin_dict.items():
    #         resample_init.best_val(key, value)
    #     resample_init.save_dsout(outfile)

    with Pool() as pool:
        list(tqdm(pool.imap(process_file_s2, filelist), total = len(filelist)))

# %%
