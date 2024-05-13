#%%
'''
# Author: Nirajan Luintel
# Date: 2024-05-07
# Created with: Visual Studio Code
# Purpose: To get the 10 daily data from daily data
Mamba Environment: climers
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

'''
#%%
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import glob
import os
import datetime
#%%
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

#get the list of directories with "nc" folder
nc_directories = find_directories(s2_spain_dir, "nc")
print("Directories named 'nc':", nc_directories)
#now remove the path to indata_parent_dir from absolute path
#and create folders in the same structure as the input data 
# but in the output directory
def get_rel_path(abs_path, parent_path):
    rel_path = os.path.relpath(abs_path, parent_path)
    return rel_path
#%%
nc_rel_path = [get_rel_path(ncpath, indata_parent_dir) for ncpath in nc_directories]
out_abs_path = [os.path.join(out_parent_dir, ncpath) for ncpath in nc_rel_path]

# %%
#define a function to resample the data based on time aggregation
#since best value differs for variables we do that separately
class resample:
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
            best_var = self.ds[var].resample(time = f'{self.sampling_rate}D').min()
        else:
            best_var = self.ds[var].resample(time = f'{self.sampling_rate}D').median()
        self.ds_out[var] = best_var

    def save_dsout(self, outfile):
        self.ds_out.attrs.update(self.ds.attrs)
        self.ds_out.to_netcdf(outfile)
#%%
# def best_val(ds, var, maxmin = None, sampling_rate = 10):
#     #calculate either maximum of minimum based on input
#         if maxmin == 'max':
#             best_var = ds[var].resample(time = f'{sampling_rate}D').max()
#         elif maxmin == 'min':
#             best_var = ds[var].resample(time = f'{sampling_rate}D').min()
#         else:
#             best_var = ds[var].resample(time = f'{sampling_rate}D').median()
#         return best_var

# def best_val(ds, var, maxmin = None, sampling_rate = 10):
#     #calculate either maximum of minimum based on input
#         if maxmin == 'max':
#             best_var = ds[var].rolling(time = sampling_rate).max()
#         elif maxmin == 'min':
#             best_var = ds[var].resample(time = sampling_rate).min()
#         else:
#             best_var = ds[var].resample(time = sampling_rate).median()
#         return best_var
# %%
sampling_rate = 30 #in days
vars = ['cloud_cover', 'ndvi', 'evi', 'ndwi', 'nmdi']
maxmins = ['min', 'max', 'max', 'min', 'unknown']
var_maxmin_dict = dict(zip(vars, maxmins))
# %%
temp_nc = glob.glob(os.path.join(nc_directories[0], '*.nc'))[0]
#resample_nc.best_val('ndvi', 'max', 10)
ds = xr.open_dataset(temp_nc)
ds.close()
#ds = ds[vars]
#%%
resample_nc = resample(temp_nc, 30)
for key, value in var_maxmin_dict.items():
    resample_nc.best_val(key, value)

# resample_nc.best_val('ndvi', 'max')
# resample_nc.best_val('nmdi', 'min')
# resample_nc.best_val('evi', 'max')
#resample_nc.save_dsout('delete.nc')
#resample_nc = best_val(ds, 'ndvi', 'max', sampling_rate)
print(resample_nc)

#%%
# #ds = ds[vars]
# # ds_roll_max = ds.rolling(time = sampling_rate).max(skipna = True, keep_attrs = True)
# # ds_roll_min = ds.rolling(time = sampling_rate).min(skipna = True)
# # ds_roll_med = ds.rolling(time = sampling_rate).median(skipna = True)
# ds_roll_max = ds.rolling(time = sampling_rate).max(skipna = True)
# ds_roll_min = ds.rolling(time = sampling_rate).min(skipna = True)
# ds_roll_med = ds.rolling(time = sampling_rate).median(skipna = True)
# ds_roll_sel = ds_roll_max.copy()
# # ds_roll_sel['cloud_cover'] = ds_roll_min['cloud_cover']
# # ds_roll_sel['ndwi'] = ds_roll_min['ndwi']
# # ds_roll_sel['nmdi'] = ds_roll_med['nmdi']
# #ds_roll_sel = ds_roll_sel.sel(time = slice(ds_roll_sel.time[sampling_rate-1], None, sampling_rate))
# #%%
# plt.plot(ds.ndvi.data[~np.isnan(ds.ndvi.data)])
# print('nothing')
# plt.plot(np.unique(ds_roll_max.ndvi.data))
# %%
ndvi = ds['ndvi']
ndvi_resample = ndvi.resample(time = '30D').mean(skipna = True)
plt.plot(ndvi_resample.data)
# %%
