#%%
'''
# Author: Nirajan Luintel
# Date: 2024-06-13
# Created with: Visual Studio Code
# Purpose: To calculate anomalies of vegetation index to use as input for ml model
Mamba Environment: climers
'''
#%%
"""
Discarded anomaly calculation for S1 data because 
it's not regular timeseries data over multiple years
Each file is for each year, and there few fields for
which data is available over multiple years (7yr-7, 6yr-32,5yr-106 )
"""
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
import warnings
import pathlib
import geopandas as gpd
# %%
class anomaly_calulation:
    """
    This class takes in input data time series in nc file
    and calculates the anomalies for each of the vegetation indexes for that time step
    """
    def __init__(self, filename):
        self.infile = filename
        self.ds = xr.open_dataset(filename)
        self.ds.close()

    def calc_mean(self):
        ds_mean = self.ds.groupby('time.dayofyear').mean(skipna = True)
        return ds_mean

    def calc_anomaly(self):
        ds_mean = self.calc_mean()

        anomaly = self.ds.values - self.calc_mean()
        ds_anomaly = self.ds.copy() #should copy the attributes as well
        ds_anomaly.values = anomaly
        self.ds_anomaly = anomaly

    def save_out(self, outfile = None):
        if outfile is None:
            pathdir = os.path.dirname(os.path.dirname(self.infile))
            basename = os.path.basename(self.infile)
            # name, ext = os.path.splitext(basename)
            outfile = os.path.join(pathdir, 'anomalies', basename)

        self.ds_anomaly.attrs['anomaly'] = f'Anomaly from long term mean at that day of year'
        self.ds_anomaly.to_netcdf(outfile)

    def process_file(self, outfile = None):
        self.calc_anomaly()
        self.save_out(outfile)
        return outfile

# def process_monthfile(file):
#     file_lib = pathlib.Path(file)
#     basename = file_lib.name
#     path = str(file_lib.parent)
#     outpath = os.path.join(path, 'anomaly')
#     # filename, ext = os.path.splitext(file)
#     outfile = os.path.join(outpath, basename)
#     # Initialize resample object
#     anomaly_init = anomaly_calulation(file, 'month')
#     anomaly_init.calc_anomaly()
    
#     anomaly_init.save_out(outfile)

#     return outfile
#%%
#specify the working directory as well because the files would flood the code folder
working_dir = '/data/yipeeo_wd'
#and change the directory in the system to make it effective else it does not change anything
os.chdir(working_dir)

parent_dir = '/data/yipeeo_wd'
yield_fields = ['madrid', 'lleida']

# for yf in yield_fields[:1]: #when : is used it is still a list
yf = 'spain'
yield_data_file = os.path.join(parent_dir, '07_data','Crop yield', 'Database', f'field_scale_{yf}.shp')
s1_file_path = os.path.join(parent_dir, 'Data', 'Predictors', 'eo_ts', 's1', 'daily')
s2_file_path = os.path.join(parent_dir, 'Data', 'Predictors', 'eo_ts', 's2', 'Spain', 'spain')
#ecostress_file_path = os.path.join(parent_dir, 'Predictors', 'eo_ts', 'ECOSTRESS')

#Later: deep learn, ECOSTRESS, other numbers of features for FS-> self optimize number of features
pd.set_option('display.max_columns', 15)
warnings.filterwarnings('ignore')
#Later: deep learn, ECOSTRESS, other numbers of features for FS-> self optimize number of features
start_pro = datetime.datetime.now()
print(f'Process for {yf} started at {start_pro}')
# %%
s1_filelist = glob.glob(os.path.join(s1_file_path,'*cr_agg_10-day.nc'))
s2_filelist = glob.glob(os.path.join(s2_file_path,'*.nc'))

s1_file = s1_filelist[0]
s2_file = s2_filelist[0]
#%%
s1_init = anomaly_calulation(s1_file)
s1_mean = s1_init.calc_mean()
s2_init = anomaly_calulation(s2_file)
s2_mean = s2_init.calc_mean()
# %%
