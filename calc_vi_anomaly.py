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
        #just groupby and take mean while skipping nans
        ds_mean = self.ds.groupby('time.dayofyear').mean(skipna = True)
        return ds_mean

    def calc_anomaly(self):
        ds_mean = self.calc_mean() #get means
        #get how many times the data has to be repeated (full-year only)
        rep = len(np.unique(self.ds['time.year']))
        #initialize the anomaly dataset by original dataset
        #then replace  the values with calculated anomalies
        ds_anomaly = self.ds.copy()

        for var in ds_mean.data_vars:
            if var == 'cloud_cover':
                continue
            #get only data as numpy array
            var_mean = ds_mean[var].data
            #repeat the mean for each year tlo match timeseries dimension
            var_mean_tile = np.tile(var_mean, rep)
            #calculate the difference
            var_anomaly = self.ds[var].data - var_mean_tile
            #save it to anomaly dataset
            ds_anomaly[var].data = var_anomaly
        
        # ds_anomaly.attrs['anomaly'] = 'Anomalies from  {rep} years" mean, anomalies invalid for cloud cover'
        self.ds_anomaly = ds_anomaly

    def save_out(self, outfile = None):
        if outfile is None:
            pathdir = os.path.dirname(os.path.dirname(self.infile))
            basename = os.path.basename(self.infile)
            # name, ext = os.path.splitext(basename)
            outfile = os.path.join(pathdir, 'anomalies', basename)

        self.ds_anomaly.attrs['anomaly'] = f'Anomaly from long term mean at that day of year except for cloud cover'
        self.ds_anomaly.to_netcdf(outfile)

#initially process file was within the class 
def process_file(infile, outfile = None):
    self = anomaly_calulation(infile)
    self.calc_anomaly()
    self.save_out(outfile)

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
s2_file_path = os.path.join(parent_dir, 'Data', 'Predictors', 'eo_ts', 's2', 'Spain', 'spain')
#ecostress_file_path = os.path.join(parent_dir, 'Predictors', 'eo_ts', 'ECOSTRESS')

#Later: deep learn, ECOSTRESS, other numbers of features for FS-> self optimize number of features
pd.set_option('display.max_columns', 15)
warnings.filterwarnings('ignore')

# %%
s2_filelist = glob.glob(os.path.join(s2_file_path,'*.nc'))
# s2_file = s2_filelist[0]
#%%
# s2_init = anomaly_calulation(s2_file)
# s2_mean = s2_init.calc_mean()
# s2_init.calc_anomaly()
# print(s2_init.ds_anomaly)
# %%
# for dv in s2_mean.data_vars:
#     s2_data = s2_mean[dv].data
#     s2_data_tile = np.tile(s2_data, 1)
#     s2_ana = s2_init.ds[dv].data - s2_data_tile
# # %%
# s2_ana_ds = s2_init.ds.copy()
# s2_ana_ds[dv].data = s2_ana
# %%
# f = '/data/yipeeo_wd/Data/Predictors/eo_ts/s2/Spain/spain/ES_8_3254621_35.nc'
# with xr.open_dataset(f) as ds:
#     print(ds)
#     plt.plot(ds['ndvi'].data)
# %%
start_pro = datetime.datetime.now()
print(f'Process started at {start_pro}')

with Pool() as pool:
    list(tqdm(pool.imap(process_file, s2_filelist), total = len(s2_filelist)))

stop_pro = datetime.datetime.now()
print(f'Process finished at {stop_pro}')
print(f'Processing took {(stop_pro - start_pro).seconds} seconds')