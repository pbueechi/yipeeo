#%%
'''
# Author: Nirajan Luintel
# Date: 2024-07-10
# Created with: Visual Studio Code
# Purpose: To fill the gaps (or could be unreliable data) in the field level S2 timeseries VI data 
Mamba Environment: yipeeo
'''
#%%
'''
I tried to use simple 10-day maximum value composite and then Savitzky-Golay filter 
with the hope that it can give me smooth timeseries data which then I would use
for phenology extraction in order to use as an input for yield prediction
But my phenology extraction failed miserably because there were instances where 
the EVI and NDVI curves were flat (probably due to linear interpolation) that
led to skipping the whole crop growing season or gave unreliable start and end
of season data.
In order to improve that I am trying to reconstruct the timeseries data using 
the climatological median as the reference data (similar to what I did for PhenoRice)
I am still skeptical because there is only 7 years of data for reliable median
'''
#%%
import numpy as np
import pandas as pd
import datetime
import os
import xarray as xr
import glob
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

#%%
#use a class to process each file as it's easier
class gap_fill:
    def __init__(self, filename, var_list, outdir, sampling_rate = 10, sgol_order = 2, sgol_len = 9):
        self.filename = filename
        self.sampling_rate = sampling_rate
        self.sgol_order = sgol_order
        self.sgol_len = sgol_len
        self.ds = xr.open_dataset(filename)
        self.ds.close()
        self.ds_out = xr.Dataset() #initializing ds out makes it easier to store result
        self.var_list = var_list #selectively perform the 
        self.outdir = outdir #provide also the output directory for being sure where data is

    def max_val_year(self, ds, var):
        #calculate either maximum of minimum based on input
        # if maxmin == 'max':
        #     best_var = self.ds[var].sel(time = slice(f'{year}-01-01',f'{year}-12-31')).resample(time = f'{self.sampling_rate}D').max(skipna = True, keep_attrs = True)
        # elif maxmin == 'min':
        #     best_var = self.ds[var].sel(time = slice(f'{year}-01-01',f'{year}-12-31')).resample(time = f'{self.sampling_rate}D').min(skipna = True, keep_attrs = True)
        # elif maxmin == 'median':
        #     best_var = self.ds[var].sel(time = slice(f'{year}-01-01',f'{year}-12-31')).resample(time = f'{self.sampling_rate}D').median(skipna = True, keep_attrs = True)
        # else:
        #     #print(f'Unknown compositing rule for {var}: Using mean')
        #     best_var = self.ds[var].resample(time = f'{self.sampling_rate}D').mean(skipna = True, keep_attrs = True)
        # return best_var
        sampling_rate = self.sampling_rate
        best_var = ds[var].resample(time = f'{sampling_rate}D').max(skipna = True, keep_attrs = True)
        return best_var
    
    def min_val_year(self, ds, var):
        """
        Not all the VIs are better with maximum value, so define minimum value composite as well
        """
        sampling_rate = self.sampling_rate
        best_var = ds[var].resample(time = f'{sampling_rate}D').min(skipna = True, keep_attrs = True)
        return best_var
    
    def max_val_ts(self, var):
        '''
        Apply maximum value composite for each year separately
        '''
        grouped = self.ds.groupby('time.year')
        resampled_collection = []
        for yr, group in grouped:
            resampled_year = self.max_val_year(group, var)
            resampled_collection.append(resampled_year)
        best_var = xr.concat(resampled_collection, dim='time')
        return best_var
    
    def min_val_ts(self, var):
        """
        Apply minimum value composite to whole time series year to year
        """
        grouped = self.ds.groupby('time.year')
        resampled_collection = []
        for yr, group in grouped:
            resampled_year = self.min_val_year(group, var)
            resampled_collection.append(resampled_year)
        best_var = xr.concat(resampled_collection, dim='time')
        return best_var
    
    def calc_ref(self, ds):
        """
        calculate the reference value by taking median and 
        then fill the gaps and smoothen the curve
        """
        #just groupby and take mean while skipping nans
        ds_median = ds.groupby('time.dayofyear').median(skipna = True)
        ds_median = ds_median.ffill(dim = 'dayofyear')
        ds_median = ds_median.bfill(dim = 'dayofyear')
        ds_median = ds_median.interpolate_na(dim='dayofyear', method='linear')
        # self.ds_median = ds_median
        # apply sgolay
        smoothed_data = savgol_filter(ds_median.values, window_length=self.sgol_len, polyorder=self.sgol_order, axis=0)
        ds_median.values = smoothed_data
        return ds_median
    
    def gap_fill_var(self, var):
        """
        Perform the gap filling using appropriate compositing for each VI
        """
        if var == 'ndwi' or 'nmdi':
            val_composite = self.min_val_ts(var)
        else:
            val_composite = self.max_val_ts(var)
        # print('val composite ok')
        # self.val_composite = val_composite
        ref_ts = self.calc_ref(val_composite)
        # print('ref ts ok')

        ref_data = ref_ts.values
        # self.ref_data = ref_data

        #repeat the ref data to make it consistent with the time series data
        ntile = np.round(len(val_composite.time.data)/len(ref_data), 0).astype(np.int16)
        # print('ntile = ',ntile)
        ref_data_tile = np.tile(ref_data, ntile)
        # self.ref_data_tile = ref_data_tile

        #instead of creating new dataset copying the already existing dataset
        #  and replacing values is easier because ref data has only dayofyear not time
        #  as dimension as such arithmetic operation fails between ts and ref data
        ref_ds = val_composite.copy()
        ref_ds.values = ref_data_tile
        
        #calculate difference from ref data, fill the gaps in diff and then put it back
        data_diff = val_composite - ref_ds
        data_diff_fill = data_diff.ffill(dim='time')
        data_diff_fill = data_diff_fill.bfill(dim='time')
        data_diff_fill = data_diff_fill.interpolate_na(dim='time', method = 'linear')
        val_reconstruct = ref_ds + data_diff_fill

        #smoothen the curve
        val_smooth = savgol_filter(val_reconstruct.values, window_length=self.sgol_len, polyorder=self.sgol_order, axis=0)
        val_reconstruct.values = val_smooth
        return val_reconstruct
    
    def process_ds(self):
        """
        Apply gap filling to all the variables concerned in dataset
        """
        ds_out = xr.Dataset()
        for var in self.var_list:
            var_filled = self.gap_fill_var(var)
            ds_out[var] = var_filled
            ds_out[var].attrs['method'] = 'filled using climatological median'
        self.ds_out = ds_out

    def save_out(self, outfile = None):
        """
        Save the output to destination directory
        """
        self.ds_out.attrs.update(self.ds.attrs)
        self.ds_out.attrs['method'] = 'filled using climatological median'
        self.ds_out.attrs['aggregation'] = f'Aggregate with composite rule at {self.sampling_rate}-daily'
        self.ds_out.attrs['smoothing'] = f'Smoothing with savitzky-golay filter at {self.sgol_len} window and {self.sgol_order} polynomial on resampled daily'
        
        if outfile is None:
            outfile = os.path.join(self.outdir, os.path.basename(self.filename))
        self.ds_out.to_netcdf(outfile)

#%%
working_dir = '/data/yipeeo_wd'
org_spain_dir = os.path.join(working_dir, '07_data/Predictors/eo_ts/s2/Spain')
indata_dir = os.path.join(org_spain_dir, 'lleida_madrid')
outdata_dir = os.path.join(working_dir, 'Data/Predictors/eo_ts/s2/Spain/spain_gapfilled') #creted folder manually for clarity

var_list = ['ndvi', 'evi', 'nmdi', 'ndwi']

#define function to process file so that I can use parallel processing
def process_file(file):
    file_init = gap_fill(file, var_list, outdata_dir, sampling_rate = 10, sgol_order = 2, sgol_len = 9)
    file_init.process_ds()
    file_init.save_out()

#%%
infilelist = sorted(glob.glob(os.path.join(indata_dir, '*.nc')))

#%%
#Test the function
# file = infilelist[0]
# ds = xr.open_dataset(file)
# ds.close()
# print(ds)
# print('\n')
# process_file(file)
# fileout = os.path.join(outdata_dir, os.path.basename(file))
# if os.path.exists(fileout):
#     ds = xr.open_dataset(fileout)
#     ds.close()
#     print(ds)
# else:
#     print('File not found')
#%%
# check the data if needed, filelist can be replaced with output files
# for file in infilelist[2000:2010]:
#     with xr.open_dataset(file) as ds:
#         # print(ds)
#         ndvi = ds.ndvi.data
#         ndvi_sum = np.nansum(ndvi.ravel())
#         # print(f'{file} has ndvi sum as {ndvi_sum}')
#         # if np.isnan(ndvi_sum):
#         #     continue
#         evi = ds.evi.data
#         nmdi = ds.nmdi.data
#         ndwi = ds.ndwi.data
#         vis = [ndvi, evi, nmdi, ndwi]
#         fig = plt.figure()
#         ax = fig.add_subplot()
#         # for vi in vis:
#         #     ax.bar(np.arange(len(vi)), vi)
#         ax.bar(np.arange(len(evi)), evi, label='EVI')
#         ax.bar(np.arange(len(nmdi)), -nmdi, label='NMDI')
#         # ax.legend()
#         fig.savefig(f'/data/yipeeo_wd/Other/{os.path.basename(file)}.png')
#         plt.close(fig)

# %%
#Apply function in parallel processing for all the files
with Pool() as pool:
    list(tqdm(pool.imap(process_file, infilelist), total = len(infilelist)))
# %%
