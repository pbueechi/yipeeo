#%%
'''
# Author: Nirajan Luintel
# Date: 2024-06-03
# Created with: Visual Studio Code
# Purpose: extract start of season and end of season from timeseries to use as input for yield forecasting
Mamba Environment: yipeeo
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
import geopandas as gpd
import time as tyme
#%% set up working environment
#set working directory
wd = '/data/yipeeo_wd'
os.chdir(wd)

#get parent directory for all the work
parent_directory = '/data/yipeeo_wd'
indata_parent_dir = os.path.join(parent_directory, 'Data')
s2_folder = os.path.join('Predictors', 'eo_ts', 's2', 'Spain')
s2_spain_dir = os.path.join(indata_parent_dir, s2_folder)
s1_folder = os.path.join('Predictors', 'eo_ts', 's1', 'daily')
s1_dir = os.path.join(indata_parent_dir, s1_folder)
fields = ['madrid', 'lleida']

out_parent_dir = os.path.join(parent_directory, 'Data')
sos_eos_dir = os.path.join(out_parent_dir, 'sos_eos')
os.makedirs(sos_eos_dir, exist_ok=True)


#%%
# #read the field data file and get the crop names
# for loc in fields:
#     yield_data_file = os.path.join(parent_directory, 'Data','Crop yield', 'Database', f'field_scale_{loc}.shp')
#     crop_data = gpd.read_file(yield_data_file)
#     crop_group = crop_data.groupby('crop_type')
#     crops = crop_group.size().sort_values(ascending=False)
#     crop_file = os.path.join(sos_eos_dir, f'crops_{loc}.csv')
#     crop_df = pd.DataFrame(crops, columns=['counts'])
#     crop_df.to_csv(crop_file, index=True)

'''
This gives only the names of crops in each datafile
Then manually find the range of sos and eos from external sources
For this case I got these values from files in /data/yipeeo_wd/Other
If more authentic files are available then use that
For other countries no need to do this because they already have data
'''
#%%

evi_min_thres = 0.3
evi_max_thres = 0.35
window = 45 #in days

# for loc in fields:
loc = fields[0]
s2_field_dir = os.path.join(s2_spain_dir, loc)
filelist = glob.glob(os.path.join(s2_field_dir, 'nc', '*.nc'))
sos_eos_file = os.path.join(sos_eos_dir, f'crops_{loc}_soseos.csv')
sos_eos_df = pd.read_csv(sos_eos_file, index_col=0)
# %%
# file = filelist[0]

# ds=xr.open_dataset(file)
# ds.close()
# ds.info()
# %%
yield_data_file = os.path.join(parent_directory, '07_data','Crop yield', 'Database', f'field_scale_{loc}.shp')
crop_data = gpd.read_file(yield_data_file)

#%%
for row in tqdm(range(len(crop_data))): #[1]:#range(5):#
    #tyme.sleep(1)
    field_data = crop_data.iloc[row,:]
    yr = field_data['c_year']
    cp = field_data['crop_type']
    ##%%
    if cp in sos_eos_df.index[:3]:
        #print('major crop')

        sos_start = sos_eos_df.loc[cp, 'sos_start']
        sos_end = sos_eos_df.loc[cp, 'sos_end']
        eos_start = sos_eos_df.loc[cp, 'eos_start']
        eos_end = sos_eos_df.loc[cp, 'eos_end']
        ##%%
        expand_time = {'sos1': int(round(0.6 * window)),
                    'sos2': window,
                    'eos1': int(round(0.6 * window)),
                    'eos2': int(round(0.5 * window)),
                    'crop_cycle': int(round(2*window))
                    } 
        sos1 = pd.Timestamp(f'{int(yr)}-{sos_start}') - pd.Timedelta(days = expand_time['sos1'])
        sos2 = pd.Timestamp(f'{int(yr)}-{sos_end}') + pd.Timedelta(days = expand_time['sos2'])
        eos1 = pd.Timestamp(f'{int(yr)}-{eos_start}') - pd.Timedelta(days = expand_time['eos1'])
        eos2 = pd.Timestamp(f'{int(yr)}-{eos_end}') + pd.Timedelta(days = expand_time['eos2'])

        #make the correction to sos to be in the previous year for autumn or winter planted crops
        if sos2 > eos1:
            sos1 = pd.Timestamp(f'{int(yr) - 1}-{sos_start}') - pd.Timedelta(days = expand_time['sos1'])
            sos2 = pd.Timestamp(f'{int(yr) - 1}-{sos_end}') + pd.Timedelta(days = expand_time['sos2'])
        
        crop_cycle_expand = pd.Timestamp(f'{int(yr)}-{sos_end}') + pd.Timedelta(days = expand_time['crop_cycle'])
        
        if sos2 < pd.Timestamp('2016-01-01'): #or eos1 > pd.Timestamp('2022-12-31'):
            continue

        if sos1 < pd.Timestamp('2016-01-01'):
            sos1 = pd.Timestamp('2016-01-01')
            if (sos2 - sos1).days < 30:
                sos2 = sos1 + pd.Timedelta(days = 30)
        
        # if eos2 > pd.Timestamp('2022-12-31'):
        #     eos2 = eos1
        
        # crop_data.loc[row, 'sowing_dat'] = pd.Timestamp(f'{int(yr)-1}-{sos_start}')

        ##%%
        file = os.path.join(s2_field_dir, 'nc', '{}.nc'.format(field_data['field_id']))
        ds=xr.open_dataset(file)
        ds.close()
        # ds.info()
        ## %%
        sos_sample = ds.sel(time = slice(sos1, sos2))
        eos_sample = ds.sel(time = slice(eos1, eos2))

        #plt.plot(sos_sample.time, sos_sample.evi)

        # time_numeric = sos_sample.time.values.astype('datetime64[D]').astype(int)
        evi_values = sos_sample.evi.values
        evi_values_cycle = ds.sel(time = slice(sos1, crop_cycle_expand)).evi.values

        # isnan = np.isnan(evi_values)
        # print('nans:', isnan)
        # print('max evi', evi_values.max())
        # print('min evi', evi_values.min())

        if evi_values.min() > evi_min_thres or evi_values_cycle.max() < evi_max_thres:
            continue
        # coefficients = np.polyfit(time_numeric, evi_values, 3)
        # evi_fitted = np.polyval(coefficients, time_numeric)
        # plt.plot(sos_sample.time, evi_fitted)

        evi_diff = np.diff(evi_values)
        evi_inc = (evi_diff>0).astype(int)
        evi_inc [evi_inc == 0] = -1 #for signalling decrease
        evi_change = np.diff(evi_inc)
        evi_minimas = np.where(evi_change == 2)[0] + 2
        evi_low = np.where(evi_values < evi_min_thres)[0]

        evi_minimas = [i for i in evi_minimas if i in evi_low]
        if len(evi_minimas)<1:
            continue

        evi_minima = evi_minimas[-1]
        # evi_minima = evi_minima[evi_values]
        
        #plt.plot(sos_sample.time[evi_minima], 0, marker = '*')
        
        sos_date = np.datetime_as_string(sos_sample.time[evi_minima].values, unit = 'D')
        crop_data.loc[row, 'sowing_dat'] = sos_date
    else:
        continue
        #print('not proceed with it')
#%%
basename = os.path.basename(yield_data_file)
crop_data_file = os.path.join(sos_eos_dir, f'{basename[:-4]}_sos.shp')
crop_data.to_file(crop_data_file, driver='ESRI Shapefile')
# %%
