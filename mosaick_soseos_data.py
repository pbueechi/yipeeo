#%%
'''
# Author: Nirajan Luintel
# Date: 2024-06-07
# Created with: Visual Studio Code
# Purpose: To mosaic the tiles of VPP data
Mamba Environment: climers
'''
#%%
'''
Checked manually the date extracted and found it did not give the kind of result I expected
For example madrid ids 8681 to 8689 have winter barley in 2022 which is to be planted within nov-dec 2021
but the plantation date for both seasons and also for 2021 second season is nowhere near that time frame 
'''
#import libraries
import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
#%%
fields = ['madrid', 'lleida']
field = fields[0]
#set working directory so that data is downloaded where it should be
wd = f'/data/yipeeo_wd/Data/VPP_Wekeo/{field}'
os.chdir(wd)
#%%
#list the variables to be separate files
variables = ['SOSD', 'EOSD', 'QFLAG']
yaers = list(range(2017, 2024))
fillval = 32765
seasons = [1,2]
filelist_downloaded = glob.glob('*.tif') #in wd only
yr = 2020; s = 1; var = 'SOSD'
tile_files = glob.glob(f'VPP_{yr}_S2_T*-010m_V101_s{s}_{var}.tif')

# %%
# for var in variables:
#     for yr in tqdm(yaers):
#         for s in seasons:
#             # List all the tiles
#             tile_files = glob.glob(f'VPP_{yr}_S2_T*-010m_V10*_s{s}_{var}.tif')

#             # Open the tiles
#             src_files_to_mosaic = []
#             for fp in tile_files:
#                 src = rasterio.open(fp)
#                 src_files_to_mosaic.append(src)

#             # Merge the tiles
#             mosaic, out_trans = merge(src_files_to_mosaic)

#             # Update metadata
#             out_meta = src_files_to_mosaic[0].meta.copy()
#             out_meta.update({
#                 "driver": "GTiff",
#                 "height": mosaic.shape[1],
#                 "width": mosaic.shape[2],
#                 "transform": out_trans
#             })

#             # Save the mosaic
#             with rasterio.open(f'VPP_{yr}_S2-010m_V101_s{s}_{var}_mosaic.tif', 'w', **out_meta) as dest:
#                 dest.write(mosaic)

#             # # Optional: Display the mosaic
#             # show(mosaic, cmap='terrain')

# %%
#Define a function to convert yyddd format to days since 2000
#This function is not 100% correct as it does not consider the leap year
#But it is sufficient to do the zonal stats with DOY values
#Will reverse the process to get the exact date after running zonal stat
def yyddd_to_days_since_2000(yyddd_array, reference_date):#, filldate = 32765):
    # if reference date is not in pandas convert it
    reference_year = pd.Timestamp(reference_date).year #both string and datetime object is valid

    # yyddd_array[np.isnan(yyddd_array)] = filldate
    # Separate year and Julian day
    years, days_of_year = np.divmod(yyddd_array, 1000)
    #convert yy format to yyyy format by adding 2000 or 1900 conditionally
    full_years = np.where(years < 50, 2000 + years, 1900 + years)
    
    # Calculate the difference in days from the reference date
    days_since_2000 = (full_years - reference_year) * 365 + days_of_year
    
    return days_since_2000

#Now use the QFLAG to signal less quality data as nans
#At the same time convert yyddd to full days since 2000
def mask_with_qc(flag_file, data_file, flag_thres, fillval, ref_date):
        
    # Open the quality flag file
    with rasterio.open(flag_file) as flag_src:
        flags = flag_src.read(1)  # Read the first band

    # Open the first dataset file
    with rasterio.open(data_file) as ds1_src:
        ds1 = ds1_src.read(1)  # Read the first band
        ds1_meta = ds1_src.meta

    # Create a mask for low quality pixels (flag values < 6)
    low_quality_mask = flags < flag_thres

    # Apply the mask to the datasets, setting low quality pixels to NaN
    ds1 = np.float16(ds1)
    ds1[low_quality_mask] = np.nan

    days_since_2000 = yyddd_to_days_since_2000(ds1, ref_date)
    days_since_2000[np.isnan(days_since_2000)] = fillval #put back the fill value
    days_since_2000 = days_since_2000.astype(np.int16)
    # Save the filtered dataset files
    ds1_meta.update(dtype=rasterio.int32, nodata = fillval)  # Update metadata to match the new data type if necessary

    with rasterio.open(f'{data_file[:-4]}_flagged_converted.tif', 'w', **ds1_meta) as dst1:
        dst1.write(days_since_2000, 1)

# %%
# flag_thres = 7
# fillval = 32765
# for yr in tqdm(yaers):
#     for s in seasons:
#         # List all the tiles
#         flag_file = glob.glob(f'VPP_{yr}_S2-010m*_s{s}_QFLAG_mosaic.tif')[0]
#         sos_file = glob.glob(f'VPP_{yr}_S2-010m*_s{s}_SOSD_mosaic.tif')[0]
#         eos_file = glob.glob(f'VPP_{yr}_S2-010m*_s{s}_EOSD_mosaic.tif')[0]
#         mask_with_qc(flag_file, sos_file, flag_thres, fillval, '2000-01-01')
#         mask_with_qc(flag_file, eos_file, flag_thres, fillval, '2000-01-01')
#         print(flag_file)

#%%
#Account for year not as numbers for zonal stat calculation
# Function to convert days since reference date back to datetime
# def days_since_reference_to_datetime(days_since_2000, ref_date): #input will be a pandas series
#     #extract year
#     ref_year = [pd.Timestamp(ref_date).year]*len(days_since_2000)
#     #reverse the days since 
#     year, doy = np.divmod(days_since_2000 + ref_year * 365, 365)
#     #convert to yyyyddd
#     yyyyddd = year * 1000 + doy

#     # if np.isnan(yyyyddd):
#     #     return np.nan
    
#     yyyyddd[np.isnan(yyyyddd)] = 2099001
#     #round up the doy, use ceil to avoid 0th day of year
#     yyyyddd = np.int32(np.ceil(yyyyddd))
#     yyyyddd = yyyyddd.astype(str)

#     yyyyddd_pd = pd.Series(yyyyddd)
#     yyyyddd_pd_dt = pd.to_datetime(yyyyddd_pd, format = '%Y%j')
#     yyyyddd_pd_dt[yyyyddd_pd_dt > '2025-01-01'] = np.nan

#     return yyyyddd_pd_dt.to_numpy()
#%%
def days_to_datestr(days_since_2000, ref_date = '2000-01-01'): #input will be a pandas series
    #extract year
    if days_since_2000 is None:
        date_str = None
        return date_str
    
    ref_year = pd.Timestamp(ref_date).year
    #reverse the days since 
    year, doy = np.divmod(days_since_2000 + ref_year * 365, 365)
    if doy <1:
        doy = 1 #even when I used ceil, it had some 0s 
    #convert to yyyyddd
    yyyyddd = year * 1000 + doy

    # if np.isnan(yyyyddd):
    #     return np.nan
    
    # yyyyddd[np.isnan(yyyyddd)] = 2099001
    #round up the doy, use ceil to avoid 0th day of year
    yyyyddd = np.int32(np.ceil(yyyyddd))
    yyyyddd = yyyyddd.astype(str)
    date_dt = datetime.strptime(yyyyddd, '%Y%j')
    date_str = datetime.strftime(date_dt, '%Y-%m-%d')

    return date_str


#%%
# # Example usage
# yyddd_array = np.array([[17360, np.nan], [17250, 18120]])

# days_since_2000 = yyddd_to_days_since_2000(yyddd_array, '2000-01-01')
# print(days_since_2000)

#%%
# days_since_2000_pd = pd.DataFrame(days_since_2000.flatten(), columns=['output'])
# dt = days_since_reference_to_datetime(days_since_2000_pd['output'], '2000-01-01')

#%%
shapefile = f'/data/yipeeo_wd/07_data/Crop yield/Database/field_scale_{field}_epsg32630.shp'
shpgpd = gpd.read_file(shapefile)
#%%
for yr in tqdm(yaers[4:]):
    for var in variables[:2]: #not QFLAG
        for s in seasons:
                        
            filelist = glob.glob(os.path.join(wd, f'VPP_{yr}_S2-*_s{s}_{var}_mosaic_flagged_converted.tif'))
            data_file = filelist[0]
            stat_mean = zonal_stats(shapefile, data_file, stats = ['mean'], all_touched = True, nodata = fillval)
            
            stat_mean_val = [days_to_datestr(i['mean']) for i in stat_mean]
            # stat_date = days_to_datestr(stat_mean_val, '2000')
            
            shpgpd[f'{var[:1]}_{yr}_s{s}'] = stat_mean_val
# %%
basename = os.path.basename(shapefile)
outfile = os.path.join(wd, f'{basename[:-4]}_soseos.shp')
shpgpd.to_file(outfile)
# %%
