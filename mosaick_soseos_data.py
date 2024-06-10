#%%
'''
# Author: Nirajan Luintel
# Date: 2024-06-07
# Created with: Visual Studio Code
# Purpose: To mosaic the tiles of VPP data
Mamba Environment: climers
'''
#%%
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


#%%
#set working directory so that data is downloaded where it should be
wd = '/data/yipeeo_wd/Data/VPP_Wekeo'
os.chdir(wd)
#%%
#list the variables to be separate files
variables = ['SOSD', 'EOSD', 'QFLAG']
yaers = list(range(2017, 2024))
seasons = [1,2]
filelist_downloaded = glob.glob('*.tif') #in wd only
yr = 2020; s = 1; var = 'QFLAG'
tile_files = glob.glob(f'VPP_{yr}_S2_T*-010m_V101_s{s}_{var}.tif')

#%%
# for var in variables:
#     for yr in yaers:
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
#Now use the QFLAG to signal less quality data as nans
def mask_with_qc(flag_file, data_file, flag_thres):
        
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

    # Save the filtered dataset files
    ds1_meta.update(dtype=rasterio.float32)  # Update metadata to match the new data type if necessary

    with rasterio.open(f'{data_file[:-4]}_flagged.tif', 'w', **ds1_meta) as dst1:
        dst1.write(ds1, 1)

# %%
# flag_thres = 7
# for yr in yaers:
#     for s in seasons:
#         # List all the tiles
#         flag_file = glob.glob(f'VPP_{yr}_S2-010m*_s{s}_QFLAG_mosaic.tif')[0]
#         sos_file = glob.glob(f'VPP_{yr}_S2-010m*_s{s}_SOSD_mosaic.tif')[0]
#         eos_file = glob.glob(f'VPP_{yr}_S2-010m*_s{s}_EOSD_mosaic.tif')[0]
#         mask_with_qc(flag_file, sos_file, flag_thres)
#         mask_with_qc(flag_file, eos_file, flag_thres)
#         print(flag_file)

# %%
#Account for year not as numbers for zonal stat calculation


def yyddd_to_days_since_2000(yyddd_array, reference_date, filldate = 32765):
    # Define the reference date
    reference_date = np.datetime64(reference_date)
    yyddd_array[np.isnan(yyddd_array)] = filldate
    # Separate year and Julian day
    years, days_of_year = np.divmod(yyddd_array, 1000)
    full_years = np.where(years < 50, 2000 + years, 1900 + years)
    
    # Create base dates for each year
    base_dates = np.array(['{}-01-01'.format(year) for year in full_years], dtype='datetime64[D]')
    
    # Add the Julian day to the base dates
    dates = base_dates + np.array(days_of_year - 1, dtype='timedelta64[D]')
    
    # Calculate the difference in days from the reference date
    days_since_2000 = (dates - reference_date).astype('timedelta64[D]').astype(int)
    
    return days_since_2000
#%%
# Example usage
yyddd_array = np.array([[17360, np.nan], [17250, 18120]])

days_since_2000 = yyddd_to_days_since_2000(yyddd_array, '2000-01-01')
print(days_since_2000)
#%%

# Function to convert raster data with NaN handling
def convert_yyddd_raster(input_file, output_file, ref_date, nan_placeholder):
    with rasterio.open(input_file) as src:
        yyddd_data = src.read(1)
        profile = src.profile

        # Convert yyddd to days since reference date with NaN handling
        days_data = np.vectorize(yyddd_to_days_since_reference)(yyddd_data, ref_date, nan_placeholder)

        # Update profile for output
        profile.update(dtype=rasterio.int32)  # Adjust dtype as necessary

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(days_data.astype(np.int32), 1)

#%%
# Function to convert days since reference date back to datetime
def days_since_reference_to_datetime(days, ref_date, nan_placeholder):
    if days == nan_placeholder:
        return np.nan
    return ref_date + timedelta(days=days)

# Calculate zonal statistics and handle different years
def zonal_stats_by_year(data_file, aoi, year, ref_date, nan_placeholder):
    with rasterio.open(data_file) as src:
        data = src.read(1)
        # Convert days since reference date back to datetime
        datetime_data = np.vectorize(days_since_reference_to_datetime)(data, ref_date, nan_placeholder)

        # Filter data for the specific year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        year_data = np.where((datetime_data >= start_date) & (datetime_data < end_date), data, nan_placeholder)

        stats = zonal_stats(aoi, year_data, stats=['mean'], nodata=nan_placeholder)
    return stats
#%%
# Convert your data files
# Define a reference date
reference_date = datetime(2000, 1, 1)
nan_placeholder = -999
convert_yyddd_raster(sos_file, 'delete_this.tif', reference_date, nan_placeholder)
# convert_yyddd_raster('path/to/your_data_file_2.tif', 'path/to/converted_data_file_2.tif', reference_date)
#%%

# Example usage for 2017 and 2018
stats_2017 = zonal_stats_by_year('path/to/converted_data_file_1.tif', aoi, 2017, reference_date)
stats_2018 = zonal_stats_by_year('path/to/converted_data_file_2.tif', aoi, 2018, reference_date)

print('Zonal stats for 2017:', stats_2017)
print('Zonal stats for 2018:', stats_2018)

# Load your area of interest (AOI) shapefile
aoi = gpd.read_file('path/to/your_aoi.shp')