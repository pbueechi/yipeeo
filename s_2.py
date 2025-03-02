import os.path
import gc
import pystac_client
import planetary_computer
import rasterio
import csv
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
import matplotlib.gridspec as gridspec
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as st
from pyproj import Proj, transform
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from rasterio.mask import mask
from pystac.extensions.eo import EOExtension as eo

# Data extraction from Sentinel-2 L2A data is done in the first function and files are stored as csv
def extract_s2(region, core=None, new=True):
    """
    :param region: str of region where fields are located which will be extract from S-2 L2A data. So far available: czr, nl, rom, ukr_chmel, ukr_horod, ukr_lviv
    :param year: int if only individual years should be considered
    :param new: binary True/False. Set True if new files should be generated. Attention this will delete potential older files there
    :return: dataframe of S2 observations per field
    """
    print('started calculating...')

    #Load fields and remove identical fields
    #ToDo adjust filepath to file including field polygons, and if file does not contain c_year, adjust year_min and max to 2016 and 2022
    fields = gpd.read_file(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale_{region}.shp')
    year_min, year_max = np.nanmin(fields.c_year.values),np.nanmax(fields.c_year.values)

    if year_max<2016:
        raise ValueError('no S-2 data available for the given fields at the time of harvesting')
    if year_min<2016:
        year_min=2016
        warnings.warn('There are harvest dates before 2016. No S-2 data is available by then. Hence, only S-2 data from 2016 onwards will be extracted')

    fields = fields.drop_duplicates(subset='field_id')

    #Define bounding box around fields where S2 data will be loaded from
    xmin, xmax = fields.bounds.min(0).minx, fields.bounds.max(0).maxx
    ymin, ymax = fields.bounds.min(0).miny, fields.bounds.max(0).maxy
    area_of_interest = {
        "type": "Polygon",
        "coordinates": [[
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]],
    }

    # Set considered time and bands that should be extracted. So far set to visual, vegetation red edge, NIR and SWIR
    time_of_interest = f'{year_min}-01-01/{year_max}-12-31'
    bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']

    # Setup connections to planetary computer and search for available S-2 images
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 80}},               #Only scenes with cloud_cover<80% are considered
    )

    items = search.item_collection()

    #Establish files where information will be stored
    row_head = ['observation', 'cloud_cover[%]'] + list(fields.field_id.values)

    if not os.path.exists(rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}'):
        os.makedirs(rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}')

    if new:
        for band in bands:
            if core:
                file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_{core}.csv'
            else:
                file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_all.csv'
            with open(file_path, 'w') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(row_head)

    # Start looping through all S-2 scenes and defined S-2 bands
    print(f"we found {len(items)} items for the given search query")
    for it_num,item in enumerate(items):
        print(it_num, item.datetime.date())
        signed_item = planetary_computer.sign(item)
        for band in bands:
            #Load S2 scenes
            src = rasterio.open(signed_item.assets[band].href)
            fields = fields.to_crs(src.crs)

            #Per scene and band extract and summarize the information per field
            fields_data = pd.DataFrame(data=None, index=range(2), columns=row_head)
            fields_data.iloc[:,0] = [f'{item.datetime.date()}_median', f'{item.datetime.date()}_std']
            fields_data.iloc[:, 1] = [eo.ext(item).cloud_cover, eo.ext(item).cloud_cover]

            if core:
                file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_{core}.csv'
            else:
                file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_all.csv'

            for ipol in range(len(fields)):
                polygon = fields[ipol:ipol+1]
                try:
                    out_image, out_transform = mask(src, polygon.geometry, crop=True)
                    out_image = np.where(out_image == 0, np.nan, out_image)
                    if np.isnan(out_image).all():
                        fields_data.loc[:,polygon.field_id.values[0]] = [np.nan, np.nan]
                    else:
                        if band=='SCL':
                            arr_flat = out_image.flatten()
                            arr_flat = arr_flat[~np.isnan(arr_flat)]
                            fields_data.loc[:,polygon.field_id.values[0]] = [int(st.mode(arr_flat)[0][0]), np.nanstd(out_image)]
                        else:
                            fields_data.loc[:,polygon.field_id.values[0]] = [np.nanmedian(out_image), np.nanstd(out_image)]
                except:
                    fields_data.loc[:, polygon.field_id.values[0]] = [np.nan, np.nan]
                    # warnings.warn('Input shapes do not overlap raster')

            with open(file_path, 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(list(fields_data.iloc[0,:]))
                writer.writerow(list(fields_data.iloc[1,:]))

            gc.collect()

# The generated csv files are converted to individual nc files per field
def table2nc(region):
    """
    :param region: str of region where fields are located which will be extract from S-2 L2A data. So far available: czr, nl, rom, ukr_chmel, ukr_horod, ukr_lviv
    :return:combines all csv files with the data of the individual S-2 L2A Bands to one netcdf file
    """

    #define all sentinal-2 L2A bands that should be considered
    bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']
    band = bands[0]

    #Load first file to get all fields names
    file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_all.csv'
    file = pd.read_csv(file_path, index_col=0)
    fields = file.columns[1:]

    #Filter fields with only nan values
    nan_fields = file.iloc[:,1:].isna().all(axis=0)
    fields = fields[~nan_fields]

    if not os.path.exists(rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\nc'):
        os.makedirs(rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\nc')

    if np.sum(nan_fields)>0:
        with open(rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\nc\read_me.txt','w') as file:
            file.write(f'There are fields with no pixels of S-2 L2A data inside. These are the fields:{fields[nan_fields].values}')

    #Loop through all fields to establish one nc field per field with all bands
    for f, field in enumerate(fields):
        #loop through all bands to collect information of all bands per field
        for b,band in enumerate(bands):
            file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_all.csv'
            file = pd.read_csv(file_path, index_col=0)
            file.iloc[:,1:] = file.iloc[:,1:].round()

            #Split file into median and std
            med_f = file.iloc[::2, :]
            std_f = file.iloc[1::2, :]
            #Remove rows with only nan
            nan_fields = med_f.iloc[:, 1:].isna().all(axis=1)
            med_f = med_f[~nan_fields.values]
            std_f = std_f[~nan_fields.values]

            #check if there are only std values in the std_f file
            char = [a.split('_')[1] for a in std_f.index]
            if not char[1:]==char[:-1]:
                raise ValueError('There are not only std values in the std file')

            #Establish a pandas Series as target variable with daily timestep from 2016 to 2022
            dates_all = pd.date_range(start='1/1/2016', end='31/12/2022')
            target_var = pd.Series(data=None, index=dates_all)

            #Establish pandas series with sentinel-2 data as loaded in the file and merge it with the target series
            dates = [datetime.strptime(a.split('_')[0], '%Y-%m-%d') for a in med_f.index]
            med_field = pd.Series(data=med_f.iloc[:, f+1].values, index=dates)
            std_field = pd.Series(data=std_f.iloc[:, f+1].values, index=dates)

            #resample to daily values for cases where there are several observations per day
            med_field = med_field.resample('D').mean()
            std_field = std_field.resample('D').mean()

            _, med_daily = target_var.align(med_field, axis=0)

            _, std_daily = target_var.align(std_field, axis=0)
            med_daily = med_daily.replace(np.nan, -9999)
            std_daily = std_daily.replace(np.nan, -9999)

            #Establish new xr Dataset in first loop. Afterwards add bands to this file.
            if b == 0:
                #Add cloud cover info
                cloud_field = pd.Series(data=med_f.iloc[:,0].values, index=dates)
                cloud_field = cloud_field.resample('D').mean()
                _, cloud_daily = target_var.align(cloud_field, axis=0)
                xr_file = xr.Dataset.from_dataframe(pd.DataFrame(cloud_daily, columns=[f'cloud_cover']))
                xr_file[f'cloud_cover'].attrs = dict(
                                        # FillValue=np.nan,
                                        units='%',
                                        long_name=f'mean cloud cover over whole S-2 scene, not per field',
                                        value_range='0-100'
                                    )

                xr_file[f'{band}_median'] = xr.DataArray.from_series(med_daily).astype('int32')
                xr_file[f'{band}_std'] = xr.DataArray.from_series(std_daily).astype('int32')

            else:
                if band == 'SCL':
                    xr_file[f'{band}_mode'] = xr.DataArray.from_series(med_daily).astype('int32')
                else:
                    xr_file[f'{band}_median'] = xr.DataArray.from_series(med_daily).astype('int32')
                    xr_file[f'{band}_std'] = xr.DataArray.from_series(std_daily).astype('int32')
            if band == 'SCL':
                xr_file['SCL_mode'].attrs = dict(
                    FillValue=-9999,
                    units='-',
                    long_name='Sentinel-2 Scene Classification',
                    scl_values=[0,1,2,3,4,5,6,7,8,9,10,11],
                    scl_classes=['No_data','Saturated_or_defective_pixel','Topographic_casted_shadows','Cloud_shadows',
                             'Vegetation','Non-vegetated','Water','Unclassified','Cloud_medium_probability',
                             'Cloud_high_probability','Thin_cirrus','Snow_or_ice']
                )
            else:
                xr_file[f'{band}_median'].attrs = dict(
                    FillValue=-9999,
                    units='-',
                    long_name=f'median of all Sentinel-2 {band} pixels laying within field',
                    value_range='1-10000'
                )
                xr_file[f'{band}_std'].attrs = dict(
                    FillValue=-9999,
                    units='-',
                    long_name=f'std of all Sentinel-2 {band} pixels laying within field',
                    value_range='1-10000'
                )
        xr_file = xr_file.rename({'index':'time'})
        # print(xr_file.time)
        # comp = dict(zlib=True, complevel=9)
        # encoding = {var: comp for var in xr_file.data_vars}
        xr_file.to_netcdf(path=rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\nc\{field}.nc')

# The nc files are cleaned (outlier removed and cloud masked) with the following function
def cleaning_s2(region):
    """
    :param region:
    :return:
    """
    path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\nc'
    fields = os.listdir(path)
    fields = [field for field in fields if field.endswith('.nc')]

    bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    bands = [band+'_median' for band in bands]

    #Load required data
    for field in fields:
        print(field)
        #Load file and extract scene classification
        ds = xr.open_dataset(os.path.join(path, field))
        scl = ds.SCL_mode.values
        if np.sum(scl==[-9999]*len(scl))==len(scl):
            print(f'file {field} has only nan values')
            continue

        # Sentinel-2 processing changed on 2022-01-25 introducing an offset of 1000
        # https://github.com/stactools-packages/sentinel2/issues/44

        time = ds.time.values
        offset_loc = np.where(time>=pd.to_datetime('2022-01-25', format='%Y-%m-%d'))[0]
        #Establish mask with valid observations. I.e. Scene classification without clouds, snow etc.
        #scl as in https://www.sciencedirect.com/science/article/pii/S0924271623002654
        ok_scene_class = [4, 5, 6]
        a_mask = np.in1d(scl, ok_scene_class)

        #Load all bands, mask them with scl mask and remove outliers
        for band in bands:
            band_values = ds[band].values
            #remove offset of 1000 since Jan 25 2022
            band_values[offset_loc] = band_values[offset_loc]-1000
            band_values_masked = band_values[a_mask]

            #remove outliers
            band_values_masked_no_ol = removeOutliers(band_values_masked)
            ds[band].values = [-9999]*len(ds[band].values)
            ds[band].values[a_mask] = band_values_masked_no_ol

        path_out = os.path.join(path, 'cleaned')
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        ds.to_netcdf(os.path.join(path_out, field))

# As a last step, Vegetation indices are added to the nc files.
def add_indices2nc(file_path):
    files = os.listdir((file_path))
    files = [file for file in files if file.endswith('.nc')]
    for file in files:
        ds = xr.open_dataset(os.path.join(file_path, file))

        B2 = ds.B02_median
        B4 = ds.B04_median
        B8 = ds.B08_median
        B11 = ds.B11_median
        B12 = ds.B12_median
        ndvi,evi,ndwi,nmdi = indices_calc(B2, B4, B8, B11, B12)

        ds['ndvi'] = ndvi
        ds['ndvi'].attrs = dict(
            # FillValue=np.nan,
            units='-',
            long_name='Normalized Difference Vegetation Index',
            value_range='-1 to 1'
        )

        ds['evi'] = evi
        ds['evi'].attrs = dict(
            # FillValue=np.nan,
            units='-',
            long_name='Enhanced Vegetation Index',
            value_range='-1 to 1'
        )

        ds['ndwi'] = ndwi
        ds['ndwi'].attrs = dict(
            # FillValue=np.nan,
            units='-',
            long_name='Normalized Difference Water Index',
            value_range='-1 to 1'
        )

        ds['nmdi'] = nmdi
        ds['nmdi'].attrs = dict(
            # FillValue=np.nan,
            units='-',
            long_name='Normalized Multiband Drought Index',
            value_range='0 to ~1'
        )

        if not os.path.exists(os.path.join(file_path,'new')):
            os.makedirs(os.path.join(file_path,'new'))
        ds.to_netcdf(path=os.path.join(file_path,'new', file))

# ----------------- The upcoming functions are not required to be run individually -------------------
# The next two functions are used in add_indeces2nc and cleaning_s2
def indices_calc(B2,B4,B8,B11,B12):
    """
    :param B2-B12: np.array of the required S-2 bands to calculate the indices.
    :return: Calculates the indices ndvi, evi, ndwi (water index), and nmdi (normalized multiband drought index)
    based on common formulas as for example in cavalaris et al., 2021
    """
    ts = B2.time.values

    B2 = np.where(B2<=-9999, np.nan, B2/10000)
    B4 = np.where(B4<=-9999, np.nan, B4/10000)
    B8 = np.where(B8<=-9999, np.nan, B8/10000)
    B11 = np.where(B11<=-9999, np.nan, B11/10000)
    B12 = np.where(B12<=-9999, np.nan, B12/10000)

    ndvi = (B8-B4)/(B8+B4)
    evi = 2.5*(B8-B4)/((B8+6*B4-7.5*B2)+1)
    ndwi = (B8-B12)/(B8+B12)
    nmdi = (B8-(B11-B12))/(B8+(B11-B12))

    ndvi_ar = xr.DataArray.from_series(pd.Series(data=ndvi, index=ts))
    evi_ar = xr.DataArray.from_series(pd.Series(data=evi, index=ts))
    ndwi_ar = xr.DataArray.from_series(pd.Series(data=ndwi, index=ts))
    nmdi_ar = xr.DataArray.from_series(pd.Series(data=nmdi, index=ts))

    ndvi_ar = ndvi_ar.rename({'index': 'time'})
    evi_ar = evi_ar.rename({'index': 'time'})
    ndwi_ar = ndwi_ar.rename({'index': 'time'})
    nmdi_ar = nmdi_ar.rename({'index': 'time'})

    return ndvi_ar,evi_ar,ndwi_ar,nmdi_ar

def removeOutliers(x, outlierConstant=2):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    a = np.where(((a>=quartileSet[0]) & (a<=quartileSet[1])),a,np.nan)
    return a

# The next and final 2 functions plot examples of the sentinel-2 data and are not required for the data extraction
def plot_s2_class(region, year=None):
    """
    :param region: str of region where fields are located which will be extract from S-2 L2A data. So far available: czr, nl, rom, ukr_chmel, ukr_horod, ukr_lviv
    :return: dataframe of S2 observations per field
    """
    print('started calculating...')

    # Load fields and remove identical fields
    fields = gpd.read_file(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale_{region}.shp')
    year_min, year_max = np.nanmin(fields.c_year.values), np.nanmax(fields.c_year.values)

    if year_max < 2016:
        raise ValueError('no S-2 data available for the given fields at the time of harvesting')
    if year_min < 2016:
        year_min = 2016
        warnings.warn(
            'There are harvest dates before 2016. No S-2 data is available by then. Hence, only S-2 data from 2016 onwards will be extracted')

    fields = fields.drop_duplicates(subset='field_id')

    # Define bounding box around fields where S2 data will be loaded from
    xmin, xmax = fields.bounds.min(0).minx, fields.bounds.max(0).maxx
    ymin, ymax = fields.bounds.min(0).miny, fields.bounds.max(0).maxy
    area_of_interest = {
        "type": "Polygon",
        "coordinates": [[
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]],
    }

    # Set considered time and bands that should be extracted. So far set to visual, vegetation red edge, NIR and SWIR
    if year:
        time_of_interest = f'{year}-01-01/{year}-12-31'
    else:
        time_of_interest = f'{year_min}-01-01/{year_max}-12-31'
    bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

    # Setup connections to planetary computer and search for available S-2 images
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 25}},  # Only scenes with cloud cover < 25% ar considered
    )

    items = search.item_collection()

    # Establish files where information will be stored
    row_head = ['observation', 'cloud_cover[%]'] + list(fields.field_id.values)

    most_cloudy_item = max(items, key=lambda item: eo.ext(item).cloud_cover)

    fig = plt.figure(figsize=(20, 8))
    outer = gridspec.GridSpec(1, 2, width_ratios=[0.5, 0.5])
    ax1 = plt.Subplot(fig, outer[0])

    asset_href = most_cloudy_item.assets["visual"].href

    #
    ds = rasterio.open(asset_href)
    band_data = ds.read()

    #Reproject to wgs84 lat lon
    inProj = Proj(ds.crs)
    outProj = Proj('epsg:4326')
    x1, y1, x2, y2 = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
    y1, x1 = transform(inProj, outProj, x1, y1)
    y2, x2 = transform(inProj, outProj, x2, y2)

    #Transpose to image and resample to 800 pixels
    img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
    w = img.size[0]
    h = img.size[1]
    aspect = w / h
    target_w = 800
    target_h = (int)(target_w / aspect)
    a = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

    ticks_size=20
    #Plot optical data
    ax1.imshow(a)
    ax1.set_xticks(np.linspace(0, a.size[0], num=5), np.round(np.linspace(x1, x2, num=5), 1), fontsize=ticks_size)
    ax1.set_yticks(np.linspace(a.size[1], 0, num=5), np.round(np.linspace(y1, y2, num=5), 1), fontsize=ticks_size)
    ax1.set_ylabel('Lat [°]', fontsize=ticks_size)
    ax1.set_xlabel('Lon [°]', fontsize=ticks_size)
    fig.add_subplot(ax1)

    #Load Scene Classification and set plot axis
    ax1 = plt.Subplot(fig, outer[1])
    asset_href = most_cloudy_item.assets["SCL"].href
    ds = rasterio.open(asset_href)

    band_data = ds.read()
    scl_classes = {0:'No data', 1:'Saturated/defective',2:'Topographic shadows',3:'Cloud shadows',4:'Vegetation',5:'Non-vegetated',
                   6:'Water',7:'Unclassified',8:'Cloud medium prob',9:'Cloud high prob',10:'Thin cirrus',11:'Snow/ice'}
    values = np.unique(band_data)
    img = Image.fromarray(band_data[0,:,:])
    im=ax1.imshow(img, cmap='jet')
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=scl_classes[values[i]])) for i in range(len(values))]

    # Plot scene classification
    ax1.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=ticks_size)
    ax1.set_xticks(np.linspace(0,ds.width,num=5), np.round(np.linspace(x1,x2, num=5),1), fontsize=ticks_size)
    ax1.set_yticks([])
    ax1.set_xlabel('Lon [°]', fontsize=ticks_size)
    fig.add_subplot(ax1)
    fig.subplots_adjust(left=0.06, right=0.77, bottom=0.07, top=0.99)
    # fig.tight_layout()
    plt.savefig('Figures/scene_class_25.png', dpi=300)

def plot_s2_ex(region):
    """
    :return: Plots the data of Sentinel-2 band 2 for checking the impact of scene classification masking
    """
    path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\nc'
    fields = os.listdir(path)
    fields = [field for field in fields if field.endswith('.nc')]
    print(fields)
    #Load required data
    ds = xr.open_dataset(os.path.join(path, fields[5]))
    cc = ds.cloud_cover.values
    # ['B02', 'B04', 'B08', 'B11', 'B12', 'SCL']
    b2 = ds.B12_median.values
    scl = ds.SCL_mode.values
    time = ds.time.values

    time = time[b2>-9999]
    cc = cc[b2>-9999]
    scl = scl[b2>-9999]
    b2 = b2[b2>-9999]

    a = pd.DataFrame(columns=['b2', 'cc', 'scl'], index=time)
    a.loc[:, 'b2'] = b2
    a.loc[:, 'cc'] = cc
    a.loc[:, 'scl'] = scl
    # print(a.head(50))

    ok_scene_class = [4, 5, 6]  # scl as in https://www.sciencedirect.com/science/article/pii/S0924271623002654
    a_mask = np.in1d(a.scl, ok_scene_class)
    a_masked = a.iloc[a_mask, :]

    #Sentinel-2 processing changed on 2022-01-25 introducing an offset of 1000
    #https://github.com/stactools-packages/sentinel2/issues/44
    offset_loc = np.where(a_masked.index>='2022-01-25')[0]
    a_masked.iloc[offset_loc,0] = a_masked.iloc[offset_loc,0]-1000

    #plotting
    plt.plot(a.index, a.b2/10000)
    plt.plot(a_masked.index, a_masked.b2/10000)
    plt.plot(a_masked.index, removeOutliers(a_masked.b2)/10000)
    plt.legend(['all_s2_data','2022 adjusted','clean data'])
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    # plt.savefig('Figures/cloud_masking_impact_clean_2016_5czr.png', dpi=300)


if __name__ == '__main__':
    # pd.set_option('display.max_rows', None)
    warnings.filterwarnings('ignore')
    start_pro = datetime.now()
    # print('started calculating...')
    # plot_s2_class(region='czr')
    # extract_s2(region='polk', new=False)
    # cleaning_s2(region='czr')
    plot_s2_ex(region='czr')

    # for region in ['polk']:
    #     print(region)
    #     extract_s2(region=region, new=False)
    #     table2nc(region=region)
    #     cleaning_s2(region=region)
    #     add_indices2nc(file_path=rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\nc\cleaned')
    # ecostress()
    print(f'calculation stopped and took {datetime.now() - start_pro}')
    # parallel_run(path=field_shape_path)