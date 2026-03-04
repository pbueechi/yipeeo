import os
import gc
import rasterio
import warnings
import csv
import planetary_computer
import pystac
import pystac_client
import time
import glob
import itertools
import datetime
import traceback
import earthaccess
import rioxarray as rio
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
from shapely.geometry import Polygon
from pytesmo.time_series import anomaly
from scipy import signal
from rasterstats import zonal_stats
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from rasterio.mask import mask
from pystac.extensions.eo import EOExtension as eo
from scipy import stats as st


class coarse_scale_data:
    """
    Class to retrieve coarse scale data to NUTS level
    """

    def __init__(self, region, data):
        """
        :param region: str of name of location. Eg. for NUTS4 Austria data: Austria
        :param data: str with name of dataset. Either ERA5-Land or CGLS
        """
        self.region = region
        self.data = data
        self.table_path = f'D:/data-write/YIPEEO/predictors/{data}'
        if data=='VODCA':
            self.basepath = 'D:\DATA\yipeeo\Predictors\VODCA'
            self.parameters = ['VODCA_CXKu']
        elif data=='CCI_SM':
            self.basepath = 'D:\DATA\yipeeo\Predictors\CCI_SM'
            self.parameters = ['sm']
        elif data=='ERA5-Land':
            self.basepath = 'M:/Datapool/ECMWF_reanalysis/01_raw/ERA5-Land/datasets/images'
            self.parameters = ['t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr']
        else:
            raise ValueError('data not available. Please choose from VODCA, CCI_SM, ERA5-Land')

        if region=='EU':
            self.nuts_file_path = 'D:/DATA/yipeeo/SC2/Crop yield/All_NUTS.shp'
            ID = 'NUTS_ID'
        elif region=='Czechia':
            # self.nuts_file_path = r'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional\maize_nuts3.shp'
            self.nuts_file_path = r'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional\cz_nuts4.shp'
            ID = 'nut_id'
        else:
            self.nuts_file_path = f'D:/DATA/yipeeo/Crop_data/Crop_yield/{self.region}/maize.shp'
            ID = 'g_id'
        if os.name=='posix':
            self.syspath = '/home/pbueechi/thinclient_drives/'
            self.table_path=os.path.join(self.syspath, self.table_path)

            self.basepath = os.path.join(self.syspath, self.basepath)
            self.nuts_file_path = os.path.join(self.syspath, self.nuts_file_path)

        if not os.path.exists((self.table_path)): os.makedirs(self.table_path)
        self.csv_path = os.path.join(self.table_path, region)
        if not os.path.exists(self.csv_path): os.makedirs(self.csv_path)


        self.fields = gpd.read_file(self.nuts_file_path)
        self.fields = self.fields.set_crs(epsg=4326)
        self.fields = self.fields.drop_duplicates(subset=ID)
        print(self.fields.columns)
        self.row_head = ['date'] + list(self.fields[ID].values)

    def extract_data(self, parameter):
        """
        :return: writes csv files with the s2 information per parameter
        """
        file_path = os.path.join(self.basepath, parameter)
        files = os.listdir(file_path)
        files = [f for f in files if f.endswith('.tif')][::5]
        dates = [a.split('_')[-1].split('.')[0] for a in files]

        #Convert field to same crs as ERA5-Land data
        ref_crs = rasterio.open(os.path.join(file_path, files[0]))
        self.fields = self.fields.to_crs(ref_crs.crs)

        #Establish file where to store values
        fields_data = pd.DataFrame(data=None, index=range(len(dates)), columns=self.row_head)
        fields_data.iloc[:, 0] = dates

        for f, file in enumerate(files[:5]):
            print(f'done: {file.split("_")[-1]} at {datetime.datetime.now()} - {parameter}')
            src = rasterio.open(os.path.join(file_path, file))

            for ipol in range(len(self.fields)):
                polygon = self.fields[ipol:ipol + 1]
                out_image, out_transform = mask(src, polygon.geometry, crop=True)
                out_image = np.where(out_image == 0, np.nan, out_image)
                if np.isnan(out_image).all():
                    fields_data.iloc[f, ipol+1] = np.nan
                else:
                    fields_data.iloc[f, ipol+1] = np.nanmedian(out_image)
            src.close()
            gc.collect()
        fields_data.to_feather(os.path.join(self.csv_path, f'{parameter}.feather'))

    def extract_nc(self, year):
        """
        :return: writes csv files with the s2 information per parameter
        """
        print(f'start: {year} at {datetime.datetime.now()}')

        self.createcsv(year)

        file_path = os.path.join(self.basepath, str(year))
        files = os.listdir(file_path)[::5]

        # find files that were already extracted
        test_file = pd.read_csv(os.path.join(self.csv_path, f'{self.parameters[-1]}_{year}.csv'))
        done_dates = [str(d)[-3:] for d in test_file.date]
        files = [f for f in files if f not in done_dates]

        #Establish file where to store values
        fields_data = pd.DataFrame(data=None, index=range(1), columns=self.row_head)
        final_dict = {p: fields_data.copy() for p in self.parameters}

        for f, file in enumerate(files):
            print(f'{f} from {len(files)} done in {year}')
            if self.data == 'ERA5-Land':
                file_name = os.listdir(os.path.join(file_path, file))[2]
                path_open = os.path.join(file_path, file, file_name)
            else:
                path_open = os.path.join(file_path, file)
            src = xr.open_dataset(path_open)
            src = src.rio.write_crs(self.fields.crs)

            for parameter in self.parameters:
                final_dict[parameter].iloc[0, :] = [np.nan]*len(final_dict[parameter].columns)
                final_dict[parameter].iloc[0, 0] = str(year)+file
                src_para = src[parameter]

                for ipol in range(len(self.fields)):
                    polygon = self.fields[ipol:ipol + 1]
                    try:
                        out_image = src_para.rio.clip(polygon.geometry.values, polygon.crs)
                        out_image = np.where(out_image == 0, np.nan, out_image)
                        final_dict[parameter].iloc[0, ipol+1] = np.nanmedian(out_image)
                    except:
                        centroid = polygon.centroid
                        closest_point = src_para.sel(lat=centroid.y.values[0], lon=centroid.x.values[0],
                                                    method='nearest')
                        final_dict[parameter].iloc[0, ipol + 1] = closest_point.values[0]

                src_para.close()

                with open(os.path.join(self.csv_path, f'{parameter}_{year}.csv'), 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow(list(final_dict[parameter].iloc[0, :]))

            src.close()
            gc.collect()

        print(f'end: {year} at {datetime.datetime.now()}')

    def extract_sf(self):
        ds = xr.open_dataset(r'D:\DATA\yipeeo\Predictors\ecmwf_forecast.grib', engine='cfgrib', filter_by_keys={'dataType': 'fcmean'})
        ds = ds.rio.write_crs(self.fields.crs)
        forecast_t = ds.step.values/(3600*24*10**9)
        months = ds.time.values
        ds_lats = ds.latitude.values
        ds_lons = ds.longitude.values
        params = ['t2m', 'erate', 'msnsrf', 'tprate']

        for parameter in params[1:]:
            path_out = os.path.join(self.csv_path, parameter)
            if not os.path.exists(path_out): os.makedirs(path_out)
            print(parameter)
            src_para = ds[parameter]
            nan_fields = []
            for ipol in range(len(self.fields)):
            # for ipol in range(648, 649):
                # Establish file where to store values
                field_data = np.full(shape=ds[parameter].values.shape[1:3], fill_value=np.nan)
                polygon = self.fields[ipol:ipol + 1]
                try:
                    out_image = src_para.rio.clip(polygon.geometry.values, polygon.crs)
                    # out_image.to_netcdf(r'D:\data-write\YIPEEO\predictors\sf\test_2.nc')
                    field_data[:, :] = np.nanmean(out_image.values, axis=(0,3,4))
                except rio.exceptions.NoDataInBounds:
                    lat, lon = np.round(polygon.centroid.y, 0).values, np.round(polygon.centroid.x, 0).values
                    lat_i, lon_i = np.where(ds_lats==lat)[0], np.where(ds_lons==lon)[0]
                    # print(src_para.values[:, :, :, lat_i, lon_i][:, :, :, 0].shape)
                    field_data[:, :] = np.nanmean(src_para.values[:, :, :, lat_i, lon_i][:, :, :, 0], axis=0)

                field_data_c = pd.DataFrame([list(x[~np.isnan(x)]) for x in field_data], index=months, columns=[f'LT{a+1}' for a in range(4)]).reset_index()
                field_data_c.to_feather(os.path.join(path_out, f'{polygon.NUTS_ID.values[0]}_{parameter}.feather'))

                print(f'{ipol}/{len(self.fields)} done for {parameter}')

            # with open(os.path.join(self.csv_path, f'read_me_{parameter}.txt'), 'w') as file:
            #     file.write(f'Regions with no sf data available are:{nan_fields}')

            src_para.close()
            gc.collect()

    def parallelrun(self):
        # years = range(2000, 2023)
        years = [2001, 2004, 2005, 2006, 2007, 2017, 2019, 2022]
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as pool:
            for year in years:
                pool.submit(self.extract_nc, year=year)

    def createcsv(self, year):
        """
        :param n_cores: int number of cores on which calculation is distributed
        :return: prepares the csv files required for self.file2tab
        """
        for parameter in self.parameters:
            file_path = os.path.join(self.csv_path, f'{parameter}_{year}.csv')

            if not os.path.exists(file_path):
                with open(file_path, 'w') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow(self.row_head)

    def table2nc(self):
        """
        :return:combines all csv files with the data of the individual S-2 L2A Bands to one netcdf file
        """
        #Load first file to get all fields names

        file_names = os.listdir(self.csv_path)
        file_names = [f for f in file_names if f.endswith('.feather')]
        file_path = os.path.join(os.path.join(self.csv_path, file_names[0]))
        file = pd.read_feather(file_path)
        fields_o = file.columns[2:]
        #Filter fields with only nan values
        nan_fields = file.iloc[:,2:].isna().all(axis=0)
        fields = fields_o[~nan_fields]

        nc_path = os.path.join(self.csv_path, 'nc')
        if not os.path.exists(nc_path):
            os.makedirs(nc_path)

        if np.sum(nan_fields)>0:
            with open(os.path.join(nc_path, 'read_me.txt'), 'w') as file:
                file.write(f'There are fields with no pixels of ERA5-Land data inside. These are the fields:{fields_o[nan_fields].values}')

        #Define target dataframe with the correct dates to which the files are mapped to
        ts = pd.date_range(start='1/1/2000', end='31/12/2022', freq='1D')
        ts_s = [str(a.strftime('%Y%j')) for a in ts]
        target_var = pd.Series(data=None, index=ts_s)

        #Loop through all fields to establish one nc field per field with all bands
        for field in fields:
            #loop through all bands to collect information of all bands per field
            for b, file_name in enumerate(file_names):
                file_path = os.path.join(os.path.join(self.csv_path, file_name))
                file = pd.read_feather(file_path)

                if self.data=='VODCA':
                    new_date = [datetime.datetime.strptime(fi.split('_')[-1].split('.')[0], '%Y-%m-%d') for fi in file.date]
                    file.date = [str(a.strftime('%Y%j')) for a in new_date]
                elif self.data == 'CCI_SM':
                    new_date = [datetime.datetime.strptime(fi.split('-')[-2][:8], '%Y%m%d') for fi in file.date]
                    file.date = [str(a.strftime('%Y%j')) for a in new_date]

                #Establish pandas series with ERA5-Land data as loaded in the file and merge it with the target series
                med_field = pd.Series(data=file.loc[:, field].values, index=[str(a) for a in file.date])

                _, med_daily = target_var.align(med_field, axis=0)
                med_daily.index = ts

                #Establish new xr Dataset in first loop. Afterwards add bands to this file.
                if b == 0:
                    xr_file = xr.Dataset.from_dataframe(pd.DataFrame(med_daily, columns=[file_name.split('.')[0]]))
                else:
                    xr_file[file_name.split('.')[0]] = xr.DataArray.from_series(med_daily)

            xr_file = xr_file.rename({'index':'time'})
            xr_file.to_netcdf(path=os.path.join(nc_path, f'{field}.nc'))

    def merge_files(self):
        """
        This function merges all tables that were established with self.run to one single table. Is only needed if
        s2 extraction is run in parallel
        :return: writes csv file
        """

        for collection in self.parameters:
            files = os.path.join(self.csv_path, f"{collection}*.csv")
            files = glob.glob(files)
            df = pd.concat(map(pd.read_csv, files), ignore_index=True)
            #ToDo Done add to only merge _all.csv
            df = df.drop_duplicates(subset='date')
            df = df.reset_index()
            rename_cols = {a: a[:-3] for a in df.columns[2:]}
            df = df.rename(columns=rename_cols)
            df.to_feather(os.path.join(self.csv_path, f"{collection}.feather"))

    def detrend_anomalies(self, detrend=True):
        """
        :return: Takes nc files from self.table2nc, calculates its detrended anomalies and saves it again as nc
        """
        path = os.path.join(self.csv_path, 'nc')
        files = [a for a in os.listdir(path) if a.endswith('.nc') if not a.endswith('anomalies.nc')]
        print(files)

        for file in files:
            df = xr.open_dataset(os.path.join(path, file))
            for parameter in self.parameters:
                # Skip parameter if there are only nan values
                if np.sum(np.isnan(df[parameter].values)) == df[parameter].values.size:  # if all values are NaNs
                    continue

                df_par = df[parameter]
                if detrend:
                # Detrend signal
                    dtv = signal.detrend(df_par.values[df_par.notnull()])
                    df_par.values[df_par.notnull()] = dtv

                # Calculate anomalies
                df_orig = pd.DataFrame(data=df_par.values, index=df.time.values)
                # The size of the moving_average window [days] that will be applied on the input Series
                # (gap filling, short-term rainfall correction)
                # first 'smooting' of input data over 5 days (really days, not time steps)
                w_orig = 20
                # The size of the moving_average window [days] that will be applied on the calculated
                # climatology (long-term event correction)
                # second 'smooting' of calculated climatology over 30 days (really days, not time steps)
                w_clim = 30  # 22 for ERA5

                # calculate climatology
                df_clim = anomaly.calc_climatology(df_orig.copy().squeeze(), moving_avg_orig=w_orig,
                                                   moving_avg_clim=w_clim,
                                                   wraparound=True, respect_leap_years=True)
                # calculate anomalies
                df[parameter].values = anomaly.calc_anomaly(df_orig.copy().squeeze(), climatology=df_clim.dropna().copy(),
                                          respect_leap_years=True)

            if detrend:
                df.to_netcdf(os.path.join(path, file.split('.')[0]+'_detrended_anomalies.nc'))
            else:
                df.to_netcdf(os.path.join(path, file.split('.')[0]+'_anomalies.nc'))


class modis:
    """
    Class to retrieve ecostress data and extract it to field level
    """
    collections = ['evi']
    def __init__(self):
        self.nuts_file_path = 'D:/DATA/yipeeo/SC2/Crop yield/All_NUTS.shp'
        self.table_path = f'D:/data-write/YIPEEO/predictors/modis'

        if os.name=='posix':
            self.syspath = '/home/pbueechi/thinclient_drives/'
            self.table_path=os.path.join(self.syspath, self.table_path)
            self.nuts_file_path = os.path.join(self.syspath, self.nuts_file_path)

    def extract_shp(self):
        # Step 1: Read the GeoTIFF file using rioxarray
        path_in = r'D:\DATA\yipeeo\Predictors\MODIS_NDVI\tif'
        path_out_file = os.path.join(self.table_path, 'MODIS_nuts.csv')

        # Step 2: Read the shapefile using geopandas
        shapes = gpd.read_file(self.nuts_file_path)
        shapes = shapes.set_crs(epsg=4326)

        row_head = ['date'] + list(self.fields.NUTS_ID)


        if not os.path.exists(path_out_file):
            with open(path_out_file, 'w') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(row_head)

        files = [a for a in os.listdir(path_in) if a.endswith('.tif')]

        files = self.find_existing(path_file=path_out_file, files=files)

        for f, file in enumerate(files):
            print(f'start file {f} from {len(files)}')
            fields_data = pd.DataFrame(data=None, index=range(1), columns=row_head)
            time_stamp = file.split('.')[1][1:]
            fields_data.iloc[0, 0] = str(datetime.strptime(time_stamp, '%Y%j').date())
            tif_dataset = rio.open_rasterio(os.path.join(path_in, file), masked=True)
            # Step 3: Reproject the raster to match the shapefile CRS, if necessary
            if tif_dataset.rio.crs != shapes.crs:
                tif_dataset = tif_dataset.rio.reproject(shapes.crs)

            # Step 4: Compute zonal statistics (mean of pixels within each polygon)
            stats = zonal_stats(
                vectors=shapes,
                raster=tif_dataset.values[0],  # Use the first band of the raster
                affine=tif_dataset.rio.transform(),
                stats=["median"]
            )
            # Step 5: Add the computed mean to the GeoDataFrame
            fields_data.iloc[0, 1:] = [stat["median"] for stat in stats]
            with open(path_out_file, 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(list(fields_data.iloc[0, :]))

    def find_existing(self, path_file, files):
        done_files = pd.read_csv(path_file)
        done_dates = list(done_files.date)
        search_new = []
        for f, file in enumerate(files):
            date_time_a = str(datetime.strptime(file.split('.')[1][1:], '%Y%j').date())
            if not date_time_a in done_dates:
                search_new.append(file)
        return search_new

    def hdf2tif(self):
        path_in = r'D:\DATA\yipeeo\Predictors\MODIS_NDVI\MOD13C1_061-20240809_085239'
        path_out = r'D:\DATA\yipeeo\Predictors\MODIS_NDVI\tif'

        shapefile_path = 'D:/DATA/yipeeo/SC2/Crop yield/All_NUTS.shp'
        shapes = gpd.read_file(shapefile_path)
        shapes = shapes.set_crs(epsg=4326)

        files = os.listdir(path_in)
        for file in files:
            hdf_dataset = rio.open_rasterio(os.path.join(path_in, file), masked=True)
            evi = hdf_dataset['CMG 0.05 Deg 16 days EVI']
            if (evi.add_offset == 0) and (evi.scale_factor == 10000):
                evi = evi.rio.clip_box(minx=-6, miny=36, maxx=45, maxy=60)
                evi = evi.rio.reproject(shapes.crs)
                evi.rio.to_raster(os.path.join(path_out, file.replace('.hdf', '.tif')))
            else:
                raise ValueError(f'file {file} has a different scale factor and/or offset')

    def rename_cols(self):
        file = pd.read_csv(os.path.join(self.table_path, 'MODIS_nuts.csv'), index_col=0)
        col_names = pd.read_csv(r'D:\data-write\YIPEEO\predictors\ERA5-Land\EU\t2m_2001.csv', index_col=0)
        file.columns = col_names.columns
        file.to_csv(os.path.join(self.table_path, 'MODIS_nuts_rn.csv'))

    def table2nc(self):
        """
        :return:combines all csv files with the data of the ECOSTRESS data collection to one netcdf file.
        Needs the results of self.merge_files
        """

        # Load first file to get all fields names
        file_path = os.path.join(self.table_path, 'MODIS_nuts_rn.csv')
        file = pd.read_csv(file_path, index_col=0)
        fields = file.columns

        # Filter fields with only nan values
        nan_fields = file.isna().all(axis=0)
        fields_t = fields[~nan_fields]
        fields_f = fields[nan_fields]

        if not os.path.exists(os.path.join(self.table_path,'nc')):
            os.makedirs(os.path.join(self.table_path, 'nc'))

        if np.sum(nan_fields) > 0:
            with open(rf'{self.table_path}\nc\read_me.txt', 'w') as txt_file:
                txt_file.write(
                    f'There are fields with no pixels of MODIS data inside. These are the fields:{fields_f.values}')

        # Loop through all fields to establish one nc field per field with all bands
        for f, field in enumerate(fields_t):
            # Remove rows with only nan
            nan_fields = file.isna().all(axis=1)
            file = file[~nan_fields.values]

            # Establish a pandas Series as target variable with daily timestep from 2019 to 2022
            dates_all = pd.date_range(start='1/1/2000', end='31/12/2024')
            target_var = pd.Series(data=None, index=dates_all)

            # Establish pandas series with sentinel-2 data as loaded in the file and merge it with the target series
            dates = [datetime.datetime.strptime(a.split('_')[0], '%Y-%m-%d') for a in file.index]
            med_field = pd.Series(data=file.loc[:, field].values/10000, index=dates)

            # resample to daily values for cases where there are several observations per day
            med_field = med_field.resample('D').mean()

            # med_daily = med_daily.replace(np.nan, -9999)

            # Establish new xr Dataset in first loop. Afterwards add bands to this file.
            xr_file = xr.Dataset.from_dataframe(pd.DataFrame(med_field, columns=[f'evi_median']))
            xr_file[f'evi_median'].attrs = dict(
                # FillValue=np.nan,
                short_name='EVI',
                long_name=f'Enhanced Vegetation Index',
                description='EVI extracted from MODIS13C1 16 daily data 0.05° resolution. Contains median observation of all pixels laying within given NUTS area'
            )

            xr_file = xr_file.rename({'index': 'time'})
            # print(xr_file.time)
            # comp = dict(zlib=True, complevel=9)
            # encoding = {var: comp for var in xr_file.data_vars}
            xr_file.to_netcdf(os.path.join(self.table_path, 'nc', f'{field}.nc'))

    def detrend_anomalies(self, detrend=True):
        """
        :return: Takes nc files from self.table2nc, calculates its detrended anomalies and saves it again as nc
        """
        path = os.path.join(self.table_path, 'nc')
        files = [a for a in os.listdir(path) if a.endswith('.nc')]
        parameter = 'evi_median'

        for file in files:
            df = xr.open_dataset(os.path.join(path, file))
            # Skip parameter if there are only nan values
            if np.sum(np.isnan(df[parameter].values)) == df[parameter].values.size:  # if all values are NaNs
                continue

            df_par = df[parameter]
            if detrend:
            # Detrend signal
                dtv = signal.detrend(df_par.values[df_par.notnull()])
                df_par.values[df_par.notnull()] = dtv

            # Calculate anomalies
            df_orig = pd.DataFrame(data=df_par.values, index=df.time.values)
            # The size of the moving_average window [days] that will be applied on the input Series
            # (gap filling, short-term rainfall correction)
            # first 'smooting' of input data over 5 days (really days, not time steps)
            w_orig = 20
            # The size of the moving_average window [days] that will be applied on the calculated
            # climatology (long-term event correction)
            # second 'smooting' of calculated climatology over 30 days (really days, not time steps)
            w_clim = 30  # 22 for ERA5

            # calculate climatology
            df_clim = anomaly.calc_climatology(df_orig.copy().squeeze(), moving_avg_orig=w_orig,
                                               moving_avg_clim=w_clim,
                                               wraparound=True, respect_leap_years=True)
            # calculate anomalies
            df[parameter].values = anomaly.calc_anomaly(df_orig.copy().squeeze(), climatology=df_clim.dropna().copy(),
                                      respect_leap_years=True)

            if detrend:
                path_out = os.path.join(path, 'detrended_anomalies')
            else:
                path_out = os.path.join(path, 'anomalies')
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            df.to_netcdf(os.path.join(path_out, file))

def merge_files(dataset):
    if dataset == 'CCI_SM':
        parameter = ['sm']
    elif dataset == 'VODCA':
        parameter = ['VODCA_CXKu']
    elif dataset == 'ERA5-Land':
        parameter = ['t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr', 'lai_lv']
    else:
        raise ValueError('dataset not available')
    print(parameter)
    for collection in parameter:
        path = fr'M:\Projects\YIPEEO\07_data\Predictors\SC2\{dataset}'
        files = os.path.join(path, f"{collection}*.csv")
        files = glob.glob(files)
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        # ToDo Done add to only merge _all.csv
        df = df.drop_duplicates(subset='date')

        df.to_csv(os.path.join(os.path.dirname(path), f"{collection}.csv"), index=False)

def add_date2df(dataset):
    path = fr'M:\Projects\YIPEEO\07_data\Predictors\SC2\{dataset}.csv'
    df = pd.read_csv(path)
    print(df)
    if dataset == 'VODCA_CXKu':
        # df['date'] = [a[-13:-2] for a in df['date']]
        # df['date'] = pd.to_datetime(df['date'])
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    elif dataset == 'sm':
        df['date'] = [a.split('-')[-2][:8] for a in df['date']]
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['date'] = pd.to_datetime(df['date'], format='%Y%j')

    df = df.sort_values('date')
    df.to_csv(path, index=False)

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    dt = datetime.datetime.now()
    # a = coarse_scale_data(region='Czechia', data='VODCA')
    a = coarse_scale_data(region='Czechia', data='CCI_SM')
    # a.merge_files()
    # a.parallelrun()
    # years = range(2000, 2023)
    # for year in years:
    #     a.extract_nc(year)
    # print(mp.cpu_count())
    # a.rename_cols()
    # a.detrend_anomalies(detrend=False)

    print(f'calculation took {datetime.datetime.now()-dt}')
    # a.table2nc()