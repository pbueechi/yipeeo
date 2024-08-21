import os
import gc
import rasterio
import warnings
import csv
import glob
import itertools
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
from pytesmo.time_series import anomaly
from scipy import signal
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from rasterio.mask import mask


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
        self.table_path = f'D:/data-write/YIPEEO/predictors/{data}'
        self.basepath = 'M:/Datapool/ECMWF_reanalysis/01_raw/ERA5-Land/datasets/images'
        self.parameters = ['t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr', 'lai_lv']

        if region=='EU':
            self.nuts_file_path = 'D:/DATA/yipeeo/SC2/Crop yield/All_NUTS.shp'
        else:
            self.nuts_file_path = f'D:/DATA/yipeeo/Crop_data/Crop_yield/{self.region}/maize.shp'
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
        self.fields = self.fields.drop_duplicates(subset='g_id')
        self.row_head = ['date'] + list(self.fields.g_id.values)

    def extract_data(self, parameter):
        """
        :return: writes csv files with the s2 information per parameter
        """
        fields = gpd.read_file(self.nuts_file_path)
        fields = fields.set_crs(epsg=4326)
        fields = fields.drop_duplicates(subset='g_id')
        # row_head = ['date'] + [str(int(i)) for i in list(fields.g_id.values)]
        row_head = ['date'] + list(fields.g_id.values)

        file_path = os.path.join(self.basepath, parameter)
        files = os.listdir(file_path)
        files = [f for f in files if f.endswith('.tif')][::5]
        dates = [a.split('_')[-1].split('.')[0] for a in files]

        #Convert field to same crs as ERA5-Land data
        ref_crs = rasterio.open(os.path.join(file_path, files[0]))
        fields = fields.to_crs(ref_crs.crs)

        #Establish file where to store values
        fields_data = pd.DataFrame(data=None, index=range(len(dates)), columns=row_head)
        fields_data.iloc[:, 0] = dates

        for f, file in enumerate(files[:5]):
            print(f'done: {file.split("_")[-1]} at {datetime.now()} - {parameter}')
            src = rasterio.open(os.path.join(file_path, file))

            for ipol in range(len(fields)):
                polygon = fields[ipol:ipol + 1]
                out_image, out_transform = mask(src, polygon.geometry, crop=True)
                out_image = np.where(out_image == 0, np.nan, out_image)
                if np.isnan(out_image).all():
                    fields_data.iloc[f, ipol+1] = np.nan
                else:
                    fields_data.iloc[f, ipol+1] = np.nanmedian(out_image)
            src.close()
            gc.collect()
        print(fields_data)
        fields_data.to_feather(os.path.join(self.csv_path, f'{parameter}.feather'))

    def extract_nc(self, year):
        """
        :return: writes csv files with the s2 information per parameter
        """
        print(f'start: {year} at {datetime.now()}')

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
            file_name = os.listdir(os.path.join(file_path, file))[2]
            src = xr.open_dataset(os.path.join(file_path, file, file_name))
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
                        final_dict[parameter].iloc[0, ipol + 1] = np.nan
                src_para.close()

                with open(os.path.join(self.csv_path, f'{parameter}_{year}.csv'), 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow(list(final_dict[parameter].iloc[0, :]))

            src.close()
            gc.collect()

        print(f'end: {year} at {datetime.now()}')

    def parallelrun(self):
        years = range(2000, 2023)
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
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
            df.to_feather(os.path.join(self.csv_path, f"{collection}.feather"))


    def detrend_anomalies(self, detrend=True):
        """
        :return: Takes nc files from self.table2nc, calculates its detrended anomalies and saves it again as nc
        """
        path = os.path.join(self.csv_path, 'nc')
        files = [a for a in os.listdir(path) if a.endswith('.nc')]

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



if __name__=='__main__':
    warnings.filterwarnings('ignore')
    dt = datetime.now()
    a = coarse_scale_data('EU', 'ERA5-Land')
    # a.parallelrun()
    # a.extract_nc(2000)
    # a.table2nc()
    a.detrend_anomalies(detrend=False)

    # for parameter in ['t2m', 'swvl1', 'PET_sum', 'Rn_sum', 'ETo_sum', 'VWC_1_avg'][:1]:
    #     a.extract_data(parameter=parameter)
    print(f'calculation took {datetime.now()-dt}')
    # a.table2nc()