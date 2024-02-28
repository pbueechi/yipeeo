import os
import gc
import rasterio
import csv
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
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
        if not os.path.exists((self.table_path)): os.makedirs(self.table_path)
        self.csv_path = os.path.join(self.table_path, region)
        if not os.path.exists(self.csv_path): os.makedirs(self.csv_path)
        self.basepath = r'D:/DATA/yipeeo/Predictors/ERA5-Land'
        self.nuts_file_path = fr'D:\DATA\yipeeo\Crop_data\Crop_yield\{self.region}\maize.shp'

    def extract_data(self, parameter):
        """
        :return: writes csv files with the s2 information per parameter
        """
        # basepath = '/eodc/private/yipeeo/ERA5_land'
        fields = gpd.read_file(self.nuts_file_path)
        fields = fields.set_crs(epsg=4326)
        fields = fields.drop_duplicates(subset='g_id')
        row_head = ['date'] + [str(int(i)) for i in list(fields.g_id.values)]

        file_path = os.path.join(self.basepath, parameter)
        files = os.listdir(file_path)
        files = [f for f in files if f.endswith('.tif')]
        dates = [a.split('_')[-1].split('.')[0] for a in files]

        #Convert field to same crs as ERA5-Land data
        ref_crs = rasterio.open(os.path.join(file_path, files[0]))
        fields = fields.to_crs(ref_crs.crs)

        #Establish file where to store values
        fields_data = pd.DataFrame(data=None, index=range(len(dates)), columns=row_head)
        fields_data.iloc[:, 0] = dates

        for f, file in enumerate(files):
            print(f'done: {file.split("_")[-1]} at {datetime.now()}')
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
        fields_data.to_feather(os.path.join(self.csv_path, f'{parameter}.feather'))

    def parallelrun(self):
        parameters = os.listdir(self.basepath)
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
            for parameter in parameters:
                pool.submit(self.extract_data, parameter=parameter)

    def table2nc(self):
        """
        :return:combines all csv files with the data of the individual S-2 L2A Bands to one netcdf file
        """
        #Load first file to get all fields names

        file_names = os.listdir(self.csv_path)
        file_names = [f for f in file_names if f.endswith('.feather')]
        file_path = os.path.join(os.path.join(self.csv_path, file_names[0]))
        file = pd.read_feather(file_path)
        fields_o = file.columns[1:]

        #Filter fields with only nan values
        nan_fields = file.iloc[:,1:].isna().all(axis=0)
        fields = fields_o[~nan_fields]

        nc_path = os.path.join(self.csv_path, 'nc')
        if not os.path.exists(nc_path):
            os.makedirs(nc_path)

        if np.sum(nan_fields)>0:
            with open(os.path.join(nc_path, 'read_me.txt'), 'w') as file:
                file.write(f'There are fields with no pixels of S-2 L2A data inside. These are the fields:{fields_o[nan_fields].values}')

        #Loop through all fields to establish one nc field per field with all bands
        for field in fields:
            #loop through all bands to collect information of all bands per field
            for b, file_name in enumerate(file_names):
                file_path = os.path.join(os.path.join(self.csv_path, file_name))
                file = pd.read_feather(file_path)
                #Establish a pandas Series as target variable with daily timestep from 2016 to 2022
                dates_all = pd.date_range(start=file.date[0], end=file.date[file.shape[0]-1])

                #Establish pandas series with sentinel-2 data as loaded in the file and merge it with the target series
                dates = [datetime.strptime(a.split('_')[0], '%Y-%m-%d') for a in file.iloc[:,0]]
                med_field = pd.Series(data=file.loc[:, field].values, index=dates)

                #Establish new xr Dataset in first loop. Afterwards add bands to this file.
                if b == 0:
                    xr_file = xr.Dataset.from_dataframe(pd.DataFrame(med_field, columns=[file_name.split('.')[0]]))
                else:
                    xr_file[file_name.split('.')[0]] = xr.DataArray.from_series(med_field)

            xr_file = xr_file.rename({'index':'time'})
            xr_file.to_netcdf(path=os.path.join(nc_path, f'{field}.nc'))

if __name__=='__main__':
    a = coarse_scale_data('Austria', 'ERA5-Land')
    # a.extract_data()
    a.table2nc()