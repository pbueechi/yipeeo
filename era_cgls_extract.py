import os
import gc
import rasterio
import csv
import numpy as np
import pandas as pd
import geopandas as gpd
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

        self.fields = gpd.read_file(fr'D:\DATA\yipeeo\Crop_data\Crop_yield\{region}\maize.shp')
        self.fields = self.fields.set_crs(epsg=4326)
        self.fields = self.fields.drop_duplicates(subset='g_id')
        self.row_head = ['date'] + [str(int(i)) for i in list(self.fields.g_id.values)]
        self.time_of_interest = f'2016-01-01/2022-12-31'

    def extract_data(self):
        """
        :return: writes csv files with the s2 information per parameter
        """
        # basepath = '/eodc/private/yipeeo/ERA5_land'
        basepath = r'D:\DATA\yipeeo\Predictors\ERA5-Land'
        parameters = os.listdir(basepath)
        for parameter in parameters[2:3]:
            file_path = os.path.join(basepath, parameter)
            files = os.listdir(file_path)
            files = [f for f in files if f.endswith('.tif')]
            dates = [a.split('_')[-1].split('.')[0] for a in files]

            #Convert field to same crs as ERA5-Land data
            ref_crs = rasterio.open(os.path.join(file_path, files[0]))
            fields = self.fields.to_crs(ref_crs.crs)

            #Establish file where to store values
            fields_data = pd.DataFrame(data=None, index=range(len(dates)), columns=self.row_head)
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

if __name__=='__main__':
    a = coarse_scale_data('Austria', 'ERA5-Land')
    a.extract_data()