import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
# from unidecode import unidecode

class france_data():
    def __init__(self, basepath):
        self.basepath = basepath

    def select_country(self, country='FR'):
        nuts = gpd.read_file(os.path.join(self.basepath, 'NUTS_RG_01M_2021_3035.shp'))
        nuts_fr_ind = np.where((nuts.CNTR_CODE==country) & (nuts.LEVL_CODE==3))[0]
        nuts_country = nuts.iloc[nuts_fr_ind,:]
        dep_name = nuts_country.NUTS_NAME
        dep_name = [d.replace('-','_') for d in dep_name]
        # dep_name = [unidecode(d.replace('’', '_')) for d in dep_name]
        nuts_country.NUTS_NAME = dep_name
        # nuts_country.to_file(os.path.join(self.basepath, 'nuts_fr.shp'))

    def add_yield(self):
        yield_data_path = r'D:\DATA\yipeeo\France\2021-001_Schauberger-et-al_Data\2021-001_Schauberger-et-al_Data_FILTERED'
        crops = ['barley_spring','barley_winter','maize_total','oats_spring','oats_winter','potatoes_total','rape_spring','rape_winter','sugarbeet_total','sunflower_total','wheat_durum_total','wheat_spring','wheat_winter','wine_total']
        file_exist = [os.path.exists(f'{yield_data_path}/{crop}_data_1900-2018_FILTERED.txt') for crop in crops]
        if len(file_exist)>np.sum(file_exist):
            raise ValueError('Not all crops are available')

        new_cols = [str(f) for f in np.arange(1900,2019,1)]

        for crop in crops:
            print(crop)
            nuts = gpd.read_file(os.path.join(self.basepath, 'nuts_fr.shp'))
            nuts = nuts.to_crs(epsg=4326)  # project to WGS84
            nuts.loc[:, new_cols] = np.nan

            crop_yield = pd.read_csv(f'{yield_data_path}/{crop}_data_1900-2018_FILTERED.txt', sep=';')
            deps = nuts.NUTS_NAME
            for dep_i,dep in enumerate(deps):
                ind = np.where(crop_yield.department==dep)[0]
                if len(ind)==0:
                    print(f'department {dep} not in crop yield dataset')
                else:
                    yield_vals = crop_yield.iloc[ind,2].values
                    nuts.iloc[dep_i,-len(yield_vals):] = yield_vals
            nuts = nuts.dropna(1, how='all')
            nuts.to_file(os.path.join(self.basepath, f'nuts_fr_{crop}.shp'))

class crop_data():
    def __init__(self, country):
        self.country = country
        if country == 'austria':
            self.basepath = r'M:\Projects\YIPEEO\07_data\Crop yield\Austria'
            self.crop_file_name = 'F_pol_2000_2022_TU_Bueechi_07_2023.csv'
        elif country == 'czech':
            self.basepath = r'D:\DATA\yipeeo\Crop_yield\Czech_Rep'
            self.crop_file_name = 'CR_yields_NUTS4_2000_2022_encoded.csv'

    def add_yield(self):
        yield_data_path = os.path.join(self.basepath,self.crop_file_name)
        yield_data = pd.read_csv(yield_data_path, encoding='latin1', decimal=',')

        if self.country == 'austria':
            crops = np.unique(yield_data.Kulturart)
        elif self.country == 'czech':
            crops = yield_data.columns[4:]
            counties = np.unique(yield_data.NAME)

        new_cols = [str(f) for f in np.arange(2000,2023,1)]

        for crop in crops[:1]:
            print(crop)
            nuts = gpd.read_file(os.path.join(self.basepath, 'STATISTIK_AUSTRIA_POLBEZ_20230101.shp'))
            nuts = nuts.to_crs(epsg=4326)  # project to WGS84
            nuts.loc[:, new_cols] = np.nan

            crop_ind = np.where(yield_data.Kulturart==crop)[0]
            crop_yield = (yield_data.iloc[crop_ind,:])
            deps = nuts.g_id
            for dep_i,dep in enumerate(deps):
                ind = np.where(crop_yield.PolNr==int(dep))[0]
                if len(ind)==0:
                    print(f'department {dep} not in crop yield dataset')
                else:
                    yield_vals = crop_yield.iloc[ind,-len(new_cols):]/10
                    nuts.iloc[dep_i,-len(new_cols):] = yield_vals
        #     nuts = nuts.dropna(1, how='all')
        #     nuts.to_file(os.path.join(self.basepath, f'nuts_at_{crop}.shp'))

def ukr2shape():
    file = pd.read_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\Subfield\soybean\Soybean_2021_22.csv', decimal=',')
    # file = pd.read_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\Subfield\Winter_wheat\WinterWheat 4 fields and points.csv', decimal=',')
    print(file.columns)
    # points = gpd.points_from_xy(x=file.X, y=file.Y, crs='epsg:32635')
    gdf = gpd.GeoDataFrame(file, geometry=gpd.points_from_xy(x=file.X, y=file.Y), crs='epsg:32635')
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\Subfield\soybean_wgs84.shp')
    # print(gdf)


if __name__ == '__main__':
    # a = france_data(r'D:\DATA\yipeeo\NUTS_data')
    # b = austria_data()
    c = crop_data(country='austria')
    c.add_yield()
    # a.select_country()
    # pd.set_option('display.max_columns', None)
    # b.add_yield()
    # ukr2shape()
    print('all done')