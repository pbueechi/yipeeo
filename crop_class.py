import os
import random
import itertools
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime

def select_crop():
    name_conv = {'MAIS CORN-COB-MIX (CCM)': 'maize', 'SOMMERGERSTE': 'spring_barley',
                 'WINTERWEICHWEIZEN': 'winter_wheat', 'WINTERHARTWEIZEN (DURUM)': 'durum_wheat'}
    for year in range(2016,2023):
        print(f'started {year} at {datetime.now()}')
        path = rf'M:\Projects\YIPEEO\07_data\LPIS\Austria\inspire_schlaege_{year}_polygon.gpkg\INSPIRE_SCHLAEGE_{year}_POLYGON.gpkg'
        file = gpd.read_file(path)
        print(file.crs)
        file = file.set_crs(epsg=31287, allow_override=True)
        file = file.to_crs(epsg=4326)
        locs = file.SNAR_BEZEICHNUNG
        file = file.drop(['GEO_ID', 'INSPIRE_ID', 'GML_ID', 'GML_IDENTIFIER', 'SNAR_CODE', 'GEO_PART_KEY', 'LOG_PKEY',
                          'GEOM_DATE_CREATED', 'FART_ID', 'GEO_TYPE', 'GML_GEOM', 'GML_LENGTH' ], axis=1)
        for crop in ['WINTERWEICHWEIZEN', 'WINTERHARTWEIZEN (DURUM)', 'SOMMERGERSTE', 'MAIS CORN-COB-MIX (CCM)']:
        # for crop in ['WINTERWEICHWEIZEN']:
            print(f'started {crop} at {datetime.now()}')
            inds = np.where(locs == crop)[0]
            new_file = file.iloc[inds, :]
            print(new_file.crs)
            new_file.to_file(rf'D:\DATA\yipeeo\Crop_data\Crop_class\{name_conv[crop]}_{year}.shp', crs='EPSG:4326')

def add_nuts():
    file_path = r'D:\DATA\yipeeo\Crop_data\Crop_class'
    files = [a for a in os.listdir(file_path) if a.endswith('.shp') and not a.startswith('durum')]
    nuts_file = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\Crop_yield\Austria\maize.shp')
    nuts_file = nuts_file.drop(['g_name']+[f'{int(a)}' for a in range(2000,2023)], axis=1)
    for file in files:
        field_file = gpd.read_file(os.path.join(file_path, file))
        field_file = field_file.set_crs(epsg=4326)
        field_file.loc[:,'doi'] = range(field_file.shape[0])
        overlay = gpd.overlay(field_file, nuts_file)
        overlay = overlay.drop_duplicates(subset='doi')
        overlay = overlay.drop(['doi'], axis=1)
        overlay.to_file(os.path.join(file_path, 'nuts', file), crs='EPSG:4326')

def convert_czr():
    path = r'D:\DATA\yipeeo\Crop_data\Crop_class\CZR'
    path_to = r'D:\data-write\YIPEEO\Crop type\CZR'
    nuts4_file = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\NUTS_data\czr_nuts4.shp')
    nuts3_file = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\NUTS_data\czr_nuts3.shp')
    nuts4_file = nuts4_file.set_crs(epsg=4326)
    nuts3_file = nuts3_file.to_crs(epsg=4326)
    nuts3_file = nuts3_file.drop(['CNTR_CODE', 'NAME_LATN', 'NUTS_NAME', 'MOUNT_TYPE', 'URBN_TYPE', 'COAST_TYPE', 'FID', 'LEVL_CODE'], axis=1)
    nuts4_file = nuts4_file.drop(['name_latn'], axis=1)
    files = [a for a in os.listdir(path) if a.endswith('.shp')]
    for file in files:
        shp = gpd.read_file(os.path.join(path, file))
        shp = shp.to_crs(epsg=4326)
        # shp = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\Crop_class\test\test_2016.shp')
        year = int(file.split('_')[1])
        print(year)
        shp.loc[:, 'FS_KENNUNG'] = [f'{year}_{a}' for a in range(shp.shape[0])]
        '''
        g_id for nuts4, nuts3
        '''

        overlay3 = gpd.overlay(shp, nuts3_file)
        overlay4 = gpd.overlay(shp, nuts4_file)

        overlay = overlay3.merge(overlay4, on='FS_KENNUNG', how='outer')
        overlay = overlay.drop(['AREA_GEO_y', 'geometry_y', 'CropName_y'], axis=1)
        overlay = overlay.rename(columns={'AREA_GEO_x':'AREA_GEO', 'NUTS_ID': 'NUTS3', 'geometry_x': 'geometry', 'nut_id': 'NUTS4'})
        overlay_gdf = gpd.GeoDataFrame(data=overlay, geometry=overlay.geometry, crs=4326)
        overlay_gdf = overlay_gdf.drop_duplicates(subset='FS_KENNUNG')

        for c_type in ['winter wheat', 'spring barley', 'maize']:
            print(c_type)
            a = overlay_gdf[overlay_gdf['CropName_x']==c_type]
            a.to_file(os.path.join(path_to, f'{c_type}_{year}.shp'), crs='EPSG:4326')

def reduce_fields(country, number_of_fields=1000):
    path = os.path.join(r'D:\DATA\yipeeo\Crop_data\Crop_class', country)
    if country=='Austria':
        atr = 'g_id'
        atr_area = 'SL_FLAECHE'
    elif country=='Czechia':
        atr = 'NUTS4'
        atr_area = 'AREA_GEO'
    else:
        raise ValueError(f'country {country} not available')
    file_names = [a for a in os.listdir(path) if a.endswith('.shp')]
    for file_name in file_names:
        shp = gpd.read_file(os.path.join(path, file_name))
        # Remove fields below 0.1 ha
        small_fields = np.where(shp[atr_area]<0.1)[0]
        shp = shp.drop(small_fields, axis=0)
        shp.index = range(shp.shape[0])
        counts = shp.value_counts(subset=[atr])
        region2many = np.where(counts>number_of_fields)[0]
        regs = list(itertools.chain(*counts[region2many].keys()))
        removs = []
        for reg in regs:
            shp_reg = np.where(shp[atr]==reg)[0]
            to_remove = random.sample(list(shp_reg), len(shp_reg)-number_of_fields)
            removs.append(to_remove)
        removs = list(itertools.chain(*removs))
        shp_new = shp.drop(removs, axis=0)
        if not os.path.exists(os.path.join(path, f'subsample_{number_of_fields}')):
            os.makedirs(os.path.join(path, f'subsample_{number_of_fields}'))
        shp_new.to_file(os.path.join(path, f'subsample_{number_of_fields}', file_name), crs='EPSG:4326')

def rename_files(path):
    files = os.listdir(path)
    for file in files:
        os.rename(os.path.join(path, file), os.path.join(path, file.replace(' ', '_')))


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    path = r'D:\DATA\yipeeo\Crop_data\Crop_class\Czechia\subsample_1000'
    rename_files(path)
    # for country in ['Czechia', 'Austria']:
    #     reduce_fields(country, 200)
    print('done all')