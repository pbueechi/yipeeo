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

def ukr_reg():
    path_data = r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\NUTS3_Ukraine_corr.csv'
    path_shp = r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\ukr_admbnda_adm1_sspe_20230201.shp'
    data_file = pd.read_csv(path_data, decimal=',', sep=';', index_col=0)
    cols = data_file.columns
    crops = ['corn grain', 'wheat', 'barley']
    crop = crops[2]
    col_crop = [col for col in cols if col.endswith(crop)]
    crop_file = data_file.loc[:,col_crop]
    plt.plot(range(2017,2023), crop_file.loc['Ukraine', :]-np.mean(crop_file.loc['Ukraine', :]))
    plt.plot(range(2017, 2023), crop_file.loc['Khersonska', :]-np.mean(crop_file.loc['Khersonska',:]))
    plt.plot(range(2017, 2023), crop_file.loc['Luhanska', :]-np.mean(crop_file.loc['Luhanska',:]))
    plt.plot(range(2017, 2023), crop_file.loc['Donetska', :]-np.mean(crop_file.loc['Donetska',:]))
    plt.grid(linestyle="--", alpha=0.5, zorder=1)
    plt.legend(['Ukraine_mean', 'Kherson', 'Luhansk', 'Donetska'])
    plt.xlabel('Year')
    plt.ylabel('Yield [dt/ha]')
    plt.title(crop)
    plt.show()
    print(crop_file)
    # plt.savefig(f'Figures/Ukraine/yield_{crop}_anom.png', dpi=300)

    # shp = gpd.read_file(path_shp)
    # # print(shp)
    # # print(data_file)
    # cols = data_file.columns
    # # for crop in ['crop grain','wheat', 'barley']:
    # for crop in ['corn grain']:
    #     col_crop = [col for col in cols if col.endswith(crop)]
    #     new_shp = shp.copy()
    #     print(new_shp)
    #     print(col_crop)

def eu_tab2shp():
    crop_o = pd.read_csv(r'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\Joint_DB_Clim4Cast.csv', sep=';')
    nuts_file_path = 'D:/DATA/yipeeo/SC2/Crop yield/All_NUTS.shp'
    shape = gpd.read_file(nuts_file_path)
    years = [str(a) for a in range(2000, 2023)]
    df = pd.DataFrame(index=np.unique(shape.g_id), columns=years)
    for crop in ['Maize', 'Winter Wheat', 'Spring Barley']:
        df_crop = crop_o.iloc[np.where(crop_o.Crop_type==crop)[0],:]
        df_crop.index = range(len(df_crop.index))
        for i in range(len(df_crop.index)):
            this_ind = np.where(df.index==df_crop.Region_id[i])[0]
            if len(this_ind)==1:
                if (crop=='Spring Barley') and (df_crop.Region_id[i].startswith('DE')):
                    df.iloc[this_ind,:] = df_crop.iloc[i, 3:]/10
                else:
                    df.iloc[this_ind,:] = df_crop.iloc[i, 3:]

        df.to_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_test.csv')

def eu_tab_fr():
    shape_path = r'M:\Projects\YIPEEO\07_data\Crop yield\France'

    crops = ['Maize', 'Spring Barley', 'Winter Wheat']
    for crop in crops:
        orig = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_test.csv', index_col=0)
        fra = gpd.read_file(os.path.join(shape_path, f'nuts_fr_{crop}.shp'))
        fra_df = fra.iloc[:,-20:-1]
        fra_df.index = fra.loc[:, 'NUTS_ID']
        for inds in fra_df.index:
            if inds in orig.index:
                orig.loc[inds,[str(a) for a in range(2000,2019)]] = fra_df.loc[inds, :]

        orig.to_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_FRA.csv')

def eu_tab_hu():
    pass



        # df.to_csv(f'Data/M/EU/yields_{crop}.csv', index=True)
    # yield_path_wheat = {'AT': [6, 31], 'CZ': [7, 16], 'DE': [7, 16], 'FR': [7, 11], 'HR': [6, 31],
    #                       'HU': [6, 31], 'PL': [7, 16], 'SI': [7, 16], 'SK': [7, 16], 'UA': [7, 16]}


def merge_hu():
    path_csv = r'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv'
    path_shp = r'D:\DATA\yipeeo\Crop_data\NUTS_data\hut_nuts3.shp'
    shp = gpd.read_file(path_shp)
    hu_files = ['HU_spring_barley.csv', 'HU_winter_wheat.csv']
    crop_conv = {'maize': 'grain maize and corn-cob-mix', 'spring_barley': 'spring barley',
             'winter_wheat': 'common wheat and spelt'}
    for file in hu_files:
        a = pd.read_csv(os.path.join(path_csv, file), decimal=',', index_col=0)
        for year in (range(2016, 2023)):
            a.loc[:,str(year)] = [np.nan]*len(a.index)
        crop = file.split('.')[0][3:]
        crop_type = crop_conv[crop]
        crop_df = shp.iloc[np.where(shp.crop_type==crop_type)[0],[1,2,7]]
        crop_df.index = range(len(crop_df.index))
        for i in crop_df.index:
            a.loc[crop_df.iloc[i, 0], crop_df.iloc[i, 1]] = crop_df.iloc[i, 2]

        a.to_csv(os.path.join(path_csv, file.split('.')[0]+'_new.csv'), index=True)

def ukr2crop():
    path = r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\NUTS3_Ukraine.csv'
    file = pd.read_csv(path, sep=';', decimal=',')
    crops = np.unique([a[5:] for a in file.columns[2:]])
    for crop in crops:
        inds = [a for a in file.columns if a.endswith((crop, 'Regions', 'Region_ID'))]
        this_crop = file.loc[:,inds]
        this_crop.columns = [a.replace(f'_{crop}','') for a in this_crop.columns]
        this_crop.to_csv(rf'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\UA_{crop}.csv', index=False)

if __name__ == '__main__':
    # pd.set_option('display.max_rows', None)
    # eu_tab2shp()
    eu_tab_fr()
    # ukr2crop()
    # merge_hu()
    # a = france_data(r'D:\DATA\yipeeo\NUTS_data')

    # b = austria_data()
    # c = crop_data(country='austria')
    # c.add_yield()
    # ukr_reg()
    # a.select_country()
    # pd.set_option('display.max_columns', None)
    # b.add_yield()
    # ukr2shape()
    print('all done')