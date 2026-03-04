import os
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy import stats
from pandas.tseries.offsets import DateOffset
from scipy.signal import detrend
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

def eu_tab_HU():
    crops = ['maize', 'spring_barley', 'winter_wheat']

    for crop in crops[1:]:
        orig = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_FRA.csv', index_col=0)
        new_country = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\HU_{crop}.csv', index_col=0)

        for ind in new_country.index:
            if ind in orig.index:
                orig.loc[ind, new_country.columns] = new_country.loc[ind, :]

        orig.to_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_HU.csv')

def eu_tab_AT():
    crops = ['maize', 'spring_barley', 'winter_wheat']
    crops_at = {'maize': 'Maize', 'spring_barley': 'Spring barley', 'winter_wheat': 'Winter wheat'}

    for crop in crops[:]:
        orig = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_HU.csv', index_col=0)
        new_country = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\crop_yields_Austria.csv',
                                  index_col=0, sep=';', decimal=',')
        crop_at = new_country.iloc[np.where(new_country.Crop_type==crops_at[crop])[0],1:]
        crop_at.index = ['AT'+str(a) for a in crop_at.index]
        for ind in crop_at.index:
            if ind in orig.index:
                orig.loc[ind, crop_at.columns] = crop_at.loc[ind, :]/10

        orig.to_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_AT.csv')

def eu_tab_CZ():
    crops = ['spring_barley', 'winter_wheat']

    for crop in crops[1:]:
        orig = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_AT.csv', index_col=0)
        new_country = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\CZ_yields_NUTS4_2000_2022.csv',
                                  index_col=0, decimal=',')
        nuts = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\NUTS_data\czr_nuts4_names_ww_2016.shp')

        name2id = pd.read_csv(r'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\cz_converter_sort.csv', index_col=0)
        name2id = {a: b for a, b in zip(name2id.index, name2id.nut_id)}
        #
        crop_df = new_country.loc[:,['NAME', 'YEAR', crop]]
        crop_df = crop_df.iloc[np.where(crop_df.NAME != 'Jablonec nad Nisou')[0], :]
        crop_df = crop_df.iloc[np.where(crop_df.NAME != 'Náchod')[0], :]
        crop_df = crop_df.iloc[np.where(crop_df.YEAR != 2023)[0], :]
        crop_df.NAME = [name2id[a][:9] for a in crop_df.NAME]
        crop_df.index = range(len(crop_df.index))
        print(crop_df)
        for i in crop_df.index:
            ind = crop_df.NAME[i]
            if ind in orig.index:
                orig.loc[ind, str(crop_df.YEAR[i])] = crop_df.iloc[i, -1]

        orig.to_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_CZ.csv')

def eu_tab_UA():
    crops = ['maize', 'spring_barley', 'winter_wheat']

    for crop in crops:
        orig = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_CZ.csv', index_col=0)
        new_country = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\UA_{crop}.csv', index_col=1).iloc[:,1:]

        for ind in new_country.index:
            if ind in orig.index:
                orig.loc[ind, new_country.columns] = new_country.loc[ind, :]/10

        orig.to_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}.csv')

def final_cleaning():
    #dropna(1); replace 0, -999 with nan, other way around, remove regions with less than 5 values, round to 2 decimals
    crops = ['maize', 'spring_barley', 'winter_wheat']
    for crop in crops:
        df = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}.csv', index_col=0)
        df_new = pd.DataFrame(np.where(df<=0, np.nan, df))
        df_new.index = df.index
        df_new.columns = df.columns
        df_new = df_new.dropna(axis=0, how='all')
        df_new = df_new.round(2)

        df_new.to_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_clean.csv')

        df_detrend = df_new.copy()
        for i in range(len(df_detrend.index)):
            y = np.array(df_new.iloc[i, :])
            x = np.array([int(a) for a in df_new.columns])
            not_nan_ind = ~np.isnan(y)
            m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind], y[not_nan_ind])
            detrend_y = y - (m * x + b)
            df_detrend.iloc[i, :] = detrend_y

        df_detrend = df_detrend.round(2)
        df_detrend.to_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{crop}_clean_det_anom.csv')

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
    path = r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\2007_2022_Ukraine.csv'
    war_map = pd.read_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\war_map.csv')
    war_map_d = {a: b for a, b in zip(war_map.Regions, war_map.Region_ID)}
    file = pd.read_csv(path, sep=';', decimal=',', index_col=0, na_values="none")/10
    file = file.round(2)
    file.index = [war_map_d[a] for a in file.index]

    crops = np.unique([a[5:] for a in file.columns[1:]])
    for crop in crops:
        inds = [a for a in file.columns if a.endswith((crop, 'Regions', 'Region_ID'))]
        this_crop = file.loc[:,inds]
        this_crop.columns = [a.replace(f'_{crop}','') for a in this_crop.columns]
        this_crop.to_csv(rf'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\UA_2007_{crop}.csv')

def ua2m(temp_res):
    for crop in ["maize", "spring_barley", "winter_wheat"]:
        path = rf'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\UA_2007_{crop}.csv'
        file = pd.read_csv(path, index_col=0)

        df_detrend = file.copy()
        for i in range(len(df_detrend.index)):
            y = np.array(file.iloc[i, :])
            x = np.array([int(a) for a in file.columns])
            not_nan_ind = ~np.isnan(y)
            m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind], y[not_nan_ind])
            detrend_y = y - (m * x + b)
            df_detrend.iloc[i, :] = detrend_y

        df_detrend = df_detrend.round(2)
        yield_data = pd.read_csv(f'Data/SC2/{temp_res}/final/{crop}_all.csv', index_col=0)
        yield_data_abs = pd.read_csv(f'Data/SC2/{temp_res}/final/{crop}_all_abs.csv', index_col=0)
        # print(file_flat.shape[0])
        # for i in range(file_flat.shape[0]):
        # for i in [1]:
        for year in df_detrend.columns:
            for reg in df_detrend.index:
                val = file.loc[reg, year]
                val_det = df_detrend.loc[reg, year]
                # print(np.where((yield_data.field_id==reg) & (yield_data.c_year==int(year))))
                yield_data.loc[(yield_data.field_id==reg) & (yield_data.c_year==int(year)), 'yield_anom'] = val_det
                yield_data_abs.loc[(yield_data_abs.field_id == reg) & (yield_data_abs.c_year == int(year)), 'yield_anom'] = val

        yield_data = yield_data.dropna(axis=0)
        yield_data_abs = yield_data_abs.dropna(axis=0)

        yield_data.to_csv(f'Data/SC2/{temp_res}/final/{crop}_all_fin.csv')
        yield_data_abs.to_csv(f'Data/SC2/{temp_res}/final/{crop}_all_abs_fin.csv')

def detrend_timeseries(df):
    # Ensure the DataFrame has 'year' and 'value' columns:
    # df = pd.DataFrame({'year': [...], 'value': [...]})

    # Set year as index
    df_indexed = df.set_index('c_year')

    # Create a continuous range of years
    full_years = pd.RangeIndex(start=df_indexed.index.min(), stop=df_indexed.index.max() + 1)

    # Reindex to full range, introducing NaN for missing years
    df_full = df_indexed.reindex(full_years)

    # Interpolate missing values
    df_full['yield_anom'] = df_full['yield_anom'].interpolate(method='linear')

    # Detrend values
    detrended_values = detrend(df_full['yield_anom'].values)

    # Compile result for original years only
    result = pd.DataFrame({'c_year': df_full.index, 'yield_anom': detrended_values+np.mean(df_indexed.yield_anom)})
    result = result.round(2)
    result = result[result['c_year'].isin(df_indexed.index)]

    return result.reset_index(drop=True)

def check_data():
    #, 'winter_wheat', 'maize'
    for crop in ['spring_barley', 'winter_wheat', 'maize']:
        path = f'Data/SC2/2W/final/{crop}_all_abs_fin.csv'
        file = pd.read_csv(path, index_col=0)
        file.index = range(len(file.index))
        for i in range(1,9):
            file.loc[:, f'year_LT{i}']=file.c_year
    # region = 'AT104'
        for region in np.unique(file.field_id):
            reg_file = file.iloc[np.where(file.field_id==region)[0], :]
            det_rg = detrend_timeseries(reg_file.iloc[:, 1:3])
            file.iloc[reg_file.index, 2] = det_rg.yield_anom
        file.to_csv(f'Data/SC2/2W/final/{crop}_all_abs_fin_year_det.csv')
    # reg_file_det = file.iloc[np.where(file.field_id == region)[0], :]
    # plt.scatter(reg_file.c_year, reg_file.yield_anom)
    # plt.scatter(reg_file_det.c_year, reg_file_det.yield_anom)
    # plt.show()

def shape_nuts322():
    # Load your NUTS 3 shapefile
    gdf_nuts3 = gpd.read_file(r"M:\Projects\YIPEEO\07_data\Crop yield\All_NUTS.shp")

    # Extract NUTS 2 codes from NUTS 3 codes by truncating or mapping
    # Usually NUTS 3 code has 5 characters; NUTS 2 is the first 4 characters
    gdf_nuts3['NUTS2_ID'] = gdf_nuts3['NUTS_ID'].str[:4]

    # Dissolve geometries by NUTS2_ID to aggregate NUTS 3 areas into NUTS 2 areas
    gdf_nuts2 = gdf_nuts3.dissolve(by='NUTS2_ID')

    # Optionally, reset index and save
    gdf_nuts2 = gdf_nuts2.reset_index()
    gdf_nuts2.to_file(r"M:\Projects\YIPEEO\07_data\Crop yield\All_NUTS4.shp")

    print("Conversion from NUTS 3 to NUTS 2 shapefile completed.")

def merge_shapes():
    # Load the original NUTS 2 shapefile (converted from NUTS 3)
    gdf_old = gpd.read_file(r"M:\Projects\YIPEEO\07_data\Crop yield\All_NUTS4.shp")

    # Load the new NUTS 2 shapefile from different countries
    gdf_new = gpd.read_file(r"M:\Projects\YIPEEO\07_data\Crop yield\Database\regional\NUTS2.shp")

    # Remove duplicates in the new file based on the NUTS2 identifier column, e.g., 'NUTS2_ID' or whatever column has the region ID
    gdf_new = gdf_new.drop_duplicates(subset='nut_id')

    # To keep only regions from the new file in case of overlap, filter old file to exclude regions that appear in the new
    overlap_ids = gdf_new['nut_id'].unique()
    gdf_old_filtered = gdf_old[~gdf_old['NUTS2_ID'].isin(overlap_ids)]

    # Now concatenate the filtered old and new GeoDataFrames
    gdf_merged = gpd.GeoDataFrame(pd.concat([gdf_old_filtered, gdf_new], ignore_index=True))

    # Optionally reset index and save the merged shapefile
    gdf_merged = gdf_merged.reset_index(drop=True)
    gdf_merged.to_file(r"M:\Projects\YIPEEO\07_data\Crop yield\All_NUTS2_fin.shp")

    print("Merging completed, with duplicates removed and new file regions prioritized in overlaps.")

def get_yields():
    crop = 'maize'
    gdf_nuts2 = gpd.read_file(r"M:\Projects\YIPEEO\07_data\Crop yield\All_NUTS2_fin.shp")

    # Load your maize CSV with NUTS3 yield data
    df_maize = pd.read_csv(f'Data/SC2/2W/final/{crop}_all_abs_fin.csv')

    # Make sure columns are as expected
    # df_maize.columns should include 'field_id', 'yield_anom', 'c_year'

    # Aggregate the maize yield data from NUTS3 to NUTS2 level:
    # Get NUTS2_ID by truncating the NUTS3 code (first 4 characters)
    df_maize['NUTS2_ID'] = df_maize['field_id'].str[:4]

    # Group by NUTS2_ID and year, calculate mean yield_anom
    df_agg = df_maize.groupby(['NUTS2_ID', 'c_year'])['yield_anom'].mean().reset_index()

    # Pivot df_agg to have one column per year with yield values
    df_pivot = df_agg.pivot(index='NUTS2_ID', columns='c_year', values='yield_anom').reset_index()

    # Rename year columns to include prefix
    df_pivot.columns = ['NUTS2_ID'] + [f'{int(year)}' for year in df_pivot.columns[1:]]

    # Merge the aggregated yield data to the NUTS2 shapefile GeoDataFrame
    gdf_merged = gdf_nuts2[['NUTS2_ID', 'geometry']].merge(df_pivot, on='NUTS2_ID', how='left')

    # Save the merged shapefile with yield columns for each year
    gdf_merged.to_file(rf"M:\Projects\YIPEEO\07_data\Crop yield\All_NUTS2_{crop}.shp")

    print("Shapefile with NUTS2 yield data for each year saved as 'nuts2_with_maize_yield.shp'")

def nl_yield():
    path = r'M:\Projects\YIPEEO\07_data\Crop yield\Netherlands\Yield_NUTS2\NUTS2.csv'
    file = pd.read_csv(path, sep=';')
    file.loc[:, 'year'] = [int(a[:4]) for a in file.Periods]
    file = file.loc[:, ['ArableCrops', 'GrossYieldPerHa_3', 'year', 'Regions']]
    # for crop in file.ArableCrops.unique():
    #     subfile = file.loc[file.ArableCrops == crop]
    df_wide = file.pivot(index=['ArableCrops', 'Regions'], columns='year', values='GrossYieldPerHa_3')
    df_wide.to_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Netherlands\Yield_NUTS2\NUTS2_ref.csv')

def check_yield():
    path = r'M:\Projects\YIPEEO\07_data\Crop yield\NUTS3_all_countries.csv'
    df = pd.read_csv(path, sep=',')
    df = df.replace(0, np.nan)

    df['Crop_type'] = [a.capitalize() for a in df['Crop_type']]

    df['NUTS2_ID'] = df['Region_id'].str[:4]

    for crop in ['Maize', 'Spring barley', 'Winter wheat']:
        subset = df[df['Crop_type'] == crop].iloc[:, 3:]
        nuts2 = subset.groupby(['NUTS2_ID']).mean().round(2)
        nuts2 = nuts2.drop(index=['PLZZ'])

        nuts2.to_csv(rf'M:\Projects\YIPEEO\07_data\Crop yield\{crop}_NUTS2.csv')

def csv2pred(crop, step_size='2W'):

    pred_file_path = r'M:\Projects\YIPEEO\07_data\Predictors\SC2'
    pred_files = [a for a in os.listdir(pred_file_path) if a.endswith('.csv')]
    preds = [a.split('.')[0] for a in pred_files]

    countries_eu = ['AT', 'CZ', 'DE', 'FR', 'HR', 'HU', 'PL', 'SI', 'SK', 'NL']

    harvest_date_maize = {a: [9, 15] for a in countries_eu}
    harvest_date_maize['UA'] = [8, 31]

    harvest_date_barley = {a: [7, 31] for a in countries_eu}
    harvest_date_barley['UA'] = [6, 30]

    harvest_date_wheat = {'AT': [6, 16], 'CZ': [6, 30], 'DE': [6, 30], 'FR': [6, 30], 'HR': [6, 16],
                          'HU': [6, 16], 'PL': [6, 30], 'SI': [6, 30], 'SK': [6, 30], 'UA': [7, 16], 'NL': [6, 30]}

    harvest_date_pc = {'Winter_wheat': harvest_date_wheat, 'Maize': harvest_date_maize,
                            'Spring_barley': harvest_date_barley}

    harvest_date = harvest_date_pc[crop]

    if step_size=='2W':
        lead_times = ['_LT8', '_LT7', '_LT6', '_LT5', '_LT4', '_LT3', '_LT2', '_LT1']
    elif step_size=='ME':
        lead_times = ['_LT4', '_LT3', '_LT2', '_LT1']
    else:
        raise ValueError('Step size must be 2W or M')

    yield_vals = pd.read_csv(fr'M:\Projects\YIPEEO\07_data\Crop yield\Regional_final\{crop}_NUTS2_det.csv', index_col=0)
    col_names = [[a + b for a in preds] for b in lead_times]
    col_names = list(itertools.chain(*col_names))

    pipeline_df = pd.DataFrame(data=None, columns=['field_id', 'c_year', 'yield'] + col_names,
                               index=range(yield_vals.shape[0]*yield_vals.shape[1]))
    regs = yield_vals.index
    years = yield_vals.columns
    for pred, pred_file in zip(preds, pred_files):
        ind_cols = [i for i, predictor in enumerate(pipeline_df.columns) if predictor.startswith(pred)]
        pred_csv = pd.read_csv(f'{pred_file_path}/{pred_file}', index_col=0)
        for r, reg in enumerate(regs):
            nut_file = pred_csv.iloc[: , np.where(pred_csv.columns==reg)[0]]
            if nut_file.empty:
                print(f'{reg} is empty')
                continue
            nut_file.index = pd.to_datetime(nut_file.index, format='%Y-%m-%d')

            # Sort by date
            nut_file_sorted = nut_file.sort_values('date')

            # Create a Series with 'date' as index and 'mean_evi' as values

            par_series = pd.Series(data=nut_file_sorted[reg].values, index=nut_file_sorted.index)

            for y, year in enumerate(years):
                ind_r = r + (y * len(regs))
                pipeline_df.loc[ind_r, 'field_id'] = reg
                pipeline_df.loc[ind_r, 'c_year'] = year
                pipeline_df.loc[ind_r, 'yield'] = yield_vals.loc[reg, year]

                par_series_m = par_series.resample(step_size).mean().interpolate()
                this_harvest_date = pd.to_datetime(f'{year}-{harvest_date[reg[:2]][0]}-{harvest_date[reg[:2]][1]}')  # harvest date -2 days to make sure harvested field is not included
                start_date = this_harvest_date - DateOffset(months=4)
                this_df = par_series_m[start_date:this_harvest_date]

                if len(this_df) >= len(lead_times):
                    this_df = this_df[-len(lead_times):]

                # print(pipeline_df.loc[len(pipeline_df), :])
                new_vals = list(this_df.values)
                pipeline_df.iloc[ind_r, ind_cols] = new_vals
            if (r%10==0) and (pred==preds[-1]):
                print(f'{r}/{len(regs)} done')
    pipeline_df = pipeline_df.dropna(axis=0, how='any')
    pipeline_df.to_csv(f'Data/SC2/nuts2/{crop}_{step_size}.csv')

def detrend_yield():
    crop = 'Spring_barley'
    path = rf'M:\Projects\YIPEEO\07_data\Crop yield\Regional_final\{crop}_NUTS2.csv'
    df = pd.read_csv(path, index_col=0)
    df_detrended = df.apply(detrend_nan, axis=0)
    path_out = rf'M:\Projects\YIPEEO\07_data\Crop yield\Regional_final\{crop}_NUTS2_det.csv'
    df_detrended.to_csv(path_out)
    # i = 30
    # plt.plot(df.iloc[i,:])
    # plt.plot(df_detrended.iloc[i, :], color='green')
    # plt.hlines(y=np.nanmean(df.iloc[i,:]), xmin=0, xmax=20, color='r')
    # plt.show()


def detrend_nan(series):
    arr = series.values
    not_nan = ~np.isnan(arr)
    detrended = np.full_like(arr, np.nan, dtype=np.float64)
    if np.sum(not_nan) > 1:  # Need at least 2 points to detrend
        detrended[not_nan] = detrend(arr[not_nan])+np.nanmean(arr)
    return pd.Series(detrended, index=series.index)



if __name__ == '__main__':
    # pd.set_option('display.max_rows', None)
    start_pro = datetime.datetime.now()
    for crop in ['Spring_barley', 'Winter_wheat', 'Maize']:
        for ss in ['2W', 'ME']:
            csv2pred(crop=crop, step_size=ss)

    # detrend_yield()
    # get_yields()
    # check_yield()
    # ua2m(temp_res='2W')
    # check_data()
    # shape_nuts322()
    # merge_shapes()
    # final_cleaning()
    # a = france_data(r'D:\DATA\yipeeo\NUTS_data')

    # b = austria_data()
    # c = crop_data(country='austria')
    # c.add_yield()
    # ukr_reg()
    # a.select_country()
    # pd.set_option('display.max_columns', None)
    # b.add_yield()
    # ukr2shape()
    end_pro = datetime.datetime.now()
    print(f'calculation stopped at {end_pro} and took {end_pro - start_pro}')