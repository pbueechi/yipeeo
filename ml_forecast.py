import os
import itertools
import csv
import statistics
import warnings
import seaborn
import matplotlib
import math
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import tensorflow as tf
from keras.regularizers import l2
from cmcrameri import cm as cmr
from pandas.tseries.offsets import DateOffset
from glob import glob
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RandomizedSearchCV, KFold
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline

from scikeras.wrappers import KerasRegressor
from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam, SGD


#ToDo: remove farms and just use ukr_horod as country name
class nc2table:
    """
    The Crop yield data and predictors are saved as nc files. This class extracts the predictors for the four months
    before harvest either in biweekly or monthly timesteps and saves them as csv files.
    """
    def __init__(self, country, crop, farm=None):
        self.country = country
        self.crop = crop
        self.farm = farm
        # Harvest dates in CZR around 25 July for winter wheat, 10 Oct for Maize, and 20 Jul Spring barley
        self.harvest_date = {'common winter wheat': [7,25], 'winter_wheat': [7,25], 'winter barley': [7,30], 'grain maize and corn-cob-mix': [10,10], 'maize':[10,10], 'spring_barley': [7,20]}   #ToDo: dont hardcode harvest dates
        # self.crop_data = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale.shp')
        self.lead_times = ['_LT4','_LT3','_LT2','_LT1']
        # Harvest dates are taken from JRC for EU countries and USDA for Ukraine
        # https://agri4cast.jrc.ec.europa.eu/dataportal/
        # https://ipad.fas.usda.gov/rssiws/al/crop_calendar/umb.aspx
        self.harvest_date_wheat = {'AT': [6, 31], 'CZ': [7, 16], 'DE': [7, 16], 'FR': [7, 11], 'HR': [6, 31],
                                   'HU': [6, 31], 'PL': [7, 16], 'SI': [7, 16], 'SK': [7, 16], 'UA': [7, 16]}
        self.harvest_date_maize = {'AT': [9, 30], 'CZ': [9, 30], 'DE': [9, 30], 'FR': [9, 30], 'HR': [9, 30],
                                   'HU': [9, 30], 'PL': [9, 30], 'SI': [9, 30], 'SK': [9, 30], 'UA': [9, 30]}
        if country=='ES':
            lleida_data = gpd.read_file(r'/home/mformane/shares/climers/Projects/YIPEEO/07_data/Crop yield/Database/field_scale_lleida.shp')
            madrid_data = gpd.read_file(r'/home/mformane/shares/climers/Projects/YIPEEO/07_data/Crop yield/Database/field_scale_madrid.shp')
            self.crop_data = pd.concat([lleida_data, madrid_data], ignore_index=True)

    def resample_s1(self, pred_file_path, temp_step='M'):
        """
        Extracts Sentinel-1 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['sig0_vv_mean_daily', 'sig0_vh_mean_daily', 'sig0_cr_mean_daily', 'sig40_vv_mean_daily', 'sig40_vh_mean_daily', 'sig40_cr_mean_daily']
        inds = np.where((self.crop_data.country_co==self.country.lower())&(self.crop_data.crop_type==self.crop)&(self.crop_data.c_year>2015))[0]
        if self.farm:
            inds = np.where((self.crop_data.farm_code == self.farm) & (self.crop_data.crop_type == self.crop) & (
                        self.crop_data.c_year > 2015))[0]
        #pred_file_path = r'D:\DATA\yipeeo\Predictors\S1\daily'
        self.crop_data = self.crop_data.iloc[inds,:]
        print(self.crop_data.head())
        pipeline_df = self.crop_data.iloc[:,[1,5,10,17]]
        pipeline_df.index = range(len(pipeline_df.index))
        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))
        pipeline_df.loc[:,col_names] = np.nan

        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id,pipeline_df.c_year):
            #field = field.split('_')[-1]
            #path_nc = os.path.join(pred_file_path, f'{self.country}_{field}_{year}_cleaned_cr_agg_daily.nc')
            fieldname = f'{field}_{year}_cleaned_cr_agg_daily.nc'
            path_nc = os.path.join(pred_file_path, fieldname)

            if not os.path.exists(path_nc):
                print(f'file {fieldname} does not exist')
                continue
            else:
                print(f'file {fieldname} exists!')
            s2 = xr.open_dataset(path_nc)
            for param in params:
                this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                vals = list(itertools.chain(*s2[param].values))
                ndvi = pd.Series(vals, s2.time)
                ndvi_m = ndvi.resample(temp_step).mean()
                this_harvest_date = pd.to_datetime(f'{year}-{self.harvest_date[self.crop][0]}-{self.harvest_date[self.crop][1]-2}') #harvest date -2 days to make sure harvested field is not included
                start_date = this_harvest_date - DateOffset(months=4)
                this_df = ndvi_m[start_date:this_harvest_date]
                if len(this_df)>=len(self.lead_times):
                    this_df = this_df[:len(self.lead_times)]
                pipeline_df.loc[df_ind, this_cols] = this_df.values

        if not os.path.exists(f'Data/{temp_step}/{self.country}'):
            os.makedirs(f'Data/{temp_step}/{self.country}')
        if self.farm:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}_s1.csv')
        else:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_s1.csv')

    def resample_s2(self, pred_file_path, temp_step='M'):
        """
        Extracts Sentinel-2 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['ndvi', 'evi', 'ndwi', 'nmdi']
        inds = np.where((self.crop_data.country_co==self.country.lower())&(self.crop_data.crop_type==self.crop)&
                        (self.crop_data.c_year>2015))[0]
        if self.farm:
            inds = np.where((self.crop_data.farm_code == self.farm) & (self.crop_data.crop_type == self.crop) & (
                        self.crop_data.c_year > 2015))[0]
        #pred_file_path = rf'D:\DATA\yipeeo\Predictors\S2_L2A\{self.country}\nc'
        if isinstance(pred_file_path, list):
            pathlist = True
        else:
            pathlist = False
        self.crop_data = self.crop_data.iloc[inds,:]

        pipeline_df = self.crop_data.iloc[:,[1,5,10,17]]
        pipeline_df.index = range(len(pipeline_df.index))

        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))+['date_last_obs']
        pipeline_df.loc[:,col_names] = np.nan

        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id,pipeline_df.c_year):
            for param in params:
                this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                if pathlist:
                    for pred_path in pred_file_path:
                        path_nc = os.path.join(pred_path, f'{field}.nc')
                        if os.path.exists(path_nc):
                            break
                else:
                    path_nc = os.path.join(pred_file_path, f'{field}.nc')

                s2 = xr.open_dataset(path_nc)
                ndvi = s2[param].to_series()
                ndvi_m = ndvi.resample(temp_step).mean()
                this_harvest_date = pd.to_datetime(f'{year}-{self.harvest_date[self.crop][0]}-{self.harvest_date[self.crop][1]-1}') #harvest date -2 days to make sure harvested field is not included
                this_df = ndvi_m[:this_harvest_date].tail(len(self.lead_times)+1)
                if np.sum(np.isnan(this_df.values))>0:
                    this_df = this_df.interpolate()
                this_df = this_df[1:]
                pipeline_df.loc[df_ind, this_cols] = this_df.values
            pipeline_df.loc[df_ind, 'date_last_obs'] = this_df.index[-1]

        if not os.path.exists(f'Data/{temp_step}/{self.country}'):
            os.makedirs(f'Data/{temp_step}/{self.country}')
        if self.farm:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}_s2.csv')
        else:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_s2.csv')

    def resample_ecostress(self, temp_step='M'):
        """
        Extracts Sentinel-2 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']

        params = ['ECO2LSTE_median']

        if self.farm:
            inds = np.where((self.crop_data.farm_code == self.farm) & (self.crop_data.crop_type == self.crop) & (
                        self.crop_data.c_year > 2018))[0]
        else:
            inds = np.where((self.crop_data.country_co == self.country) & (self.crop_data.crop_type == self.crop) & (
                        self.crop_data.c_year > 2018))[0]
        pred_file_path = rf'D:\DATA\yipeeo\Predictors\ECOSTRESS\{self.country}\nc'
        self.crop_data = self.crop_data.iloc[inds,:]
        pipeline_df = self.crop_data.iloc[:,[1,5,10]]
        pipeline_df.index = range(len(pipeline_df.index))

        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))+['date_last_obs']
        pipeline_df.loc[:,col_names] = np.nan

        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id,pipeline_df.c_year):
            for param in params:
                this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                s2 = xr.open_dataset(os.path.join(pred_file_path, f'{field}.nc'))
                ndvi = s2[param].to_series()
                ndvi = ndvi.replace(-9999, np.nan)
                ndvi_m = ndvi.resample(temp_step).mean()
                this_harvest_date = pd.to_datetime(f'{year}-{self.harvest_date[self.crop][0]}-{self.harvest_date[self.crop][1]-1}') #harvest date -1 day to make sure harvested field is not included
                this_df = ndvi_m[:this_harvest_date].tail(len(self.lead_times)+1)
                if np.sum(np.isnan(this_df.values))>0:
                    this_df = this_df.interpolate()
                this_df = this_df[1:]
                pipeline_df.loc[df_ind, this_cols] = this_df.values
            pipeline_df.loc[df_ind, 'date_last_obs'] = this_df.index[-1]

        if not os.path.exists(f'Data/{temp_step}/{self.country}'):
            os.makedirs(f'Data/{temp_step}/{self.country}')
        if self.farm:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}_eco.csv')
        else:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_eco.csv')

    def resample_era(self, temp_step='M'):
        """
        Extracts Sentinel-1 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['ETo_sum', 'P_sum', 'PET_sum', 'Rn_sum', 'T_avg', 'VWC_1_avg']
        crop_data = gpd.read_file(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional\{self.crop}.shp')
        pred_file_path = rf'D:\data-write\YIPEEO\predictors\ERA5-Land\{self.country}\nc'
        files = [a for a in os.listdir(pred_file_path) if a.endswith('.nc')]
        nuts = [a.split('.')[0] for a in files]

        years = range(2016, 2023)
        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))
        pipeline_df = pd.DataFrame(data=None, columns=['field_id', 'c_year', 'yield']+col_names, index=range(len(nuts*len(years))))
        year_col = [[a] * len(nuts) for a in list(years)]
        year_col = list(itertools.chain(*year_col))
        pipeline_df.c_year = year_col
        pipeline_df.field_id = nuts*len(years)
        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id, pipeline_df.c_year):
            path_nc = os.path.join(pred_file_path, f'{field}.nc')
            if self.country=='Austria':
                field_nuts = 'AT4'+field
            else:
                field_nuts = field
            x,y = np.where((crop_data.nut_id==field_nuts) & (crop_data.c_year==year))[0], np.where(crop_data.columns=='yi_hu_eu')[0]
            pipeline_df.loc[df_ind,'yield'] = crop_data.iloc[x,y].values
            s2 = xr.open_dataset(path_nc)
            for param in params:
                this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                vals = s2[param].values
                ndvi = pd.Series(vals, s2.time)
                ndvi_m = ndvi.resample(temp_step).mean()
                this_harvest_date = pd.to_datetime(f'{year}-{self.harvest_date[self.crop][0]}-{self.harvest_date[self.crop][1]-2}') #harvest date -2 days to make sure harvested field is not included
                start_date = this_harvest_date - DateOffset(months=4)
                this_df = ndvi_m[start_date:this_harvest_date]
                if len(this_df)>=len(self.lead_times):
                    this_df = this_df[:len(self.lead_times)]
                pipeline_df.loc[df_ind, this_cols] = this_df.values

        pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_era.csv')

    def resample_era_EU(self, temp_step='M'):
        """
        Extracts Sentinel-1 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr']
        crop_data = gpd.read_file(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional\{self.crop}.shp')
        pred_file_path = rf'D:\data-write\YIPEEO\predictors\ERA5-Land\{self.country}\nc'
        files = [a for a in os.listdir(pred_file_path) if a.endswith('.nc')]
        nuts = [a.split('.')[0] for a in files]

        years = range(2016, 2023)
        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))
        pipeline_df = pd.DataFrame(data=None, columns=['field_id', 'c_year', 'yield']+col_names, index=range(len(nuts*len(years))))
        year_col = [[a] * len(nuts) for a in list(years)]
        year_col = list(itertools.chain(*year_col))
        pipeline_df.c_year = year_col
        pipeline_df.field_id = nuts*len(years)
        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id, pipeline_df.c_year):
            path_nc = os.path.join(pred_file_path, f'{field}.nc')
            if self.country=='Austria':
                field_nuts = 'AT4'+field
            else:
                field_nuts = field
            x,y = np.where((crop_data.nut_id==field_nuts) & (crop_data.c_year==year))[0], np.where(crop_data.columns=='yi_hu_eu')[0]
            pipeline_df.loc[df_ind,'yield'] = crop_data.iloc[x,y].values
            s2 = xr.open_dataset(path_nc)
            for param in params:
                this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                vals = s2[param].values
                ndvi = pd.Series(vals, s2.time)
                ndvi_m = ndvi.resample(temp_step).mean()
                this_harvest_date = pd.to_datetime(f'{year}-{self.harvest_date[self.crop][0]}-{self.harvest_date[self.crop][1]-2}') #harvest date -2 days to make sure harvested field is not included
                start_date = this_harvest_date - DateOffset(months=4)
                this_df = ndvi_m[start_date:this_harvest_date]
                if len(this_df)>=len(self.lead_times):
                    this_df = this_df[:len(self.lead_times)]
                pipeline_df.loc[df_ind, this_cols] = this_df.values

        pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_era_eu.csv')

    def resample_s2_glob(self, year, temp_step='M'):
        """
        Extracts Sentinel-1 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['evi', 'ndvi', 'nmdi', 'ndwi']
        crop_data = gpd.read_file(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional\{self.crop}.shp')

        pred_file_path = rf'D:\DATA\yipeeo\Predictors\S2_L2A\{self.country}\{self.crop}\{str(year)}\nc\cleaned\new'
        files = [a for a in os.listdir(pred_file_path) if a.endswith('.nc')]
        nuts = [a.split('.')[0] for a in files]

        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))
        pipeline_df = pd.DataFrame(data=None, columns=['field_id', 'c_year', 'yield']+col_names, index=range(len(nuts)))
        year_col = [year]*len(nuts)
        # year_col = list(itertools.chain(*year_col))
        pipeline_df.c_year = year_col
        pipeline_df.field_id = nuts

        for df_ind, field in zip(pipeline_df.index, pipeline_df.field_id):
            path_nc = os.path.join(pred_file_path, f'{field}.nc')
            if self.country=='Austria':
                field_nuts = 'AT4'+field
            else:
                field_nuts = field
            x,y = np.where((crop_data.nut_id==field_nuts) & (crop_data.c_year==year))[0], np.where(crop_data.columns=='yi_hu_eu')[0]
            pipeline_df.loc[df_ind,'yield'] = crop_data.iloc[x,y].values
            s2 = xr.open_dataset(path_nc)
            for param in params:
                this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                vals = s2[param].values
                ndvi = pd.Series(vals, s2.time)
                ndvi_m = ndvi.resample(temp_step).mean()
                this_harvest_date = pd.to_datetime(f'{year}-{self.harvest_date[self.crop][0]}-{self.harvest_date[self.crop][1]-2}') #harvest date -2 days to make sure harvested field is not included
                start_date = this_harvest_date - DateOffset(months=4)
                this_df = ndvi_m[start_date:this_harvest_date]
                if len(this_df)>=len(self.lead_times):
                    this_df = this_df[:len(self.lead_times)]
                pipeline_df.loc[df_ind, this_cols] = this_df.values

        pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_s2_{year}.csv')

    def resample_s1_glob(self, temp_step='M'):
        """
        Extracts Sentinel-1 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['sig0_vv_mean_daily', 'sig0_vh_mean_daily', 'sig0_cr_mean_daily', 'sig40_vv_mean_daily', 'sig40_vh_mean_daily', 'sig40_cr_mean_daily']

        pred_file_path = rf'D:\data-write\YIPEEO\predictors\S1\{self.country}\{self.crop}'
        crop_data = gpd.read_file(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional\{self.crop}.shp')
        files = [a for a in os.listdir(pred_file_path) if a.endswith('.nc')]
        nuts = [a.split('.')[0] for a in files]
        years = [a for a in range(2016, 2023)]

        col_names = [[a + b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))
        pipeline_df = pd.DataFrame(data=None, columns=['field_id', 'c_year', 'yield'] + col_names, index=range(len(nuts)*len(years)))
        pipeline_df.field_id = list(nuts) * len(years)
        year_col = [[year] * len(nuts) for year in years]
        year_col = list(itertools.chain(*year_col))
        pipeline_df.c_year = year_col

        for ind in pipeline_df.index:
            field = pipeline_df.iloc[ind, 0]
            year = pipeline_df.iloc[ind, 1]
            path_nc = os.path.join(pred_file_path, f'{field}.nc')
            if self.country=='Austria':
                field_nuts = 'AT4'+field
            else:
                field_nuts = field
            x,y = np.where((crop_data.nut_id==field_nuts) & (crop_data.c_year==year))[0], np.where(crop_data.columns=='yi_hu_eu')[0]
            pipeline_df.loc[ind,'yield'] = crop_data.iloc[x,y].values

            s2 = xr.open_dataset(path_nc)
            for param in params:
                this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                vals = list(s2[param].values)
                ndvi = pd.Series(vals, s2.time)
                ndvi_m = ndvi.resample(temp_step).mean()
                this_harvest_date = pd.to_datetime(f'{year}-{self.harvest_date[self.crop][0]}-{self.harvest_date[self.crop][1]-2}') #harvest date -2 days to make sure harvested field is not included
                start_date = this_harvest_date - DateOffset(months=4)
                this_df = ndvi_m[start_date:this_harvest_date]
                if len(this_df)>=len(self.lead_times):
                    this_df = this_df[:len(self.lead_times)]
                pipeline_df.loc[ind, this_cols] = this_df.values
        pipeline_df = pipeline_df.dropna(axis=0, how='all', subset=pipeline_df.columns[3:])
        pipeline_df.index = range(pipeline_df.shape[0])
        if not os.path.exists(f'Data/{temp_step}/{self.country}'):
            os.makedirs(f'Data/{temp_step}/{self.country}')
        pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_s1.csv')

    def merge_s2(self):
        csv_path = f'Data/M/{self.country}'
        files = os.path.join(csv_path, f"{self.crop}_s2_*.csv")
        files = glob(files)
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df.to_csv(os.path.join(csv_path, f"{self.crop}_s2.csv"), index=0)

    def previous_crop(self, temp_step='M'):
        print(self.crop_data)
        inds = np.where((self.crop_data.country_co==self.country)&(self.crop_data.crop_type==self.crop)&(self.crop_data.c_year>2018))[0]
        self.crop_data = self.crop_data.iloc[inds,:]

        pipeline_df = self.crop_data.iloc[:,[1,5,10]]
        pipeline_df.index = range(len(pipeline_df.index))

        pipeline_df.loc[:,'prev_year_crop'] = np.nan
        print(pipeline_df)
        for i in range(len(pipeline_df.index)):
            ind = np.where((self.crop_data.field_id == pipeline_df.field_id[i]) & (self.crop_data.c_year == pipeline_df.c_year[i]-1))[0]
            value = self.crop_data.iloc[ind, 6]
            if len(value)==0:
                value = 'unknown'
            pipeline_df.iloc[i, -1] = value

        if not os.path.exists(f'Data/{temp_step}/{self.country}'):
            os.makedirs(f'Data/{temp_step}/{self.country}')
        if self.farm:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}_prevcrop.csv')
        else:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_prevcrop.csv')

    def merge_s1_s2(self,temp_step='M'):
        """
        :param temp_res: str either M or 2W for aggregating the data monthly or Biweekly
        :return: merges all predictor csv file to one single csv file. Needs to be used if both S-1 and S-2 data should
        bes used as predictors at the same time
        """
        if self.farm:
            path = f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}*.csv'
        else:
            path = f'Data/{temp_step}/{self.country}/{self.crop}*.csv'
        files = [a for a in glob(path) if a.endswith(('s1.csv', 's2.csv'))]
        print(files)
        for i, file in enumerate(files):
            if i==0:
                csv_fin = pd.read_csv(file, index_col=0)
            else:
                csv_next = pd.read_csv(file, index_col=0)
                if not (len(csv_fin.c_year)==np.sum(csv_fin.c_year==csv_next.c_year)) and (len(csv_fin.field_id)==np.sum(csv_fin.field_id==csv_next.field_id)):
                    raise ValueError('the files do not correspond')
                csv_fin = csv_fin.merge(csv_next)
        if self.farm:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}_all.csv')
        else:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_all.csv')

    def merge_by_ind(self, temp_step='M', era=False):
        path = f'Data/{temp_step}/{self.country}/{self.crop}*.csv'
        if era:
            files = [a for a in glob(path) if a.endswith(('_s1.csv', '_s2.csv', '_era.csv'))]
        else:
            files = [a for a in glob(path) if a.endswith(('_s1.csv', '_s2.csv'))]

        for i, file in enumerate(files):
            if i==0:
                csv_fin = pd.read_csv(file, index_col=0)
                csv_fin.index = [str(a)+'_'+str(b) for a,b in zip(csv_fin.field_id, csv_fin.c_year)]
            else:
                csv_next = pd.read_csv(file, index_col=0)
                csv_next.index = [str(a) + '_' + str(b) for a, b in zip(csv_next.field_id, csv_next.c_year)]
                csv_fin = pd.merge(csv_fin, csv_next.iloc[:,3:], left_index=True, right_index=True)
        if era:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_all.csv')
        else:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_s1s2.csv')

    def merge_all(self,temp_step):
        """
        :param temp_res: str either M or 2W for aggregating the data monthly or Biweekly
        :return: merges all predictor csv file to one single csv file. Needs to be used if both S-1 and S-2 data should
        bes used as predictors at the same time
        """
        if self.farm:
            path = f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}*.csv'
        else:
            path = f'Data/{temp_step}/{self.country}/{self.crop}*.csv'
        files = [a for a in glob(path) if not a.endswith('all.csv')]
        print(files)
        for i, file in enumerate(files):
            if i==0:
                csv_fin = pd.read_csv(file, index_col=0)
            else:
                csv_next = pd.read_csv(file, index_col=0)
                csv_fin = csv_fin.merge(csv_next)
        csv_fin = csv_fin.drop('date_last_obs', axis=1)
        if self.farm:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}_all_2018.csv')
        else:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_all_2018.csv')

    def clean_nuts3(self, dataset):
        csv_path = f'Data/M/{self.country}'
        file = pd.read_csv(os.path.join(csv_path, f"{self.crop}_{dataset}.csv"), index_col=0)
        reg_list = [a[:5] for a in file.field_id]
        regs = np.unique(reg_list)
        years = [a for a in range(2016,2023)]
        new_file = pd.DataFrame(index=range(len(regs)*len(years)), columns=file.columns)
        new_file.field_id = list(regs)*len(years)
        year_col = [[year]*len(regs) for year in years]
        year_col = list(itertools.chain(*year_col))
        new_file.c_year = year_col

        for i in new_file.index:
            inds = np.where((np.array(reg_list)==str(new_file.iloc[i,0])) & (file.c_year==int(new_file.iloc[i,1])))[0]
            if len(inds)>=1:
                new_file.iloc[i, 3:] = file.iloc[inds,3:].mean(axis=0)
                new_file.iloc[i, 2] = file.iloc[inds[0],2]
        new_file.to_csv(os.path.join(csv_path, f'{self.crop}_{dataset}_nuts3.csv'))

def merge_countries(field=False):
    if field:
        countries = ['polk', 'rost']
        crops = ['grain maize and corn-cob-mix', 'spring barley', 'common winter wheat']
        crops_new_name = ['maize', 'spring_barley', 'winter_wheat']
        crops_names = {crop_old: crop_new for crop_old, crop_new in zip(crops, crops_new_name)}
        origin_path = 'Data/M/cz'
        target_path = 'Data/M/TL'
        for crop in crops:
            files = [os.path.join(origin_path, f'{country}_{crop}_s1s2.csv') for country in countries]
            df = pd.concat(map(pd.read_csv, files), ignore_index=True)
            df = df.drop(columns=[df.columns[0], df.columns[-1]], axis=1)
            df.to_csv(os.path.join(target_path, crops_names[crop] + '_s1s2_field.csv'), index=False)
    else:
        countries = ['Austria', 'Czechia']
        crops = ['maize', 'spring_barley', 'winter_wheat']
        origin_path = 'Data/M'
        target_path = 'Data/M/TL'
        for crop in crops:
            files = [os.path.join(origin_path, country, crop+'_s1s2.csv') for country in countries]
            df = pd.concat(map(pd.read_csv, files), ignore_index=True)
            df = df.drop(columns=df.columns[0], axis=1)
            df.to_csv(os.path.join(target_path, crop+'_s1s2_regional.csv'), index=False)


class ml:
    def __init__(self, crop, country, path_out=None, temp_res='M', farm=None):
        self.crop = crop
        self.country = country
        self.farm = farm
        self.temp_res = temp_res
        self.path_out = path_out
        if path_out:
            if not os.path.exists(path_out): os.makedirs(path_out)
        self.nameconvert = {'maize':'grain maize and corn-cob-mix', 'winter_wheat':'common winter wheat', 'spring_barley':'spring barley'}

    #---------------------------------------------- Exploratory data analysis ----------------------------------------
    def calc_corrs(self, filename):
        """
        :return: calculates correlations between predictors and crop yields
        """
        #file = pd.read_csv(f'Data/M/{self.country}/{self.crop}_all_2018.csv', index_col=0)
        file = pd.read_csv(filename, index_col=0)
        predictors = list(file.columns[4:])
        if 'prev_year_crop' in predictors:
            predictors.remove('prev_year_crop')
        if 'date_last_obs' in predictors:
            predictors.remove('date_last_obs')
        crop_yield = file.loc[:, 'yield']
        index = np.unique([a[:-4] for a in predictors])
        print(index)
        lead_times = self.get_lead_times()
        forecast_month = [f'LT{i}' for i in lead_times]
        corr_df = pd.DataFrame(columns=forecast_month, index=index)
        corr_low = pd.DataFrame(columns=forecast_month, index=index)
        corr_high = pd.DataFrame(columns=forecast_month, index=index)

        for d, dep_var in enumerate(index):
            print(d, dep_var)
            for m, fc_month in enumerate(forecast_month):
                cor_file = pd.DataFrame([crop_yield, file.loc[:, f'{dep_var}_{fc_month}']]).transpose()
                cor_file = cor_file.dropna(axis=0)
                r = spearmanr(cor_file.iloc[:, 0], cor_file.iloc[:, 1])[0]

                num = len(cor_file.index)
                stderr = 1.0 / math.sqrt(num - 3)
                delta = 1.96 * stderr
                lo, hi = math.tanh(math.atanh(r) - delta), math.tanh(math.atanh(r) + delta)

                corr_df.iloc[d, m] = r
                corr_low.iloc[d, m] = lo
                corr_high.iloc[d, m] = hi

        # corr_df.to_csv(f'Results/explore_data/Spearman_{self.crop}_2018.csv')
        # corr_low.to_csv(f'Results/explore_data/Spearman_{self.crop}_2018_low.csv')
        # corr_high.to_csv(f'Results/explore_data/Spearman_{self.crop}_2018_high.csv')
        return corr_df, corr_high, corr_low

    def plot_corrs(self, filename):
        """
        :param correlation_dataframe: df in the format as returning from calculate_correlations
            considered months as columns, dataset names as index
        :param corr_low: same as correlation_dataframe, only with the low boundary of the confidence interval
        :param corr_high: equivalent to corr_low for upper boundaries

        :return: shows a figure of the correlations of each dataset to the yields
        """
        corr_df, corr_high, corr_low = self.calc_corrs(filename)
        yers = corr_high - corr_low

        independent_variables = corr_df.index
        leg_names = [a.replace('_mean_daily','').replace('_median','') for a in independent_variables]
        month = corr_df.columns
        pos = np.arange(len(month))
        bar_width = 0.08
        a = len(independent_variables)
        # colors = cm.get_cmap('tab20', a).colors
        colors = np.zeros(shape=(16, 4))
        cols = cm.get_cmap('tab20c', 20).colors
        colors[:1,:] = cols[4:5, :]
        colors[1:5, :] = cols[8:12, :]
        colors[5:8, :] = cols[:3, :]
        colors[8:11, :] = cols[16:19, :]

        posi = np.arange(-((a - 1) / 2), (a / 2), 1) * bar_width
        plt.figure(figsize=(12, 7.2))
        for l in range(len(corr_df.index)):
            corr = list(corr_df.iloc[l, :])
            yer = list(yers.iloc[l, :])
            plt.bar(pos + posi[l], corr, bar_width, color=colors[l], edgecolor='black', yerr=yer, capsize=2)
        ftsz = 18
        plt.yticks(fontsize=ftsz)
        plt.xticks(pos, month, fontsize=ftsz)
        plt.xlabel('Month', fontsize=ftsz)
        plt.ylabel("Pearson's R", fontsize=ftsz)
        plt.ylim([-0.8,0.8])
        # plt.title('Correlation of the Anomalies per Month to Final Yield', fontsize=14)
        plt.legend(leg_names, loc=0, ncol=4, fontsize=16)
        plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
        # plt.show()
        if not os.path.exists(f'Figures/{self.temp_res}/Predictor_analysis/'):
            os.makedirs(f'Figures/{self.temp_res}/Predictor_analysis/')
        plt.savefig(f'Figures/{self.temp_res}/Predictor_analysis/corr_{self.crop}.png', dpi=300)
        plt.close()

    def cross_cor_predictors(self, lt=None, preds=None):
        fs = 26
        file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/rost_{self.crop}_s1s2.csv', index_col=0).iloc[:,2:-1]
        file = self.select_preds(file=file, lt=lt, preds=preds)
        ticknames = [i.replace('_mean_daily', '') for i in file.columns]
        ticknames = [i.replace('_LT1', '') for i in ticknames]
        file.columns = ticknames
        matrix = file.corr(method='pearson')
        plt.figure(figsize=(15, 15), dpi=300)
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        ax = seaborn.heatmap(matrix, annot=False, mask=mask, cmap=cmr.vik_r, vmin=-1, vmax=1)
        plt.yticks(rotation=0, fontsize=fs, weight='bold')
        plt.xticks(rotation=45, fontsize=fs, weight='bold')
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fs)
        plt.subplots_adjust(left=0.14, bottom=0.13, right=0.95, top=0.95)
        # plt.show()
        if lt:
            plt.savefig(f'Figures/Predictor_analysis/cor_matrix_{lt}_{self.temp_res}_new.png', dpi=300)
        if preds:
            plt.savefig(f'Figures/Predictor_analysis/cor_matrix_fewpreds_{self.temp_res}.png', dpi=300)
        if not lt and not preds:
            plt.savefig(f'Figures/Predictor_analysis/cor_matrix_all_{self.temp_res}.png', dpi=300)

    #---------------------------------------------- Model tuning -----------------------------------------------------
    def feature_selection(self, X, y, model, n_features=10):
        # X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=5)
        predictors = X.columns
        estimator = self.get_default_regressions(model)
        selector = RFE(estimator, n_features_to_select=n_features, verbose=0)
        selector = selector.fit(X, y)

        selected_features = list(itertools.compress(predictors,selector.support_))
        X_s = X.loc[:,selected_features]

        df_rank = pd.DataFrame(index=X.columns, columns=['rank_rfe'])
        df_rank.iloc[:,0] = selector.ranking_

        return selected_features, X_s, df_rank

    def hyper_tune(self, X, y, model='RF'):
        #Check Probst et al. (2019) for settings  or:
        #https://gsverhoeven.github.io/post/random-forest-rfe_vs_tuning/
        if model=='XGB':
            params = {'max_depth': [3, 6, 10],
                      'learning_rate': [0.1, 0.3, 0.5],
                      'n_estimators': [100, 500, 1000],
                      'colsample_bytree': [0.5, 1]}

        elif model=='RF':
            params = {'max_depth': [20, None],
                      'min_samples_split': [2, 5, 10],
                      'n_estimators': [50, 100, 250],
                      'bootstrap': [True, False]}

        estimator = self.get_default_regressions(model)
        train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
        rf_random = RandomizedSearchCV(estimator=estimator, param_distributions=params, n_iter=50, cv=20, n_jobs=6,
                                       verbose=0, random_state=1)
        rf_random.fit(train_features, train_labels)
        # print(rf_random.best_params_)
        estimator.set_params(**rf_random.best_params_)
        #pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
        pipe = make_pipeline(StandardScaler(), estimator)
        scores_kf = cross_val_score(pipe, X, y, cv=20, scoring="explained_variance")
        return rf_random.best_params_

    #---------------------------------------------- Set up models ----------------------------------------------------
    def runforecast(self, lead_time, model='RF', merge_months=False, feature_select=False, hyper_tune=False, filename = None):
        X, X_test, y, y_test = self.get_train_test(lead_time=lead_time, filename=filename, merge_months=merge_months)
        estimator = self.get_default_regressions(model)
        #pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomFor)])
        pipe = make_pipeline(StandardScaler(), estimator)
        print(pipe)
        scores_kf = cross_val_score(pipe, X, y, cv=5, scoring="explained_variance")
        print(f'test perf for lead_time:{lead_time}, R^2: {np.median(scores_kf)}')
        if feature_select:
            selected_features, X_s, _ = self.feature_selection(X,y, model=model, n_features=10)
            # selected_features = ['sig0_vh_mean_daily_LT4', 'sig0_vh_mean_daily_LT3', 'sig0_cr_mean_daily_LT3',
            #                      'sig40_vh_mean_daily_LT3', 'sig40_cr_mean_daily_LT3', 'sig40_vh_mean_daily_LT2',
            #                      'sig40_vv_mean_daily_LT1', 'ndvi_LT1', 'evi_LT1', 'nmdi_LT1']
            selected_features = [p for p in X.columns if p.startswith(('sig40_vh','evi'))]

            X, X_test = X.loc[:, selected_features], X_test.loc[:, selected_features]
            estimator = self.get_default_regressions(model)
            pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
            scores_kf_fe = cross_val_score(pipe, X, y, cv=30, scoring="explained_variance")
            print(f'train perf for lead_time:{lead_time} after FE, R^2: {np.median(scores_kf_fe)}')
        else:
            selected_features = X.columns
            scores_kf_fe = []

        if hyper_tune:
            params = None
            if model == 'XGB':
                params = {'max_depth': [3, 6, 10],
                          'learning_rate': [0.1, 0.3, 0.5],
                          'n_estimators': [100, 500, 1000],
                          'colsample_bytree': [0.5, 1]}

            elif model == 'RF':
                params = {'randomforestregressor__max_depth': [20, None],
                          'randomforestregressor__min_samples_split': [2, 5, 10],
                          'randomforestregressor__n_estimators': [50, 100, 250],
                          'randomforestregressor__bootstrap': [True, False]}
            rf_random = RandomizedSearchCV(estimator=pipe, param_distributions=params, n_iter=50, cv=5, n_jobs=6,
                                           verbose=0, random_state=1)
            rf_random.fit(X, y)
            best_hp = rf_random.best_params_
            best_estimator = rf_random.best_estimator_
            scores_kf_hp = best_estimator.score(X_test, y_test)
            print(f'best hyperparameters: {best_hp}')
            print(f'test score: {scores_kf_hp}')
            #X, X_test, y, y_test = self.get_train_test(lead_time=lead_time, merge_months=merge_months)
            #hp_tuned_vals = self.hyper_tune(X, y, model=model, selected_features=X.columns)
            #estimator.set_params(**hp_tuned_vals)
            #pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
            #scores_kf_hp = cross_val_score(pipe, X, y, cv=5, scoring="explained_variance")
            #print(f'train perf for lead_time:{lead_time} after hp tuning, R^2: {np.median(scores_kf_hp)}')
        else:
            scores_kf_hp = []

        if hyper_tune and feature_select:
            hp_tuned_vals = self.hyper_tune(lead_time=lead_time, model=model, selected_features=selected_features)
            estimator.set_params(**hp_tuned_vals)
            pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
            scores_kf_fe_hp = cross_val_score(pipe, X, y, cv=30, scoring="explained_variance")
            print(f'train perf for lead_time:{lead_time} after hp tuning and FE, R^2: {np.median(scores_kf_fe_hp)}')
        else:
            scores_kf_fe_hp = []

        return scores_kf, scores_kf_fe, scores_kf_hp, scores_kf_fe_hp

    def runforecast_loocv(self, model='RF', min_obs=15, optimize=False):
        if self.farm:
            file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.farm}_{self.crop}_all.csv', index_col=0)
        else:
            file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.crop}_all.csv', index_col=0)
        file = file.dropna(axis=0)
        predictors = [p for p in file.columns[3:] if not p == 'date_last_obs']

        #lead_times = [4,3,2,1]
        lead_times = self.get_lead_times()
        cols = ['year', 'observed'] + [f'forecast_LT_{lt}' for lt in lead_times]
        results = pd.DataFrame(index=[], columns=cols)
        estimator = self.get_default_regressions(model)
        for l,lead_time in enumerate(lead_times):
            used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
            years = file.c_year
            years_obs = years.value_counts()
            years_test = years_obs[years_obs > min_obs].index

            for y,year in enumerate(years_test.values):
                year_ind = np.where(years==year)[0]
                year_ind_not = np.where(years!=year)[0]
                file_train = file.iloc[year_ind_not,:]
                file_test = file.iloc[year_ind,:]
                X_train, X_test = file_train.loc[:,used_predictors], file_test.loc[:,used_predictors]
                y_train, y_test = file_train.loc[:,'yield'], file_test.loc[:,'yield']
                if optimize:
                    selected_features,_, _ = self.feature_selection(X_train, y_train, model)
                    X_train, X_test = X_train.loc[:, selected_features], X_test.loc[:, selected_features]
                    hp_tuned_vals = self.hyper_tune(X_train, y_train, model)
                    estimator.set_params(**hp_tuned_vals)
                # hp_tuned_vals = {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.3, 'colsample_bytree': 0.5}
                pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
                pipe.fit(X_train, y_train)
                y_test_pred = pipe.predict(X_test)
                y_train_pred = pipe.predict(X_train)
                if l==0:
                    dict = {'year': [year]*len(y_test),
                            'observed': y_test,
                            f'forecast_LT_{lead_time}': y_test_pred
                            }

                    df = pd.DataFrame(dict)
                    results = pd.concat([results, df])
                else:
                    results.loc[X_test.index,f'forecast_LT_{lead_time}'] = y_test_pred
        if self.farm:
            results.to_csv(f'Results/forecasts/{self.country}_{self.farm}_{self.crop}_{model}_loocv_opt={optimize}.csv')
        else:
            results.to_csv(f'Results/forecasts/{self.country}_{self.crop}_{model}_loocv_opt={optimize}.csv')

    def runforecast_country(self, model='RF', optimize=False):
        file_train = pd.read_csv(f'Data/M/cz/rost_{self.crop}_all.csv', index_col=0)
        file_test = pd.read_csv(f'Data/M/{self.country}/polk_{self.crop}_all.csv', index_col=0)
        file_train = file_train.dropna(axis=0)
        file_test = file_test.dropna(axis=0)
        predictors = [p for p in file_train.columns[3:] if not p == 'date_last_obs']
        #lead_times = [4,3,2,1]
        lead_times = self.get_lead_times()
        cols = ['year', 'observed'] + [f'forecast_LT_{lt}' for lt in lead_times]
        results = pd.DataFrame(index=[], columns=cols)
        results.iloc[:,:2] = file_test.iloc[:,1:3]
        estimator = self.get_default_regressions(model)
        for lead_time in lead_times:
            print(lead_time)
            used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
            X_train, X_test = file_train.loc[:,used_predictors], file_test.loc[:,used_predictors]
            y_train, y_test = file_train.loc[:,'yield'], file_test.loc[:,'yield']
            if optimize:
                selected_features,_,_ = self.feature_selection(X_train, y_train, model)
                X_train, X_test = X_train.loc[:, selected_features], X_test.loc[:, selected_features]
                hp_tuned_vals = self.hyper_tune(X_train, y_train, model)
                estimator.set_params(**hp_tuned_vals)
            pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
            pipe.fit(X_train, y_train)
            y_test_pred = pipe.predict(X_test)
            # y_train_pred = pipe.predict(X_train)
            results.loc[:,f'forecast_LT_{lead_time}'] = y_test_pred
        #
        results.to_csv(f'Results/forecasts/{self.country}_polk_{self.crop}_{model}_loocv_opt={optimize}.csv')

    def run_eracast(self, lead_time, predictor, model='XGB'):
        file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.crop}_{predictor}.csv', index_col=0)
        if file.dtypes['yield'] == 'object':
            file.index = range(file.shape[0])
            inds = np.where(file['yield'] == '[]')[0]
            file = file.drop(axis=0, index=inds)
            file['yield'] = file['yield'].astype('float')
        # predictors = file.columns[3:]
        # lt4_predictors = [a for a in predictors if int(a[-1]) == 4]
        # file = file.drop(lt4_predictors, axis=1)
        file = file.dropna(axis=0)
        predictors = file.columns[3:]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        predictor_file = file.loc[:, used_predictors]
        X, X_test, y, y_test = train_test_split(predictor_file, file.loc[:, 'yield'], test_size=0.2, random_state=5)
        estimator = self.get_default_regressions(model)
        pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
        pipe.fit(X, y)
        y_test_pred = pipe.predict(X_test)
        print(f'train perf for {self.crop} lead_time:{lead_time} R: {pearsonr(y_test_pred, y_test)}')
        return y_test, y_test_pred

    def cross_train(self, lead_time, datasets, model='RF', downscale=False):
        file_train = pd.read_csv(f'Data/M/cz/rost_{self.nameconvert[self.crop]}_{datasets}.csv', index_col=0).drop(
            ['date_last_obs'], axis=1)
        file_test = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.crop}_{datasets}.csv', index_col=0)
        if downscale:
            file_test, file_train = file_train, file_test
        file_train = file_train.dropna(axis=0)
        file_test = file_test.dropna(axis=0)

        predictors = file_train.columns[3:]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        X_train, y_train = file_train.loc[:, used_predictors], file_train.loc[:, 'yield']
        X_test, y_test = file_test.loc[:, used_predictors], file_test.loc[:, 'yield']
        estimator = self.get_default_regressions(model)
        pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
        pipe.fit(X_train, y_train)
        y_test_pred = pipe.predict(X_test)
        # print(y_test_pred, y_test)
        print(f'train perf for {self.crop} lead_time:{lead_time} R: {pearsonr(y_test_pred, y_test)}')
        return y_test, y_test_pred

    def field(self, lead_time, datasets, model='XGB'):
        file = pd.read_csv(f'Data/M/cz/rost_{self.nameconvert[self.crop]}_{datasets}.csv', index_col=0).drop(['date_last_obs'], axis=1)
        # file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.crop}_s2.csv', index_col=0)
        file = file.dropna(axis=0)
        predictors = file.columns[3:]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        X, X_test, y, y_test = train_test_split(file.loc[:, used_predictors], file.loc[:, 'yield'], test_size=0.2, random_state=5)
        estimator = self.get_default_regressions(model)
        pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
        pipe.fit(X, y)
        y_test_pred = pipe.predict(X_test)
        print(f'train perf for lead_time:{lead_time} R: {pearsonr(y_test_pred, y_test)}')
        return y_test, y_test_pred

    def rundeepcast(self, lead_time, train=False):
        # set-up model
        if self.farm:
            file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.farm}_{self.crop}_s1s2.csv', index_col=0)
        else:
            file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.crop}_s1s2.csv', index_col=0)
        file = file.dropna(axis=0)

        # reg = pd.read_csv(f'Data/M/TL/{self.crop}_s1s2_regional.csv')
        # reg = reg.dropna(axis=0)
        # field = pd.read_csv(f'Data/M/TL/{self.crop}_s1s2_field.csv')
        # field = field.dropna(axis=0)
        # file = reg.merge(field, how='outer')

        predictors = [p for p in file.columns[3:] if not p in ['date_last_obs', 'prev_year_crop']]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        used_predictors = [a for a in used_predictors if a.startswith(('sig40_cr', 'evi'))]
        predictor_file = file.loc[:, used_predictors]
        X, X_test, y, y_test = train_test_split(predictor_file, file.loc[:, 'yield'], test_size=0.1, random_state=5)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.11, random_state=5)
        X_train = df2np(X_train)
        X_val = df2np(X_val)
        X_test = df2np(X_test)

        if train:
            model = Sequential()
            model.add(InputLayer(X_train.shape[1:]))
            model.add(LSTM())
            model.add(Dense(8, 'relu'))
            model.add(Dense(1, 'linear'))

            print(model.summary())

            cp = ModelCheckpoint(f'Results/models/lstm3_{self.crop}/', save_best_only=True)
            model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[cp])
            train_predictions = model.predict(X_train).flatten()
            test_predictions = model.predict(X_test).flatten()
            train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
            print(train_results.describe())
            print(pearsonr(train_predictions, y_train)[0])
            print(pearsonr(test_predictions, y_test)[0])

        else:
            model = load_model('Results/models/lstm1/')
            train_predictions = model.predict(X_train).flatten()
            train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
            print(train_results.describe())
            print(pearsonr(train_predictions, y_train)[0])

    def transferlearn(self, lead_time, field2reg=False, tl_train_layers=2, perc_tl_samples=0.5,
                      hidden_layer_sizes=[100, 50, 50, 1], epochs=100, batch_size=10, dropout_perc=0.3,
                      optimizer='adam', early_stopping_patience=3, cv=30, cv_method='random'):
        """
        :param lead_time: int number of months before harvest that forecast is set up
        :param field2reg: boolean transferlearning from field to reg if true
        :param tl_train_layers: int number of hidden layers that are adjusted during tl
        :param perc_tl_samples: float (0,1) percantage of field samples used for updating ANN
        all other parameters are the ones used to set up the dl model. See function dl for further information
        :return:
        """
        reg = pd.read_csv(f'Data/M/TL/{self.crop}_s1s2_regional.csv')
        reg = reg.dropna(axis=0)
        years_reg = reg.c_year
        field = pd.read_csv(f'Data/M/TL/{self.crop}_s1s2_field.csv')
        field = field.dropna(axis=0)
        years_field = field.c_year
        all = reg.merge(field, how='outer')
        predictors = reg.columns[3:]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        # used_predictors = [a for a in used_predictors if a.startswith(('sig40_cr', 'evi'))]

        X_field = field.loc[:, used_predictors]
        X_reg = reg.loc[:, used_predictors]
        X_all = all.loc[:, used_predictors]
        y_field, y_reg, y_all = field.loc[:, 'yield'], reg.loc[:, 'yield'], all.loc[:, 'yield']

        X_train, y_train, X_test_o, y_test_o = X_reg, y_reg, X_field, y_field

        if field2reg:
            X_train, y_train, X_test_o, y_test_o = X_field, y_field, X_reg, y_reg

        if cv_method=='loocv':
            years_1 = np.unique(years_reg)
            years_2 = np.unique(years_field)
            years_u = list(set(years_1).intersection(years_2))
            cv = len(years_u)
            results = pd.DataFrame(index=years_u, columns=['reg_train', 'reg_test', 'tl_train', 'tl_test', 'field_train', 'field_test'])
            results_values = pd.DataFrame(index=[], columns=['year', 'yield_forecast', 'yield_obs'])
        else:
            results = pd.DataFrame(index=range(cv), columns=['reg_train', 'reg_test', 'tl_train', 'tl_test', 'field_train', 'field_test'])

        for i_cv in range(cv):
        # for i_cv in range(1):
            if cv_method=='random':
                X_transfer, X_test, y_transfer, y_test = train_test_split(X_test_o, y_test_o, train_size=0.8)
                print(X_transfer.shape, X_test.shape)
                # X_transfer, _, y_transfer, _ = train_test_split(X_transfer, y_transfer, train_size=perc_tl_samples)
                # print(X_test.shape, X_transfer.shape)
                # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=5)
            elif cv_method=='loocv':
                year = years_u[i_cv]
                year_test, year_transfer = np.where(years_field==year)[0], np.where(years_field!=year)[0]
                year_train = np.where(years_reg!=year)[0]
                X_train, y_train = X_reg.iloc[year_train,:], y_reg.iloc[year_train]
                X_transfer, y_transfer = X_field.iloc[year_transfer,:], y_field.iloc[year_transfer]
                X_test, y_test = X_field.iloc[year_test,:], y_field.iloc[year_test]

            else:
                raise ValueError(f'cv_method: {cv_method} not available. Please choose either random or loocv')
        #
            pipe = dl(X_train.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer, early_stopping_patience)
            pipe.fit(X_train, y_train)

            y_fore = pipe.predict(X_test)
            y_fore_train = pipe.predict(X_train)

            if cv_method == 'loocv':
                results.loc[year, 'reg_train'] = pearsonr(y_fore_train, y_train)[0]
                results.loc[year, 'reg_test'] = pearsonr(y_fore, y_test)[0]
            else:
                results.loc[i_cv, 'reg_train'] = pearsonr(y_fore_train, y_train)[0]
                results.loc[i_cv, 'reg_test'] = pearsonr(y_fore, y_test)[0]

            for i in range(len(hidden_layer_sizes)-tl_train_layers):
                pipe['mlp'].model.layers[i].trainable = False

            pipe.fit(X_transfer, y_transfer)
            y_fore = pipe.predict(X_test)
            y_fore_train = pipe.predict(X_transfer)

            if cv_method == 'loocv':
                results.loc[year, 'tl_train'] = pearsonr(y_fore_train, y_transfer)[0]
                results.loc[year, 'tl_test'] = pearsonr(y_fore, y_test)[0]
            else:
                results.loc[i_cv, 'tl_train'] = pearsonr(y_fore_train, y_transfer)[0]
                results.loc[i_cv, 'tl_test'] = pearsonr(y_fore, y_test)[0]

            if cv_method=='loocv':
                new_rows = {'year': [year]*len(y_fore), 'yield_forecast': y_fore, 'yield_obs': y_test}
                results_values = pd.concat([results_values, pd.DataFrame(new_rows)])
            pipe = 0

            pipe_n = dl(X_transfer.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer, early_stopping_patience)
            pipe_n.fit(X_transfer, y_transfer)
            y_fore = pipe_n.predict(X_test)
            y_fore_train = pipe_n.predict(X_transfer)

            if cv_method == 'loocv':
                results.loc[year, 'field_train'] = pearsonr(y_fore_train, y_transfer)[0]
                results.loc[year, 'field_test'] = pearsonr(y_fore, y_test)[0]
            else:
                results.loc[i_cv, 'field_train'] = pearsonr(y_fore_train, y_transfer)[0]
                results.loc[i_cv, 'field_test'] = pearsonr(y_fore, y_test)[0]
            pipe_n = 0

        if not os.path.exists(os.path.join(self.path_out, 'configs.txt')):
            with open(os.path.join(self.path_out, 'configs.txt'), 'w') as file:
                file.write(f'used configs: field2reg={field2reg}, tl_train_layers={tl_train_layers}, perc_tl_samples='
                           f'{perc_tl_samples}, hidden_layer_sizes={hidden_layer_sizes}, epochs={epochs}, batch_size='
                           f'{batch_size}, dropout_perc={dropout_perc}, optimizer={optimizer}, early_stopping_patience='
                           f'{early_stopping_patience}, cv={cv}')

        results.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_{np.round(perc_tl_samples,1)}.csv')
        if cv_method == 'loocv':
            results_values.to_csv(f'{self.path_out}/{self.crop}_values_{lead_time}_{perc_tl_samples}.csv')

    def get_lead_times(self):
        if self.temp_res=='2W':
            lead_times = [8, 7, 6, 5, 4, 3, 2, 1]
        elif self.temp_res=='M':
            lead_times = [4, 3, 2, 1]
        return lead_times

    #--------------------------------------------- Run and evaluate models -------------------------------------------
    def write_results(self):
        cols = ['lead_time', 'default_run', 'FE', 'HPT', 'FE&HPT']
        lead_times = self.get_lead_times()
        csv_file = pd.DataFrame(data=None, index=lead_times, columns=cols)
        csv_file.loc[:, 'lead_time'] = lead_times
        for lt,lead_time in enumerate(lead_times):
            csv_file.iloc[lt, 1:] = self.runforecast(lead_time=lead_time, model='XGB', feature_select=True, hyper_tune=True)
        csv_file.to_pickle(f'Results/Validation/FE_HPT_{self.temp_res}_unmerged.csv')

    def s1_vs_s2(self, model='XGB'):
        cols = ['s1', 's2', 'both']
        lead_times = self.get_lead_times()
        csv_file = pd.DataFrame(data=None, index=lead_times, columns=cols)
        for lead_time in lead_times:
            for predictor in cols:
                X, X_test, y, y_test = self.get_train_test(lead_time=lead_time)
                if predictor=='s1':
                    selected_features = [a for a in X.columns if not a.startswith(('ndvi','evi','ndwi','nmdi'))]
                elif predictor=='s2':
                    selected_features = [a for a in X.columns if a.startswith(('ndvi', 'evi', 'ndwi', 'nmdi'))]
                else:
                    selected_features = X.columns

                X, X_test = X.loc[:, selected_features], X_test.loc[:, selected_features]
                hp_tuned_vals = {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.3, 'colsample_bytree': 0.5}
                # hp_tuned_vals = self.hyper_tune(lead_time=lead_time, model=model, selected_features=selected_features)
                estimator = self.get_default_regressions(model)
                estimator.set_params(**hp_tuned_vals)
                pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
                print(X.columns)
                scores_kf_hp = cross_val_score(pipe, X, y, cv=30, scoring="explained_variance")
                print(f'train perf for lead_time:{lead_time} R^2: {np.median(scores_kf_hp)}')
                csv_file.loc[lead_time, predictor] = scores_kf_hp
        csv_file.to_pickle(f'Results/Validation/{self.crop}_{model}_s1_s2_{self.temp_res}_unmerged.csv')

    def s1_vs_s2_eco(self, model):
        cols = ['s1', 's2', 'eco', 's1-s2', 'all']
        #lead_times = [4, 3, 2, 1]
        lead_times = self.get_lead_times()
        csv_file = pd.DataFrame(data=None, index=lead_times, columns=cols)
        for lead_time in lead_times:
            for predictor in cols:
                X, X_test, y, y_test = self.get_train_test(lead_time=lead_time)
                if predictor=='s1':
                    selected_features = [a for a in X.columns if a.startswith('sig')]
                elif predictor=='s2':
                    selected_features = [a for a in X.columns if a.startswith(('ndvi', 'evi', 'ndwi', 'nmdi'))]
                elif predictor=='s1-s2':
                    selected_features = [a for a in X.columns if a.startswith(('sig','ndvi', 'evi', 'ndwi', 'nmdi'))]
                elif predictor=='eco':
                    selected_features = [a for a in X.columns if a.startswith('ECO2LSTE')]
                else:
                    selected_features = [a for a in X.columns if a.startswith(('sig','ndvi', 'evi', 'ndwi', 'nmdi', 'ECO'))]

                X, X_test = X.loc[:, selected_features], X_test.loc[:, selected_features]
                print(X.columns)
                # hp_tuned_vals = self.hyper_tune(lead_time=lead_time, model=model, selected_features=selected_features)
                hp_tuned_vals = {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.3, 'colsample_bytree': 0.5}
                estimator = self.get_default_regressions(model)
                estimator.set_params(**hp_tuned_vals)
                pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
                scores_kf_hp = cross_val_score(pipe, X, y, cv=30, scoring="explained_variance")
                print(f'{predictor} test perf for lead_time:{lead_time} R^2: {np.median(scores_kf_hp)}')
                csv_file.loc[lead_time, predictor] = scores_kf_hp
        csv_file.to_pickle(f'Results/Validation/{model}_{self.crop}_s1_s2_ecostress_{self.temp_res}_new.csv')

    def plot_res(self, comp='model_opt', model='XGB'):
        """
        :return: Plots the results generated by self.write_results to a boxplot comparing the performance of the model
                    using S-1, S-2, and all data
        """
        if comp=='eco':
            file = pd.read_pickle(f'Results/Validation/{model}_{self.crop}_s1_s2_ecostress_{self.temp_res}_new.csv')
            file.loc[:,'lead_time'] = file.index

            s1 = file.loc[:, ['lead_time', 's1']]
            s2 = file.loc[:, ['lead_time', 's2']]
            eco = file.loc[:, ['lead_time', 'eco']]
            s1s2 = file.loc[:, ['lead_time', 's1-s2']]
            all = file.loc[:, ['lead_time', 'all']]

            s1 = s1.rename(columns={'s1': 'perf'})
            s2 = s2.rename(columns={'s2': 'perf'})
            eco = eco.rename(columns={'eco': 'perf'})
            s1s2 = s1s2.rename(columns={'s1-s2': 'perf'})
            all = all.rename(columns={'all': 'perf'})

            s1.loc[:, 'predictors'] = ['s1'] * len(s1.lead_time)
            s2.loc[:, 'predictors'] = ['s2'] * len(s2.lead_time)
            eco.loc[:, 'predictors'] = ['eco'] * len(eco.lead_time)
            s1s2.loc[:, 'predictors'] = ['s1-s2'] * len(s1s2.lead_time)
            all.loc[:, 'predictors'] = ['all'] * len(all.lead_time)

            final = pd.concat([s1, s2, s1s2, eco, all])

        elif comp=='s1s2':
            file = pd.read_pickle(f'Results/Validation/{self.crop}_XGB_s1_s2_{self.temp_res}_unmerged.csv')
            # file = pd.read_pickle(f'Results/Validation/s1_vs_s2.csv')
            file.loc[:,'lead_time'] = file.index
            s1 = file.loc[:, ['lead_time', 's1']]
            s2 = file.loc[:, ['lead_time', 's2']]
            all = file.loc[:, ['lead_time', 'both']]
            s1 = s1.rename(columns={'s1': 'perf'})
            s2 = s2.rename(columns={'s2': 'perf'})
            all = all.rename(columns={'both': 'perf'})
            s1.loc[:, 'predictors'] = ['s1'] * len(s1.lead_time)
            s2.loc[:, 'predictors'] = ['s2'] * len(s2.lead_time)
            all.loc[:, 'predictors'] = ['both'] * len(all.lead_time)

            final = pd.concat([s1, s2, all])

        elif comp=='model_opt':
            file = pd.read_pickle('Results/Validation/FE_HPT_M_unmerged.csv')
            default_run = file.loc[:, ['lead_time', 'default_run']]
            FE = file.loc[:, ['lead_time', 'FE']]
            HPT = file.loc[:, ['lead_time', 'HPT']]
            FE_HPT = file.loc[:, ['lead_time', 'FE&HPT']]
            default_run = default_run.rename(columns={'default_run': 'perf'})
            FE = FE.rename(columns={'FE': 'perf'})
            HPT = HPT.rename(columns={'HPT': 'perf'})
            FE_HPT = FE_HPT.rename(columns={'FE&HPT': 'perf'})
            default_run.loc[:, 'predictors'] = ['default_run'] * len(default_run.lead_time)
            FE.loc[:, 'predictors'] = ['FE'] * len(FE.lead_time)
            HPT.loc[:, 'predictors'] = ['HPT'] * len(HPT.lead_time)
            FE_HPT.loc[:, 'predictors'] = ['FE_HPT'] * len(FE_HPT.lead_time)

            final = pd.concat([default_run, FE, HPT, FE_HPT])

        seaborn.set_style('whitegrid')
        print(final)
        print(final.explode('perf'))
        # s = seaborn.boxplot(data=final.explode('perf'), x='lead_time', y='perf', hue='predictors', zorder=2)
        # if self.temp_res=='2W':
        #     s.set_xticklabels(np.linspace(2, 16, 8))
        # elif self.temp_res=='M':
        #     s.set_xticklabels(np.linspace(1, 4, 4))
        # s.set_xlabel('Lead Time [months]')
        # s.set_ylabel('explained variance')
        # s.set_title(self.crop)
        # s.set_ylim([-2,1])
        # lw=0.5
        # lin_col = 'gray'
        # alpha = 0.5
        # s.legend_.set_title(None)
        # [s.axhline(x + .1, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        # [s.axhline(x + .2, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        # [s.axhline(x + .3, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        # [s.axhline(x + .4, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        # [s.axhline(x, color='k', linewidth=lw, zorder=1) for x in s.get_yticks()]
        # # plt.show()
        # plt.savefig(fr'M:\Projects\YIPEEO\04_deliverables_documents\03_ATBD\Figs\ml_validation/{comp}_{self.crop}_{self.temp_res}-2.png', dpi=300)
        # plt.close()

    def feature_imp(self, lead_time, model='XGB'):
        X, X_test, y, y_test = self.get_train_test(lead_time=lead_time, merge_months=False)
        estimator = self.get_default_regressions(model)
        # pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
        # scores_kf = cross_val_score(pipe, X, y, cv=30, scoring="explained_variance")
        # print(f'test perf for lead_time:{lead_time}, R^2: {np.median(scores_kf)}')
        _,_,feature_sel = self.feature_selection(X, y, model=model, n_features=10)
        pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
        pipe.fit(X, y)
        feature_sel.loc[:,'FI'] = pipe.steps[1][1].feature_importances_
        feature_sel.to_csv(f'Results/forecasts/feature_imp/{self.crop}.csv')


    #--------------------------------------------- Auxiliary functions -----------------------------------------------
    def get_default_regressions(self, regression_names):
        """
        Function that returns default regression.

        :param regression_names: (list) - list containing 'regression names' as strings,
               options: [LASSO, SVR, RF, SGD, KNN, GPR, DTR]
        :return: regression
        """
        regressions = {}

        # ---------- LASSO REGRESSION ----------
        if 'LASSO' in regression_names:
            LASSO = Lasso()
            regressions = LASSO

        # ---------- SUPPORT VECTOR MACHINES ----------
        elif 'SVR' in regression_names:
            SVR = svm.SVR(kernel='rbf',
                          degree=3,
                          gamma='scale',
                          coef0=0.0,
                          tol=0.001,
                          C=1.0,
                          epsilon=0.1,
                          shrinking=True,
                          cache_size=200,
                          verbose=False,
                          max_iter=-1)
            regressions = SVR

        # ---------- RANDOM FOREST ----------
        elif 'RF' in regression_names:
            RF = RandomForestRegressor(
                n_estimators=200,
                # min_samples_split=10,
                # max_leaf_nodes=5,
                # max_depth=None,
            )
            regressions = RF

        # ---------- XGB ----------
        elif 'XGB' in regression_names:
            XGB = xgb.XGBRegressor()
            regressions = XGB

        # ---------- Linear Regression ----------
        elif 'LR' in regression_names:
            LR = LinearRegression()
            regressions = LR

        # ---------- MLP ----------
        elif 'MLP' in regression_names:
            MLP = MLPRegressor(
                hidden_layer_sizes=(100,50,50),
                activation='relu'
            )
            regressions = MLP

        # ---------- DUMMY Regression ----------
        elif 'Dummy' in regression_names:
            Dum = DummyRegressor()
            regressions = Dum

        return regressions

    #ToDo merge only from t-2 on
    def merge_previous_months(self, file):
        all_predictors = file.columns
        lead_times = np.unique([int(a[-1]) for a in all_predictors])
        if len(lead_times) > 2:
            predictors = np.unique([a[:-4] for a in all_predictors])
            col_names = [[a + b for a in predictors] for b in ['_latest_month', '_mean_before']]
            col_names = list(itertools.chain(*col_names))
            file_out = pd.DataFrame(index=file.index, columns=col_names)
            for predictor in predictors:
                file_out.loc[:, predictor + '_latest_month'] = file.loc[:, predictor + f'_LT{np.min(lead_times)}']
                time_before = [a for a in all_predictors if (a.startswith(predictor)) and (int(a[-1]) > 0)]
                df_before = file.loc[:, time_before]
                file_out.loc[:, predictor + '_mean_before'] = df_before.mean(1)
        else:
            file_out = file

        return file_out

    #ToDo: finish select_preds
    def select_preds(self, file, lt, preds):
        if lt and preds:
            raise ValueError('Please only use lt or preds')
        if not lt and not preds:
            file=file
        if lt and not preds:
            predictors = file.columns
            used_pred = [p for p in predictors if p.endswith(str(lt))]
            file = file.loc[:,used_pred]
        if preds and not lt:
            predictors = file.columns
            used_pred = [p for p in predictors if p.startswith(tuple(preds))]
            file = file.loc[:,used_pred]
        return file

    def get_train_test(self, lead_time, filename=None, merge_months=False):
        if filename is None:
            if self.farm:
                filename = f'Data/{self.temp_res}/{self.country}/{self.farm}_{self.crop}_s1s2.csv'
            else:
                filename = f'Data/{self.temp_res}/{self.country}/{self.crop}_s1s2.csv'
        if self.farm:
            file = pd.read_csv(filename, index_col=0)
        else:
            file = pd.read_csv(filename, index_col=0)
        file = file.dropna(axis=0)
        predictors = [p for p in file.columns[4:] if not p in ['date_last_obs', 'prev_year_crop']]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        # used_predictors.append('prev_year_crop')
        if merge_months:
            predictor_file = self.merge_previous_months(file.loc[:, used_predictors])
        else:
            predictor_file = file.loc[:, used_predictors]

        X, X_test, y, y_test = train_test_split(predictor_file, file.loc[:, 'yield'], test_size=0.2, random_state=5)
        return X, X_test, y, y_test


def write_res_field(method):
    for crop in ['maize', 'spring_barley', 'winter_wheat']:
        ds = 's1s2'
        if crop=='winter_wheat':
            ds = 's2'
        a = ml(crop=crop, country='Czechia', temp_res='M')
        for i,lead_time in enumerate([4,3,2,1]):
            if method=='cross_trained':
                y_test, y_test_pred = a.cross_train(lead_time=lead_time, datasets=ds, model='XGB', downscale=True)
            else:
                y_test, y_test_pred = a.field(lead_time=lead_time, datasets=ds, model='XGB')
            if i==0:
                output = pd.DataFrame(index=range(len(y_test)), columns=['observed','LT4','LT3','LT2','LT1'])
                output.iloc[:,0] = y_test
                output.iloc[:, i+1] = y_test_pred
            else:
                output.iloc[:, i+1] = y_test_pred
        output.to_csv(f'Results/forecasts/EDD/field_scale/{crop}_{method}.csv', index=0)

def write_res_reg(country, method, dataset):
    for crop in ['winter_wheat']:
        a = ml(crop=crop, country=country, temp_res='M')
        ds = 's1s2'
        if crop=='winter_wheat':
            ds = 's2'
        for i,lead_time in enumerate([4,3,2,1]):
            print('predict regional data:')
            if method == 'cross_trained':
                y_test, y_test_pred = a.cross_train(lead_time=lead_time, datasets=ds, model='XGB', downscale=False)
            else:
                y_test, y_test_pred = a.run_eracast(lead_time=lead_time, predictor=dataset, model='XGB')
            if i==0:
                output = pd.DataFrame(index=range(len(y_test)), columns=['observed','LT4','LT3','LT2','LT1'])
                output.iloc[:,0] = y_test
                output.iloc[:, i+1] = y_test_pred
            else:
                output.iloc[:, i+1] = y_test_pred
        if method == 'cross_trained':
            output.to_csv(f'Results/forecasts/EDD/regional_scale/{country}/{crop}_{method}.csv', index=0)
        else:
            output.to_csv(f'Results/forecasts/EDD/regional_scale/{country}/{crop}_reg_trained_{dataset}.csv', index=0)

def run_write_reg():
    dss = ['s1s2']
    for country in ['Czechia', 'Austria']:
        # write_res_reg(country=country, method='cross_trained', dataset=ds)
        for ds in dss:
            write_res_reg(country=country, method='reg_trained', dataset=ds)

def df2np(df):
    X = []
    lts = list(np.unique([int(a[-1]) for a in df.columns]))
    cl = {}
    for lt in lts:
        this_cols = [a for a in df.columns if a.endswith(str(lt))]
        cl[lt] = df.loc[:,this_cols].to_numpy()
    lts.sort(reverse=True)
    for i in range(len(df.index)):
        row = [list(cl[lt][i]) for lt in lts]
        X.append(row)
    return np.array(X)

def dl(input_shape, hidden_layer_sizes=[100,50,50,1], epochs=100, batch_size=10, dropout_perc=0.1, optimizer='adam', early_stopping_patience=None):
    """
    tuning the MLP
    :param input_shape: int number of predictors
    :param hidden_layer_sizes: list of model architecture. list entries are number of neurons per hidden layer
    :param epochs: int number of epochs over which model is fitted
    :param batch_size: int number of samples to work through before updating the internal model parameters
    :param dropout: float (0,1) percentage of neurons used for dropout in first hidden layer
    :param optimizer: str either adam or sgd. SGD is pretuned with learning rate=0.01 adn momentum=0.9.
    :param early_stopping_patience: int number of epochs without loss reduction until model stops calibrating
    :return: scikit pipeline
    """
    model = Sequential()
    for i, neur in enumerate(hidden_layer_sizes):
        if i==0:
            model.add(Dense(neur, input_shape=input_shape, activation='relu'))
            if dropout_perc:
                model.add(Dropout(dropout_perc))
        else:
            model.add(Dense(neur, 'relu'))

    if optimizer=='sgd':
        sgd = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[RootMeanSquaredError()])
    elif optimizer=='adam':
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[RootMeanSquaredError()])
    else:
        raise ValueError(f'optimizer {optimizer} not available')

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    if early_stopping_patience:
        estimators.append(('mlp', KerasRegressor(model=model, epochs=epochs, batch_size=batch_size,
                                callbacks=[EarlyStopping(monitor='loss', patience=early_stopping_patience)])))
    else:
        estimators.append(('mlp', KerasRegressor(model=model, epochs=epochs, batch_size=batch_size)))
    pipe = Pipeline(estimators)

    return pipe

def plotting():
    """
    Paper_plot
    :return: Plots the results generated by self.write_results to a boxplot comparing the performance of the model
                using S-1, S-2, and all data
    """
    lead_times = [2,1]
    fig = plt.figure(figsize=(12, 5))

    for pt, crop in enumerate(['maize', 'winter_wheat', 'spring_barley']):
        crop_name = crop.replace('_', ' ').capitalize()
        for lt in lead_times:
            file = pd.read_csv(f'Results/Validation/dl/run1/{crop}_{lt}.csv', index_col=0)
            file.loc[:,'lead_time'] = [lt]*len(file.index)
            reg2field = file.loc[:, ['reg_test', 'lead_time']]
            tl = file.loc[:, ['tl_test', 'lead_time']]
            field = file.loc[:, ['field_test', 'lead_time']]
            reg2field = reg2field.rename(columns={'reg_test': 'perf'})
            tl = tl.rename(columns={'tl_test': 'perf'})
            field = field.rename(columns={'field_test': 'perf'})

            reg2field.loc[:, 'predictors'] = ['reg2field'] * len(reg2field.lead_time)
            tl.loc[:, 'predictors'] = ['tl'] * len(tl.lead_time)
            field.loc[:, 'predictors'] = ['field'] * len(field.lead_time)

            final_lt = pd.concat([reg2field, tl, field])
            if lt==np.max(lead_times):
                final = final_lt
            else:
                final = pd.concat([final, final_lt])

        outer = gridspec.GridSpec(1, 3, wspace=0.12)
        ax = plt.Subplot(fig, outer[pt])

        seaborn.set_style('whitegrid')
        final.index = range(len(final.index))
        s = seaborn.boxplot(data=final, x='lead_time', y='perf', hue='predictors', zorder=2, ax=ax)
        s.set_xticklabels(np.linspace(1, 2, 2).astype(int))
        s.set_xlabel('Lead Time [months]')
        s.set_title(crop_name)
        s.set_ylim([-1,1])
        lw=0.5
        lin_col = 'gray'
        alpha = 0.5
        s.legend_.set_title(None)
        [s.axhline(x + .1, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        [s.axhline(x + .2, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        [s.axhline(x + .3, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        [s.axhline(x + .4, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        [s.axhline(x, color='k', linewidth=lw, zorder=1) for x in s.get_yticks()]
        if pt==0:
            s.set_ylabel("Pearson's R")
        else:
            ax.set_yticks([])
            s.set_ylabel("")
        fig.add_subplot(ax)
    # plt.show()
    plt.savefig(f'Results/Figs/tl_comparison_2.png', dpi=300)
    plt.close()

def plot_loocv():
    """
    :return: Paper_plot
    """
    fig = plt.figure(figsize=(12, 6))
    seaborn.set(font_scale=1.2)
    seaborn.set_style('ticks')

    crops = ['maize', 'winter_wheat', 'spring_barley']
    for pt, crop in enumerate(crops):
        crop_name = crop.replace('_', ' ').capitalize()
        a = pd.read_csv(f'Results/Validation/dl/loocv0/run5/{crop}_values_1.csv', index_col=0)
        corrs = pd.read_csv(f'Results/Validation/dl/loocv0/run5/{crop}_1.csv', index_col=0)
        corrs.loc[:, 'year'] = corrs.index
        mini, maxi = np.nanmin([a.yield_forecast, a.yield_obs]), np.nanmax([a.yield_forecast, a.yield_obs])

        outer = gridspec.GridSpec(1, len(crops)+1, wspace=0.2, width_ratios=[0.3, 0.3, 0.3, 0.1])
        inner = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=outer[pt], hspace=0.2, height_ratios=[0.4, 0.6])

        ax1 = plt.Subplot(fig, inner[1])

        seaborn.scatterplot(data=a, x='yield_forecast', y='yield_obs', hue='year', palette=cmr.managua, ax=ax1)
        line = np.linspace(0, 20, 100)
        ax1.plot(line, line, ':k', alpha=0.8)
        ax1.legend([],[], frameon=False)
        ax1.legend_.set_title(f"R={np.round(pearsonr(a.yield_forecast, a.yield_obs)[0],2)}")
        ax1.set_xlabel('Forecasted yield [t/ha]')
        if pt==0:
            ax1.set_ylabel('Observed yield [t/ha]')
        else:
            ax1.set_ylabel('')
        ax1.set_ylim(mini, maxi)
        ax1.set_xlim(mini, maxi)
        ha, le = ax1.get_legend_handles_labels()
        fig.add_subplot(ax1)

        ax0 = plt.Subplot(fig, inner[0])
        seaborn.barplot(corrs.tl_test, ax=ax0)
        ax0.set_title(crop_name, fontsize=16)
        ax0.set_ylim(-1, 1)
        ax0.grid(axis='y', linestyle='--', color='gray', alpha=0.5)

        ax0.legend([], [], frameon=False)

        if pt==0:
            ax0.set_ylabel("Pearson's R")
        else:
            ax0.set_ylabel('')
        fig.add_subplot(ax0)

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[-1], hspace=0.2, height_ratios=[0.4, 0.6])
    ax2 = plt.Subplot(fig, inner[-1])
    ax2.axis('off')
    # ax2.legend(ha,le,ncol=9, loc='center', fontsize=12)
    ax2.legend(ha, le, ncol=1, bbox_to_anchor=(1.02, 1))
    fig.add_subplot(ax2)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.95)
    # plt.show()
    plt.savefig(f'Results/Figs/loocv.png', dpi=300)
    plt.close()
    #


def plot_tlvsfield():
    samples = [np.round(a, 1) for a in np.linspace(0.1, 0.9, 9)] + [1]
    tl = pd.DataFrame(index=samples, columns=['perc_samples', 'median', 'mean', 'std'])
    field = pd.DataFrame(index=samples, columns=['perc_samples', 'median', 'mean', 'std'])
    fontsize=16
    crops = ['maize', 'winter_wheat', 'spring_barley']
    fig = plt.figure(figsize=(12, 6))
    outer = gridspec.GridSpec(2, 1, hspace=0.5, height_ratios=[0.9, 0.05])
    inner = gridspec.GridSpecFromSubplotSpec(1, len(crops), subplot_spec=outer[0], wspace=0.18, width_ratios=[0.3, 0.3, 0.3])

    for j, crop in enumerate(crops):
        crop_name = crop.replace('_', ' ').capitalize()
        for i in samples:
            a = pd.read_csv(f'Results/Validation/dl/tl_vs_st0/run4/{crop}_1_{i}.csv')
            tl_t = a.tl_test
            field_t = a.field_test
            tl.loc[i, :] = [i * 0.8, np.nanmedian(tl_t), np.nanmean(tl_t), np.nanstd(tl_t)]
            field.loc[i, :] = [i * 0.8, np.nanmedian(field_t), np.nanmean(field_t), np.nanstd(field_t)]

        perc_samples = np.array(list(tl['perc_samples']))*100
        tl_med = np.array(list(tl['median']))
        tl_er = np.array(list(tl['std']))
        field_med = np.array(list(field['median']))
        field_er = np.array(list(field['std']))

        ax = plt.Subplot(fig, inner[j])

        # Plot the first dataset with error bands
        ax.plot(perc_samples, tl_med, color='blue', label='Transfer learned')
        ax.fill_between(perc_samples, tl_med - tl_er, tl_med + tl_er, alpha=0.2, color='blue')

        # Plot the second dataset with error bands
        ax.plot(perc_samples, field_med, color='red', label='Field learned')
        ax.fill_between(perc_samples, field_med - field_er, field_med + field_er, alpha=0.2, color='red')

        ax.set_ylim([0,1])

        # Add labels and title
        ax.set_xlabel('Used samples [%]', fontsize=fontsize)
        if j==0:
            ax.set_ylabel("Pearson's R", fontsize=fontsize)
        else:
            ax.set_ylabel('')
        ax.grid(axis='both', linestyle='--', color='gray', alpha=0.5)
        ha, le = ax.get_legend_handles_labels()
        ax.set_title(crop_name, fontsize=fontsize+2)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        fig.add_subplot(ax)

    axl = plt.Subplot(fig, outer[1])
    axl.axis('off')
    axl.legend(ha, le, ncol=2, fontsize=fontsize, loc='lower center')
    # print(axl.get_position())
    # axl.set_position([-2,-1,-1,0])
    fig.add_subplot(axl)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.01)
    # Show the plot
    # plt.show()
    plt.savefig(f'Results/Figs/tlvsfield.png', dpi=300)
    plt.close()

#ToDo: hyperparatuning in DL
# plot L1yOCV
# How much data required for tl compared to field training
def run_dl():
    path_out_t = 'Results/Validation/dl/tl_vs_st0'
    folders = [int(a[-1]) for a in os.listdir(path_out_t)]
    if len(folders)==0:
        new_folder = f'run1'
    else:
        new_folder = f'run{np.nanmax(folders) + 1}'
    #
    for crop in ['maize', 'winter_wheat', 'spring_barley']:  #'spring_barley', 'winter_wheat', 'maize'
        a = ml(crop=crop, country='TL', path_out=os.path.join(path_out_t, new_folder))
        for lt in [1]:
            # for p_tl_sa in np.linspace(0.1, 0.9, 9):
            a.transferlearn(lead_time=lt, field2reg=False, tl_train_layers=1, perc_tl_samples=1,
                      hidden_layer_sizes=[100, 50, 50, 1], epochs=100, batch_size=10, dropout_perc=0.3,
                      optimizer='adam', early_stopping_patience=3, cv=30, cv_method='random')


if __name__ == '__main__':
    #Later: deep learn, other numbers of features for FS-> self optimize number of features
    s1_path = '/home/mformane/shares/climers/Projects/YIPEEO/07_data/Predictors/eo_ts/s1/field_level'
    pd.set_option('display.max_columns', 15)
    warnings.filterwarnings('ignore')
    start_pro = datetime.now()
    a = nc2table(country='ES', crop='common winter wheat')
    a.resample_s1(s1_path, temp_step='M')
    #a.resample_era_EU()
    # run_write_reg()
    # write_res(method='local_trained')
    # for country in ['Austria', 'Czechia']:
    #     for crop in ['maize', 'winter_wheat', 'spring_barley']:
    #     # crop = 'maize'
    #         a = nc2table(country=country, crop=crop)
    #         a.merge_by_ind(era=False)
        # a.resample_era()
        # a.resample_s1_glob(temp_step='M')
    # a.clean_nuts3('era')
    # for year in range(2016, 2017):
    #     a.resample_s2_glob(temp_step='M', year=year)

    # merge_countries(field=False)

    # a = ml(crop='grain maize and corn-cob-mix', country='cz')
    # a.plot_res()
    # run_dl()
    # plotting()
    # plot_loocv()
    # plot_tlvsfield()

    print(f'calculation stopped and took {datetime.now() - start_pro}')


    #ToDo: later:
    # Find optimal number of features
    # always merge monthly but starting every two weeks
    # Global model???
    # feature_selection based on crosscors




    

