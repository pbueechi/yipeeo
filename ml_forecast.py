import os
import itertools
import csv
import statistics
import warnings
import seaborn
import matplotlib
import pickle
import math
import shap
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Optional, List, Dict
from cmcrameri import cm as cmr
from pandas.tseries.offsets import DateOffset
from glob import glob
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RandomizedSearchCV, KFold
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.dummy import DummyRegressor

from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam, SGD

KERAS_AVAILABLE = True

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
        self.harvest_date = {'common winter wheat': [7,25], 'winter_wheat': [7,25], 'grain maize and corn-cob-mix': [10,10], 'maize':[10,10], 'spring_barley': [7,20]}   #ToDo: dont hardcode harvest dates
        # self.crop_data = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale.shp')
        self.lead_times = ['_LT4','_LT3','_LT2','_LT1']
        # Harvest dates are taken from JRC for EU countries and USDA for Ukraine
        # https://agri4cast.jrc.ec.europa.eu/dataportal/
        # https://ipad.fas.usda.gov/rssiws/al/crop_calendar/umb.aspx
        harvest_date_wheat = {'AT': [6, 30], 'CZ': [7, 16], 'DE': [7, 16], 'FR': [7, 11], 'HR': [6, 30],
                                   'HU': [6, 30], 'PL': [7, 16], 'SI': [7, 16], 'SK': [7, 16], 'UA': [7, 16]}
        harvest_date_maize = {'AT': [9, 30], 'CZ': [9, 30], 'DE': [9, 30], 'FR': [9, 30], 'HR': [9, 30],
                                   'HU': [9, 30], 'PL': [9, 30], 'SI': [9, 30], 'SK': [9, 30], 'UA': [9, 30]}
        harvest_date_barley = {'AT': [6, 30], 'CZ': [7, 16], 'DE': [7, 16], 'FR': [7, 11], 'HR': [6, 30],
                                   'HU': [6, 30], 'PL': [7, 16], 'SI': [7, 16], 'SK': [7, 16], 'UA': [7, 16]}
        self.harvest_date_pc = {'winter_wheat': harvest_date_wheat, 'maize': harvest_date_maize, 'spring_barley': harvest_date_barley}

    def resample_s1(self, temp_step='M'):
        """
        Extracts Sentinel-1 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['sig0_vv_mean_daily', 'sig0_vh_mean_daily', 'sig0_cr_mean_daily', 'sig40_vv_mean_daily', 'sig40_vh_mean_daily', 'sig40_cr_mean_daily']
        inds = np.where((self.crop_data.country_co==self.country)&(self.crop_data.crop_type==self.crop)&(self.crop_data.c_year>2015))[0]
        if self.farm:
            inds = np.where((self.crop_data.farm_code == self.farm) & (self.crop_data.crop_type == self.crop) & (
                        self.crop_data.c_year > 2015))[0]
        pred_file_path = r'D:\DATA\yipeeo\Predictors\S1\daily'
        self.crop_data = self.crop_data.iloc[inds,:]

        pipeline_df = self.crop_data.iloc[:,[1,5,10]]
        pipeline_df.index = range(len(pipeline_df.index))
        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))
        pipeline_df.loc[:,col_names] = np.nan

        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id,pipeline_df.c_year):
            field = field.split('_')[-1]
            path_nc = os.path.join(pred_file_path, f'{self.country}_{field}_{year}_cleaned_cr_agg_daily.nc')
            if not os.path.exists(path_nc):
                print(f'file {self.country}_{field}_{year}_cleaned_cr_agg_daily.nc does not exist')
                continue
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

    def resample_s2(self, temp_step='M'):
        """
        Extracts Sentinel-2 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['ndvi', 'evi', 'ndwi', 'nmdi']
        inds = np.where((self.crop_data.country_co==self.country)&(self.crop_data.crop_type==self.crop)&
                        (self.crop_data.c_year>2015))[0]
        if self.farm:
            inds = np.where((self.crop_data.farm_code == self.farm) & (self.crop_data.crop_type == self.crop) & (
                        self.crop_data.c_year > 2015))[0]
        pred_file_path = rf'D:\DATA\yipeeo\Predictors\S2_L2A\{self.country}\nc'
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

    def resample_data_EU(self, data_source='era', temp_step='M', anoms=True):
        """
        Extracts ERA5-Land or MODIS data from nc files and saves them as csv
        :param data_source: str either 'era' for ERA5-Land or 'modis' for MODIS data
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :param anoms: bool whether to use anomaly data or absolute values
        :return: saves a csv file which will be used for the ML
        """
        if temp_step == '2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5', '_LT4', '_LT3', '_LT2', '_LT1']

        # Configure parameters and paths based on data source
        if data_source.lower() == 'era':
            params = ['t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr']
            pred_file_path = rf'D:\data-write\YIPEEO\predictors\ERA5-Land\{self.country}\nc'
        elif data_source.lower() == 'modis':
            params = ['evi_median']
            pred_file_path = rf'D:\data-write\YIPEEO\predictors\modis\nc'
        elif data_source == 'vod':
            params = ['VODCA_CXKu']
            pred_file_path = rf'D:\data-write\YIPEEO\predictors\VODCA\EU\nc'
        elif data_source == 'sm':
            params = ['sm']
            pred_file_path = rf'D:\data-write\YIPEEO\predictors\CCI_SM\EU\nc'
        else:
            raise ValueError("data_source must be either 'era' or 'modis'")

        # Load crop data
        if anoms:
            crop_data = pd.read_csv(
                rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{self.crop}_clean_det_anom.csv', index_col=0)
        else:
            crop_data = pd.read_csv(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\Regional_csv\ALL_{self.crop}_clean.csv',
                                    index_col=0)

        # Get available regions
        regions = [a.split('.')[0] for a in os.listdir(pred_file_path) if a.endswith('.nc') if not a.endswith('anomalies.nc')]

        nuts = list(crop_data.index)
        years = [int(a) for a in crop_data.columns]

        # Create column names
        if data_source.lower() == 'era':
            col_names = [[a + b for a in params] for b in self.lead_times]
            col_names = list(itertools.chain(*col_names))
        else:  # modis
            col_names = [params[0] + b for b in self.lead_times]

        # Initialize pipeline DataFrame
        pipeline_df = pd.DataFrame(
            data=None,
            columns=['field_id', 'c_year', 'yield_anom'] + col_names,
            index=range(len(nuts) * len(years))
        )
        year_col = [[a] * len(nuts) for a in years]
        year_col = list(itertools.chain(*year_col))
        pipeline_df.c_year = year_col
        pipeline_df.field_id = nuts * len(years)

        # Process each field and year
        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id, pipeline_df.c_year):
            if field in regions:
                filename = f'{field}.nc'
                if anoms: filename = f'{field}_detrended_anomalies.nc'
                path_nc = os.path.join(pred_file_path, filename)
                pipeline_df.loc[df_ind, 'yield_anom'] = crop_data.loc[field, str(year)]
                s2 = xr.open_dataset(path_nc)

                for param in params:
                    this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                    vals = s2[param].values
                    ndvi = pd.Series(vals, s2.time)

                    # Apply interpolation for MODIS data
                    if data_source.lower() == 'modis':
                        ndvi_b = ndvi.resample('W').mean().interpolate()
                    else:
                        ndvi_b = ndvi.resample('W').mean()

                    this_harvest_date = pd.to_datetime(
                        f'{year}-{self.harvest_date_pc[self.crop][field[:2]][0]}-{self.harvest_date_pc[self.crop][field[:2]][1] - 2}'
                    )
                    start_date = this_harvest_date - DateOffset(months=4)
                    this_df = series2list(ndvi_b[start_date:this_harvest_date], temp_step=temp_step)
                    pipeline_df.loc[df_ind, this_cols] = this_df
            else:
                print(f'region {field} is not available in {year}')

        # Save output
        if anoms:
            output_filename = f'Data/SC2/{temp_step}/{self.crop}_{data_source}.csv'
        else:
            output_filename = f'Data/SC2/{temp_step}/{self.crop}_{data_source}_abs.csv'

        pipeline_df.to_csv(output_filename)

    def era2maize(self, temp_step='M'):
        """
        Extracts Sentinel-1 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr']

        crop_data = pd.read_csv('Data/M/TL/maize_s1s2_regional.csv', index_col=0)
        pred_file_path = rf'D:\data-write\YIPEEO\predictors\ERA5-Land\{self.country}\nc'
        regions_era = [a.split('.')[0] for a in os.listdir(pred_file_path) if not a.endswith('anomalies.nc')]

        nuts = list(crop_data.index)
        years = [int(a) for a in crop_data.columns]

        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))
        pipeline_df = pd.DataFrame(data=None, columns=['field_id', 'c_year', 'yield_anom']+col_names, index=range(len(nuts)*len(years)))
        year_col = [[a] * len(nuts) for a in list(years)]
        year_col = list(itertools.chain(*year_col))
        pipeline_df.c_year = year_col
        pipeline_df.field_id = nuts*len(years)

        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id, pipeline_df.c_year):
            if field in regions_era:
                path_nc = os.path.join(pred_file_path, f'{field}.nc')
                pipeline_df.loc[df_ind,'yield_anom'] = crop_data.loc[field, str(year)]
                s2 = xr.open_dataset(path_nc)
                for param in params:
                    this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                    vals = s2[param].values
                    ndvi = pd.Series(vals, s2.time)
                    ndvi_b = ndvi.resample('W').mean()
                    this_harvest_date = pd.to_datetime(f'{year}-{self.harvest_date_pc[self.crop][field[:2]][0]}-{self.harvest_date_pc[self.crop][field[:2]][1]-2}') #harvest date -2 days to make sure harvested field is not included
                    start_date = this_harvest_date - DateOffset(months=4)
                    this_df = series2list(ndvi_b[start_date:this_harvest_date], temp_step=temp_step)
                    pipeline_df.loc[df_ind, this_cols] = this_df
            else:
                print(f'region {field} is not available in {year}')

        # pipeline_df = pipeline_df.dropna(axis=0, how='any')
        # pipeline_df.to_csv(f'Data/SC2/{temp_step}/{self.crop}_era_eu_abs.csv')

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
        # pipeline_df.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_s1.csv')
        pipeline_df.to_csv(f'Data/{temp_step}/TL/{self.crop}_s1.csv')

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
        files = [a for a in glob(path) if a.endswith(('_s1.csv', '_s2.csv'))]
        print(files)
        for i, file in enumerate(files):
            if i==0:
                csv_fin = pd.read_csv(file, index_col=0)
            else:
                csv_next = pd.read_csv(file, index_col=0)
                # if not (len(csv_fin.c_year)==np.sum(csv_fin.c_year==csv_next.c_year)) and (len(csv_fin.field_id)==np.sum(csv_fin.field_id==csv_next.field_id)):
                #     raise ValueError('the files do not correspond')
                csv_fin = csv_fin.merge(csv_next)
        if self.farm:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}/{self.farm}_{self.crop}_s1s2.csv')
        else:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}/{self.crop}_s1s2.csv')

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

    def merge_cz_at(self):
        path_at = 'Data/M/Austria/winter_wheat_s1s2.csv'
        path_cz = 'Data/M/Czechia/winter_wheat_s1s2.csv'
        at = pd.read_csv(path_at, index_col=0)
        cz = pd.read_csv(path_cz, index_col=0)
        both = pd.concat([at, cz], axis=0)
        both.to_csv('Data/M/TL/winter_wheat_s1s2_regional.csv', index=False)

    def merge_s_era(self):
        path_era = f'Data/SC2/M/{self.crop}_era_eu.csv'
        path_s = f'Data/M/TL/{self.crop}_s1s2_regional.csv'
        era = pd.read_csv(path_era, index_col=0)
        era.columns = list(era.columns[:3]) + ['era_' + a for a in era.columns[3:]]
        s = pd.read_csv(path_s)
        s.field_id = [a[:6] if a.startswith('CZ') else 'AT' + a for a in s.field_id]
        era.field_id = [a[:6] if a.startswith('CZ') else a for a in era.field_id]
        s.loc[:, 'field_year'] = [a + '_' + str(b) for a, b in zip(s.field_id, s.c_year)]
        era.loc[:, 'field_year'] = [a + '_' + str(b) for a, b in zip(era.field_id, era.c_year)]
        both = s.merge(era.iloc[:, 3:], on='field_year').drop('field_year', axis=1).dropna(axis=0)
        print(s.dropna(axis=0).shape)
        print(both.shape)
        both.to_csv(f'Data/M/TL/meteo/{self.crop}_regional.csv')

    def merge_s_era_maize(self):
        path_era = f'Data/SC2/M/maize_era_eu_cz.csv'
        path_era_at = f'Data/SC2/M/maize_era_eu.csv'
        path_s = f'Data/M/TL/maize_s1s2_regional.csv'
        era = pd.read_csv(path_era, index_col=0)
        era.columns = list(era.columns[:3]) + ['era_' + a for a in era.columns[3:]]
        era_at = pd.read_csv(path_era_at, index_col=0)
        era_at.columns = list(era_at.columns[:3]) + ['era_' + a for a in era_at.columns[3:]]
        era['region_group'] = era['field_id'].str[:5]
        # print(era_at)
        s = pd.read_csv(path_s)
        inds = [a for a in np.unique(s.field_id) if a.startswith('CZ')]
        era_cz = era.groupby(['c_year', 'region_group']).mean(numeric_only=True).reset_index()
        era_cz = era_cz.iloc[np.where((era_cz.c_year>2015) & (era_cz.c_year<2023))[0], :]
        era_cz = era_cz.rename(columns={"region_group": "field_id"})

        era_at = era_at[era_at['field_id'].str.startswith('AT')]
        era_at = era_at.iloc[np.where((era_at.c_year > 2015) & (era_at.c_year < 2023))[0], :]
        era = pd.concat([era_cz, era_at], ignore_index=True, join='inner')
        era.yield_anom = [np.nan] * len(era.yield_anom)

        s.field_id = [a[:6] if a.startswith('CZ') else 'AT' + a for a in s.field_id]
        s.loc[:, 'field_year'] = [a + '_' + str(b) for a, b in zip(s.field_id, s.c_year)]
        era.loc[:, 'field_year'] = [a + '_' + str(b) for a, b in zip(era.field_id, era.c_year)]
        both = s.merge(era.iloc[:, 3:], on='field_year').drop('field_year', axis=1).dropna(axis=0)
        both.to_csv(f'Data/M/TL/meteo/maize_regional.csv')

    def merge_EU(self,temp_step='M', abs=False):
        """
        :param temp_res: str either M or 2W for aggregating the data monthly or Biweekly
        :return: merges all predictor csv file to one single csv file. Needs to be used if both S-1 and S-2 data should
        bes used as predictors at the same time
        """
        path = f'Data/SC2/{temp_step}/{self.crop}*.csv'
        if abs:
            files = [a for a in glob(path) if a.endswith('_abs.csv')]
        else:
            files = [a for a in glob(path) if not a.endswith('_abs.csv')]
        print(files)
        for i, file in enumerate(files):
            if i==0:
                csv_fin = pd.read_csv(file, index_col=0)
            else:
                csv_next = pd.read_csv(file, index_col=0)
                # if not (len(csv_fin.c_year)==np.sum(csv_fin.c_year==csv_next.c_year)) and (len(csv_fin.field_id)==np.sum(csv_fin.field_id==csv_next.field_id)):
                #     raise ValueError('the files do not correspond')
                csv_fin = csv_fin.merge(csv_next)
        if abs:
            csv_fin.to_csv(f'Data/SC2/{temp_step}/final/{self.crop}_all_abs.csv')
        else:
            csv_fin.to_csv(f'Data/SC2/{temp_step}/final/{self.crop}_all.csv')

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

    def era2field(self):
        field = pd.read_csv(f'Data/M/TL/{self.crop}_s1s2_field.csv')
        if self.crop=='maize':
            era = pd.read_csv('Data/SC2/M/maize_era_eu_cz.csv', index_col=0)
        else:
            era = pd.read_csv(f'Data/SC2/M/{self.crop}_era_eu.csv', index_col=0)
        nuts_id = 'CZ0646000'

        for newcol in era.columns[3:]:
            field[newcol] = np.nan

        for year in np.unique(field.c_year):
            print(year)
            new_row_i = np.where((era.c_year==year) & (era.field_id==nuts_id))[0]
            new_row = era.iloc[new_row_i, 3:]
            field_year_i = np.where(field.c_year==year)[0]
            field.iloc[field_year_i, -len(era.columns[3:]):] = new_row
        field = field.reset_index(drop=True)
        field.to_csv(f'Data/M/TL/meteo/{self.crop}_field.csv', index=False)

def preproc_tabs(crop):
    crop_data = pd.read_csv(f'Data/M/TL/{crop}_regional_o.csv')
    [crop_data[c].where(crop_data[c] < 0, np.nan, inplace=True) for c in crop_data.columns if c.startswith('sig')]
    crop_data = crop_data.dropna(axis=0)
    crop_data.to_csv(f'Data/M/TL/{crop}_regional.csv', index=False)


def _create_ann_model(input_dim: int, hidden_layer_sizes: List[int], dropout_perc: float,
                      optimizer: str, learning_rate: float = 0.001, momentum: float = 0.9):
    """
    Helper function to create ANN model

    Args:
        input_dim: Number of input features
        hidden_layer_sizes: List of neurons per layer
        dropout_perc: Dropout percentage
        optimizer: 'adam' or 'sgd'
        learning_rate: Learning rate for SGD
        momentum: Momentum for SGD

    Returns:
        Function that creates and compiles a Keras model
    """

    def create_model():
        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        # Add hidden layers
        for i, n_neurons in enumerate(hidden_layer_sizes[:-1]):
            model.add(Dense(n_neurons, activation='relu'))
            if dropout_perc > 0:
                model.add(Dropout(dropout_perc))

        # Output layer
        model.add(Dense(hidden_layer_sizes[-1], activation=None))

        # Compile
        if optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=momentum)
        else:
            opt = 'adam'

        model.compile(
            loss='mean_squared_error',
            optimizer=opt,
            metrics=[RootMeanSquaredError()]
        )

        return model

    return create_model


def _create_keras_regressor(input_dim: int, hidden_layer_sizes: List[int], epochs: int,
                            batch_size: int, dropout_perc: float, optimizer: str,
                            early_stopping_patience: Optional[int]) -> KerasRegressor:
    """
    Create a KerasRegressor with specified configuration
    """
    create_model_fn = _create_ann_model(input_dim, hidden_layer_sizes, dropout_perc, optimizer)

    callbacks = []
    if early_stopping_patience:
        callbacks.append(EarlyStopping(monitor='loss', patience=early_stopping_patience))

    return KerasRegressor(
        model=create_model_fn,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks if callbacks else None,
        verbose=0
    )


def _normalize_rmse(rmse: float, y_true: np.ndarray) -> float:
    """Calculate normalized RMSE"""
    return rmse / np.nanmean(y_true)

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
    def calc_corrs(self):
        """
        :return: calculates correlations between predictors and crop yields
        """
        file = pd.read_csv(f'Data/M/{self.country}/{self.crop}_all_2018.csv', index_col=0)
        predictors = list(file.columns[3:])
        predictors.remove('prev_year_crop')
        crop_yield = file.loc[:, 'yield']
        index = np.unique([a[:-4] for a in predictors])
        forecast_month = [f'LT{i}' for i in [4,3,2,1]]
        corr_df = pd.DataFrame(columns=forecast_month, index=index)
        corr_low = pd.DataFrame(columns=forecast_month, index=index)
        corr_high = pd.DataFrame(columns=forecast_month, index=index)

        for d, dep_var in enumerate(index):
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

    def plot_corrs(self):
        """
        :param correlation_dataframe: df in the format as returning from calculate_correlations
            considered months as columns, dataset names as index
        :param corr_low: same as correlation_dataframe, only with the low boundary of the confidence interval
        :param corr_high: equivalent to corr_low for upper boundaries

        :return: shows a figure of the correlations of each dataset to the yields
        """
        corr_df, corr_high, corr_low = self.calc_corrs()
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
        plt.savefig(f'Figures/Predictor_analysis/corr_{self.crop}', dpi=300)
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
        pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
        scores_kf = cross_val_score(pipe, X, y, cv=20, scoring="explained_variance")
        return rf_random.best_params_

    #---------------------------------------------- Set up models ----------------------------------------------------
    def runforecast(self, lead_time, model='RF', merge_months=False, feature_select=False, hyper_tune=False):
        X, X_test, y, y_test = self.get_train_test(lead_time=lead_time, merge_months=merge_months)
        estimator = self.get_default_regressions(model)
        pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
        scores_kf = cross_val_score(pipe, X, y, cv=30, scoring="explained_variance")
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
            X, X_test, y, y_test = self.get_train_test(lead_time=lead_time, merge_months=merge_months)
            hp_tuned_vals = self.hyper_tune(lead_time=lead_time, model=model, selected_features=X.columns)
            estimator.set_params(**hp_tuned_vals)
            pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
            scores_kf_hp = cross_val_score(pipe, X, y, cv=30, scoring="explained_variance")
            print(f'train perf for lead_time:{lead_time} after hp tuning, R^2: {np.median(scores_kf_hp)}')
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

        lead_times = [4,3,2,1]
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
        lead_times = [4,3,2,1]
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

    def transferlearn_old(self, lead_time, field2reg=False, tl_train_layers=2, perc_tl_samples=1,
                      hidden_layer_sizes=[100, 50, 50, 1], epochs=100, batch_size=10, dropout_perc=0.3,
                      optimizer='adam', early_stopping_patience=3, cv=30, cv_method='random', preds='all',
                      not_random=False, meteo=False, standardize=False):
        """
        :param lead_time: int number of months before harvest that forecast is set up
        :param field2reg: boolean transferlearning from field to reg if true
        :param tl_train_layers: int number of hidden layers that are adjusted during tl
        :param perc_tl_samples: float (0,1) fraction of field samples used for updating ANN
        all other parameters are the ones used to set up the dl model. See function dl for further information
        :param cv_method: str random for random cross validation, loocv for leave one year out cv
        :param preds: tuple of used predictors. If all predictors should be used, use str all
        :return:
        """
        if meteo:
            reg = pd.read_csv(f'Data/M/TL/meteo/{self.crop}_regional.csv')
            field = pd.read_csv(f'Data/M/TL/meteo/{self.crop}_field.csv')
        else:
            reg = pd.read_csv(f'Data/M/TL/cum/{self.crop}_regional.csv')
            field = pd.read_csv(f'Data/M/TL/cum/{self.crop}_field.csv')
        if self.crop=='maize':
            reg = reg[reg.field_id != 'CZ064']
        else:
            reg = reg[reg.field_id != 'CZ0646000000']

        reg = reg.dropna(axis=0)
        years_reg = reg.c_year
        field = field.dropna(axis=0)
        years_field = field.c_year
        all = reg.merge(field, how='outer')
        predictors = reg.columns[3:]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        if not preds=='all':
            used_predictors = [a for a in used_predictors if a.startswith(preds)]
        preds_unique = np.unique([a[:-4] for a in used_predictors])

        X_field = field.loc[:, used_predictors]
        X_reg = reg.loc[:, used_predictors]
        X_all = all.loc[:, used_predictors]
        y_field, y_reg, y_all = field.loc[:, 'yield'], reg.loc[:, 'yield'], all.loc[:, 'yield']

        X_train, y_train, X_test_o, y_test_o = X_reg, y_reg, X_field, y_field

        if field2reg:
            X_train, y_train, X_test_o, y_test_o = X_field, y_field, X_reg, y_reg

        X_train, X_test_reg, y_train, y_test_reg = train_test_split(X_train, y_train, train_size=0.8)
        col_names = ['reg_train', 'reg_test', 'tl_train', 'tl_test', 'field_train', 'field_test', 'reg2reg_test',
                     'xgb_train', 'xgb_test']
        if cv_method=='loocv':
            years_1 = np.unique(years_reg)
            years_2 = np.unique(years_field)
            years_u = list(set(years_1).intersection(years_2))
            cv = len(years_u)
            results = pd.DataFrame(index=years_u, columns=col_names)
            results_values = pd.DataFrame(index=[], columns=['year', 'yield_forecast', 'yield_obs'])
        else:
            results = pd.DataFrame(index=range(cv), columns=col_names)
            # results_values = pd.DataFrame(index=[], columns=['cv', 'yield_forecast', 'yield_obs'])
            # results_values_dict = {a: results_values for a in col_names}
        results_rmse = results.copy()

        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.01, colsample_bytree=0.3)

        for i_cv in range(cv):
        # for i_cv in range(1):
            if cv_method=='random':
                X_transfer, X_test, y_transfer, y_test = train_test_split(X_test_o, y_test_o, train_size=0.8)
                if perc_tl_samples<1:
                    if not_random:
                        n_samples = int(np.round(len(y_transfer)*perc_tl_samples))
                        y_transfer = y_transfer.rename('crop_yield')
                        combined = pd.concat([X_transfer, y_transfer], axis=1)
                        combined = combined.sort_values('crop_yield', axis=0)
                        X_transfer = combined.iloc[:n_samples, :-1]
                        y_transfer = combined.iloc[:n_samples, -1]
                    else:
                        X_transfer, _, y_transfer, _ = train_test_split(X_transfer, y_transfer, train_size=perc_tl_samples)

                X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=5)
            elif cv_method=='loocv':
                year = years_u[i_cv]
                year_test, year_transfer = np.where(years_field==year)[0], np.where(years_field!=year)[0]
                year_train = np.where(years_reg!=year)[0]
                X_train, y_train = X_reg.iloc[year_train,:], y_reg.iloc[year_train]
                X_transfer, y_transfer = X_field.iloc[year_transfer,:], y_field.iloc[year_transfer]
                X_test, y_test = X_field.iloc[year_test,:], y_field.iloc[year_test]

            else:
                raise ValueError(f'cv_method: {cv_method} not available. Please choose either random or loocv')

            if standardize:
                y_train = y_train.values.reshape(-1,1)
                y_test = y_test.values.reshape(-1, 1)
                y_transfer = y_transfer.values.reshape(-1, 1)
                scaler_xtrain = StandardScaler().fit(X_train)
                scaler_xtest = StandardScaler().fit(X_test)
                scaler_xtransfer = StandardScaler().fit(X_transfer)
                scaler_ytrain = StandardScaler().fit(y_train)
                scaler_ytest = StandardScaler().fit(y_test)
                scaler_ytransfer = StandardScaler().fit(y_transfer)

                X_train = pd.DataFrame(scaler_xtrain.transform(X_train))
                X_test = pd.DataFrame(scaler_xtest.transform(X_test))
                X_transfer = pd.DataFrame(scaler_xtransfer.transform(X_transfer))

                y_train = pd.Series(scaler_ytrain.transform(y_train).flatten())
                y_test = pd.Series(scaler_ytest.transform(y_test).flatten())
                y_transfer = pd.Series(scaler_ytransfer.transform(y_transfer).flatten())

            pipe = dl_old(X_train.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer, early_stopping_patience)
            pipe.fit(X_train, y_train)

            if i_cv==0:
                pipe['mlp'].model.save(os.path.join(self.path_out, f'{self.crop}_{lead_time}_init_model.keras'))

            y_fore = pipe.predict(X_test)
            y_fore_train = pipe.predict(X_train)
            y_fore_reg = pipe.predict(X_test_reg)

            if cv_method == 'loocv':
                ind_i = year
            else:
                ind_i = i_cv
            results.loc[ind_i, 'reg_train'] = pearsonr(y_fore_train, y_train)[0]**2
            results.loc[ind_i, 'reg_test'] = pearsonr(y_fore, y_test)[0]**2
            results.loc[ind_i, 'reg2reg_test'] = pearsonr(y_fore_reg, y_test_reg)[0]**2
            results_rmse.loc[ind_i, 'reg_train'] = root_mean_squared_error(y_fore_train, y_train)/np.mean(y_train)
            results_rmse.loc[ind_i, 'reg_test'] = root_mean_squared_error(y_fore, y_test)/np.mean(y_train)
            results_rmse.loc[ind_i, 'reg2reg_test'] = root_mean_squared_error(y_fore_reg, y_test_reg)/np.mean(y_train)

            """
            results_values_dict['reg_train'] = add_rows(results_values_dict['reg_train'], cv=i_cv, y_fore=y_fore_train, y_obs=y_train)
            results_values_dict['reg_test'] = add_rows(results_values_dict['reg_test'], cv=i_cv, y_fore=y_fore, y_obs=y_test)
            results_values_dict['reg2reg_test'] = add_rows(results_values_dict['reg2reg_test'], cv=i_cv, y_fore=y_fore_reg, y_obs=y_test_reg)
            """
            for i in range(len(hidden_layer_sizes)-tl_train_layers):
                pipe['mlp'].model.layers[i].trainable = False

            pipe.fit(X_transfer, y_transfer)

            pipe['mlp'].model.save(os.path.join(self.path_out, f'{self.crop}_{lead_time}_transfer_model.keras'))

            y_fore = pipe.predict(X_test)
            y_fore_train = pipe.predict(X_transfer)

            results.loc[ind_i, 'tl_train'] = pearsonr(y_fore_train, y_transfer)[0]**2
            results.loc[ind_i, 'tl_test'] = pearsonr(y_fore, y_test)[0]**2
            results_rmse.loc[ind_i, 'tl_train'] = root_mean_squared_error(y_fore_train, y_transfer)/np.mean(y_train)
            results_rmse.loc[ind_i, 'tl_test'] = root_mean_squared_error(y_fore, y_test)/np.mean(y_train)

            """
            results_values_dict['tl_train'] = add_rows(results_values_dict['tl_train'], cv=i_cv, y_fore=y_fore_train, y_obs=y_transfer)
            results_values_dict['tl_test'] = add_rows(results_values_dict['tl_test'], cv=i_cv, y_fore=y_fore, y_obs=y_test)
            """
            if cv_method=='loocv':
                new_rows = {'year': [year]*len(y_fore), 'yield_forecast': y_fore, 'yield_obs': y_test}
                results_values = pd.concat([results_values, pd.DataFrame(new_rows)])
            pipe = 0

            pipe_n = dl(X_transfer.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer, early_stopping_patience)
            pipe_n.fit(X_transfer, y_transfer)
            y_fore = pipe_n.predict(X_test)
            y_fore_train = pipe_n.predict(X_transfer)

            results.loc[ind_i, 'field_train'] = pearsonr(y_fore_train, y_transfer)[0]**2
            results.loc[ind_i, 'field_test'] = pearsonr(y_fore, y_test)[0]**2
            results_rmse.loc[ind_i, 'field_train'] = root_mean_squared_error(y_fore_train, y_transfer)/np.mean(y_train)
            results_rmse.loc[ind_i, 'field_test'] = root_mean_squared_error(y_fore, y_test)/np.mean(y_train)

            """
            results_values_dict['field_train'] = add_rows(results_values_dict['field_train'], cv=i_cv, y_fore=y_fore_train, y_obs=y_transfer)
            results_values_dict['field_test'] = add_rows(results_values_dict['field_test'], cv=i_cv, y_fore=y_fore, y_obs=y_test)
            """
            pipe_n = 0

            xgb_model.fit(X_transfer, y_transfer)

            y_fore_xgb = xgb_model.predict(X_test)
            y_fore_train_xgb = xgb_model.predict(X_transfer)

            results.loc[ind_i, 'xgb_train'] = pearsonr(y_fore_train_xgb, y_transfer)[0] ** 2
            results.loc[ind_i, 'xgb_test'] = pearsonr(y_fore_xgb, y_test)[0] ** 2
            results_rmse.loc[ind_i, 'xgb_train'] = root_mean_squared_error(y_fore_train_xgb, y_transfer)/np.mean(y_train)
            results_rmse.loc[ind_i, 'xgb_test'] = root_mean_squared_error(y_fore_xgb, y_test)/np.mean(y_train)


            """
            results_values_dict['dummy_train'] = add_rows(results_values_dict['dummy_train'], cv=i_cv, y_fore=y_fore_train_dummy, y_obs=y_transfer)
            results_values_dict['dummy_test'] = add_rows(results_values_dict['dummy_test'], cv=i_cv, y_fore=y_fore_dummy, y_obs=y_test)
            results_values_dict['xgb_train'] = add_rows(results_values_dict['xgb_train'], cv=i_cv, y_fore=y_fore_train_xgb, y_obs=y_transfer)
            results_values_dict['xgb_test'] = add_rows(results_values_dict['xgb_test'], cv=i_cv, y_fore=y_fore_xgb, y_obs=y_test)
            """

        if not os.path.exists(os.path.join(self.path_out, 'configs.txt')):
            with open(os.path.join(self.path_out, 'configs.txt'), 'w') as file:
                file.write(f'used configs: field2reg={field2reg}\n tl_train_layers={tl_train_layers}\n fraction_tl_samples='
                           f'{perc_tl_samples}\n hidden_layer_sizes={hidden_layer_sizes}\n epochs={epochs}\n batch_size='
                           f'{batch_size}\n dropout_perc={dropout_perc}\n optimizer={optimizer}\n early_stopping_patience='
                           f'{early_stopping_patience}\n cv={cv}\n cv_method={cv_method}'
                           f'\n standardized={standardize}')

        if perc_tl_samples is not None:
            results.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_{np.round(perc_tl_samples,1)}_r2.csv')
            results_rmse.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_{np.round(perc_tl_samples, 1)}_nrmse.csv')
            # with open(f'{self.path_out}/{self.crop}_{lead_time}_{np.round(perc_tl_samples,1)}.pkl', 'wb') as f:
            #     pickle.dump(results_values_dict, f)
        else:
            results.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_0.8_r2.csv')
            results_rmse.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_0.8_nrmse.csv')
            # with open(f'{self.path_out}/{self.crop}_{lead_time}_{np.round(perc_tl_samples,1)}.pkl', 'wb') as f:
            #     pickle.dump(results_values_dict, f)
        if cv_method == 'loocv':
            results_values.to_csv(f'{self.path_out}/{self.crop}_values_{lead_time}_1.csv')

    def transferlearn(self, lead_time, field2reg=False, tl_train_layers=2, perc_tl_samples=1,
                      hidden_layer_sizes=[100, 50, 50, 1], epochs=100, batch_size=10, dropout_perc=0.3,
                      optimizer='adam', early_stopping_patience=3, cv=30, cv_method='random', preds='all',
                      not_random=False, meteo=False, standardize=False, random_state=42):
        """
        Transfer learning for crop yield forecasting

        :param lead_time: int number of months before harvest that forecast is set up
        :param field2reg: boolean transferlearning from field to reg if true
        :param tl_train_layers: int number of hidden layers that are adjusted during tl
        :param perc_tl_samples: float (0,1) fraction of field samples used for updating ANN
        :param hidden_layer_sizes: list of neurons per layer for the ANN architecture
        :param epochs: int number of training epochs
        :param batch_size: int batch size for training
        :param dropout_perc: float dropout percentage
        :param optimizer: str optimizer type ('adam' or 'sgd')
        :param early_stopping_patience: int patience for early stopping
        :param cv: int number of cross-validation folds (only used if cv_method='random')
        :param cv_method: str 'random' for random CV, 'loocv' for leave-one-year-out CV
        :param preds: str or tuple of predictor prefixes to use, or 'all' for all predictors
        :param not_random: bool if True, select samples by yield (lowest first) instead of random
        :param meteo: bool if True, use meteorological data
        :param standardize: bool if True, standardize features and targets
        :param random_state: int random seed for reproducibility
        :return: None (saves results to CSV files)
        """

        # Load data
        if meteo:
            reg = pd.read_csv(f'Data/M/TL/meteo/{self.crop}_regional.csv')
            field = pd.read_csv(f'Data/M/TL/meteo/{self.crop}_field.csv')
        else:
            reg = pd.read_csv(f'Data/M/TL/cum/{self.crop}_regional.csv')
            field = pd.read_csv(f'Data/M/TL/cum/{self.crop}_field.csv')

        # Remove specific fields
        if self.crop == 'maize':
            reg = reg[reg.field_id != 'CZ064']
        else:
            reg = reg[reg.field_id != 'CZ0646000000']

        # Clean data
        reg = reg.dropna(axis=0)
        years_reg = reg.c_year
        field = field.dropna(axis=0)
        years_field = field.c_year
        all_data = reg.merge(field, how='outer')

        # Select predictors based on lead time
        predictors = reg.columns[3:]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        if preds != 'all':
            used_predictors = [a for a in used_predictors if a.startswith(preds)]

        # Prepare datasets
        X_field = field.loc[:, used_predictors]
        X_reg = reg.loc[:, used_predictors]
        y_field = field.loc[:, 'yield']
        y_field.index = field.field_id
        y_reg = reg.loc[:, 'yield']

        # Define source and target domains
        X_source, y_source = X_reg.copy(), y_reg.copy()
        X_target, y_target = X_field.copy(), y_field.copy()

        if field2reg:
            X_source, y_source = X_field.copy(), y_field.copy()
            X_target, y_target = X_reg.copy(), y_reg.copy()

        # Create holdout set for reg2reg evaluation (from source domain)
        X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
            X_source, y_source, train_size=0.8, random_state=random_state
        )

        # Setup results dataframes
        col_names = ['reg_train', 'reg_test', 'tl_train', 'tl_test', 'field_train',
                     'field_test', 'reg2reg_test', 'xgb_train', 'xgb_test']

        if cv_method == 'loocv':
            years_1 = np.unique(years_reg)
            years_2 = np.unique(years_field)
            years_u = list(set(years_1).intersection(years_2))
            cv = len(years_u)
            results = pd.DataFrame(index=years_u, columns=col_names)
            results_values = pd.DataFrame(columns=['year', 'yield_forecast', 'yield_obs', 'field_id'])
        else:
            results = pd.DataFrame(index=range(cv), columns=col_names)

        results_rmse = results.copy()

        # Initialize XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.01,
            colsample_bytree=0.3,
            random_state=random_state
        )

        # Cross-validation loop
        for i_cv in range(cv):
            # Split data based on CV method
            if cv_method == 'random':
                X_transfer, X_test, y_transfer, y_test = train_test_split(
                    X_target, y_target, train_size=0.8, random_state=random_state + i_cv
                )

                # Subsample transfer data if requested
                if perc_tl_samples < 1:
                    if not_random:
                        # Select lowest-yielding samples
                        n_samples = int(np.round(len(y_transfer) * perc_tl_samples))
                        y_transfer_named = y_transfer.rename('crop_yield')
                        combined = pd.concat([X_transfer, y_transfer_named], axis=1)
                        combined = combined.sort_values('crop_yield', axis=0)
                        X_transfer = combined.iloc[:n_samples, :-1]
                        y_transfer = combined.iloc[:n_samples, -1]
                    else:
                        # Random subsample
                        X_transfer, _, y_transfer, _ = train_test_split(
                            X_transfer, y_transfer,
                            train_size=perc_tl_samples,
                            random_state=random_state + i_cv + 1000
                        )

            elif cv_method == 'loocv':
                year = years_u[i_cv]
                year_test = np.where(years_field == year)[0] if not field2reg else np.where(years_reg == year)[0]
                year_transfer = np.where(years_field != year)[0] if not field2reg else np.where(years_reg != year)[0]
                year_train = np.where(years_reg != year)[0] if not field2reg else np.where(years_field != year)[0]

                X_source_train = X_source.iloc[year_train, :]
                y_source_train = y_source.iloc[year_train]
                X_transfer = X_target.iloc[year_transfer, :]
                y_transfer = y_target.iloc[year_transfer]
                X_test = X_target.iloc[year_test, :]
                y_test = y_target.iloc[year_test]
            else:
                raise ValueError(f'cv_method: {cv_method} not available. Please choose either random or loocv')

            # ===== STANDARDIZATION =====
            if standardize:
                # Fit scalers ONLY on source training data
                scaler_x = StandardScaler().fit(X_source_train)
                scaler_y = StandardScaler().fit(y_source_train.values.reshape(-1, 1))

                # Transform all datasets using source training statistics
                X_source_train_scaled = pd.DataFrame(
                    scaler_x.transform(X_source_train),
                    index=X_source_train.index,
                    columns=X_source_train.columns
                )
                X_source_test_scaled = pd.DataFrame(
                    scaler_x.transform(X_source_test),
                    index=X_source_test.index,
                    columns=X_source_test.columns
                )
                X_transfer_scaled = pd.DataFrame(
                    scaler_x.transform(X_transfer),
                    index=X_transfer.index,
                    columns=X_transfer.columns
                )
                X_test_scaled = pd.DataFrame(
                    scaler_x.transform(X_test),
                    index=X_test.index,
                    columns=X_test.columns
                )

                y_source_train_scaled = pd.Series(
                    scaler_y.transform(y_source_train.values.reshape(-1, 1)).flatten(),
                    index=y_source_train.index
                )
                y_source_test_scaled = pd.Series(
                    scaler_y.transform(y_source_test.values.reshape(-1, 1)).flatten(),
                    index=y_source_test.index
                )
                y_transfer_scaled = pd.Series(
                    scaler_y.transform(y_transfer.values.reshape(-1, 1)).flatten(),
                    index=y_transfer.index
                )
                y_test_scaled = pd.Series(
                    scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(),
                    index=y_test.index
                )

                # Use scaled data for training
                X_train_use = X_source_train_scaled
                y_train_use = y_source_train_scaled
                X_source_test_use = X_source_test_scaled
                y_source_test_use = y_source_test_scaled
                X_transfer_use = X_transfer_scaled
                y_transfer_use = y_transfer_scaled
                X_test_use = X_test_scaled
                y_test_use = y_test_scaled
            else:
                # Use original data
                X_train_use = X_source_train
                y_train_use = y_source_train
                X_source_test_use = X_source_test
                y_source_test_use = y_source_test
                X_transfer_use = X_transfer
                y_transfer_use = y_transfer
                X_test_use = X_test
                y_test_use = y_test

            # ===== PHASE 1: Train on source domain =====
            pipe = dl_old(
                X_train_use.shape[1:],
                hidden_layer_sizes,
                epochs,
                batch_size,
                dropout_perc,
                optimizer,
                early_stopping_patience
            )
            pipe.fit(X_train_use, y_train_use)

            # Save initial model (first CV iteration only)
            if i_cv == 0:
                pipe.model_.save(os.path.join(self.path_out, f'{self.crop}_{lead_time}_init_model.keras'))

            # Evaluate on source and target domains before transfer learning
            y_fore_train = pipe.predict(X_train_use)
            y_fore_test = pipe.predict(X_test_use)
            y_fore_source_test = pipe.predict(X_source_test_use)

            # Inverse transform predictions if standardized
            if standardize:
                y_fore_train = scaler_y.inverse_transform(y_fore_train.reshape(-1, 1)).flatten()
                y_fore_test = scaler_y.inverse_transform(y_fore_test.reshape(-1, 1)).flatten()
                y_fore_source_test = scaler_y.inverse_transform(y_fore_source_test.reshape(-1, 1)).flatten()

                # Use original (unscaled) targets for metrics
                y_train_metric = y_source_train
                y_test_metric = y_test
                y_source_test_metric = y_source_test
            else:
                y_train_metric = y_train_use
                y_test_metric = y_test_use
                y_source_test_metric = y_source_test_use

            # Store results
            ind_i = years_u[i_cv] if cv_method == 'loocv' else i_cv

            results.loc[ind_i, 'reg_train'] = r2_score(y_train_metric, y_fore_train)
            results.loc[ind_i, 'reg_test'] = r2_score(y_test_metric, y_fore_test)
            results.loc[ind_i, 'reg2reg_test'] = r2_score(y_source_test_metric, y_fore_source_test)
            results_rmse.loc[ind_i, 'reg_train'] = root_mean_squared_error(y_fore_train, y_train_metric)/np.mean(y_train_metric)
            results_rmse.loc[ind_i, 'reg_test'] = root_mean_squared_error(y_fore_test, y_test_metric)/np.mean(y_test_metric)
            results_rmse.loc[ind_i, 'reg2reg_test'] = root_mean_squared_error(y_fore_source_test, y_source_test_metric)/np.mean(y_source_test_metric)

            # ===== PHASE 2: Transfer learning - freeze layers and fine-tune =====
            # Identify Dense layers to freeze
            dense_layers = [i for i, layer in enumerate(pipe.model_.layers) if isinstance(layer, Dense)]
            num_layers_to_freeze = len(dense_layers) - tl_train_layers

            for i in range(num_layers_to_freeze):
                layer_idx = dense_layers[i]
                pipe.model_.layers[layer_idx].trainable = False

            # Recompile model after changing trainable status
            if optimizer == 'sgd':
                sgd_opt = SGD(learning_rate=0.01, momentum=0.9)
                pipe.model_.compile(
                    loss='mean_squared_error',
                    optimizer=sgd_opt,
                    metrics=[RootMeanSquaredError()]
                )
            elif optimizer == 'adam':
                pipe.model_.compile(
                    loss='mean_squared_error',
                    optimizer='adam',
                    metrics=[RootMeanSquaredError()]
                )

            # Fine-tune on transfer data
            pipe.fit(X_transfer_use, y_transfer_use)

            # Save transfer learning model
            pipe.model_.save(os.path.join(self.path_out, f'{self.crop}_{lead_time}_transfer_model.keras'))

            # Evaluate after transfer learning
            y_fore_transfer = pipe.predict(X_transfer_use)
            y_fore_test_tl = pipe.predict(X_test_use)

            # Inverse transform if standardized
            if standardize:
                y_fore_transfer = scaler_y.inverse_transform(y_fore_transfer.reshape(-1, 1)).flatten()
                y_fore_test_tl = scaler_y.inverse_transform(y_fore_test_tl.reshape(-1, 1)).flatten()
                y_transfer_metric = y_transfer
            else:
                y_transfer_metric = y_transfer_use

            results.loc[ind_i, 'tl_train'] = r2_score(y_transfer_metric, y_fore_transfer)
            results.loc[ind_i, 'tl_test'] = r2_score(y_test_metric, y_fore_test_tl)
            results_rmse.loc[ind_i, 'tl_train'] = root_mean_squared_error(y_fore_transfer, y_transfer_metric)/np.mean(y_transfer_metric)
            results_rmse.loc[ind_i, 'tl_test'] = root_mean_squared_error(y_fore_test_tl, y_test_metric)/np.mean(y_test_metric)

            if cv_method == 'loocv':
                new_rows = pd.DataFrame({
                    'year': [years_u[i_cv]] * len(y_fore_test_tl),
                    'yield_forecast': y_fore_test_tl,
                    'yield_obs': y_test_metric.values,
                    'field_id': y_test.index
                })
                results_values = pd.concat([results_values, new_rows], ignore_index=True)

            del pipe

            # ===== PHASE 3: Train from scratch on target domain (baseline) =====
            pipe_scratch = dl_old(
                X_transfer_use.shape[1:],
                hidden_layer_sizes,
                epochs,
                batch_size,
                dropout_perc,
                optimizer,
                early_stopping_patience
            )
            pipe_scratch.fit(X_transfer_use, y_transfer_use)

            y_fore_field_train = pipe_scratch.predict(X_transfer_use)
            y_fore_field_test = pipe_scratch.predict(X_test_use)

            if standardize:
                y_fore_field_train = scaler_y.inverse_transform(y_fore_field_train.reshape(-1, 1)).flatten()
                y_fore_field_test = scaler_y.inverse_transform(y_fore_field_test.reshape(-1, 1)).flatten()

            results.loc[ind_i, 'field_train'] = r2_score(y_transfer_metric, y_fore_field_train)
            results.loc[ind_i, 'field_test'] = r2_score(y_test_metric, y_fore_field_test)
            results_rmse.loc[ind_i, 'field_train'] = root_mean_squared_error(y_fore_field_train, y_transfer_metric)/np.mean(y_transfer_metric)
            results_rmse.loc[ind_i, 'field_test'] = root_mean_squared_error(y_fore_field_test, y_test_metric)/np.mean(y_test_metric)

            del pipe_scratch

            # ===== PHASE 4: XGBoost baseline =====
            # XGBoost uses original (unscaled) data
            xgb_model.fit(X_transfer, y_transfer)
            y_fore_xgb_train = xgb_model.predict(X_transfer)
            y_fore_xgb_test = xgb_model.predict(X_test)

            results.loc[ind_i, 'xgb_train'] = r2_score(y_transfer, y_fore_xgb_train)
            results.loc[ind_i, 'xgb_test'] = r2_score(y_test, y_fore_xgb_test)
            results_rmse.loc[ind_i, 'xgb_train'] = root_mean_squared_error(y_fore_xgb_train, y_transfer)/np.mean(y_transfer)
            results_rmse.loc[ind_i, 'xgb_test'] = root_mean_squared_error(y_fore_xgb_test, y_test)/np.mean(y_test)

        # ===== Save results =====
        # Save configuration file
        if not os.path.exists(os.path.join(self.path_out, 'configs.txt')):
            with open(os.path.join(self.path_out, 'configs.txt'), 'w') as file:
                file.write(
                    f'used configs:\n'
                    f'field2reg={field2reg}\n'
                    f'tl_train_layers={tl_train_layers}\n'
                    f'fraction_tl_samples={perc_tl_samples}\n'
                    f'hidden_layer_sizes={hidden_layer_sizes}\n'
                    f'epochs={epochs}\n'
                    f'batch_size={batch_size}\n'
                    f'dropout_perc={dropout_perc}\n'
                    f'optimizer={optimizer}\n'
                    f'early_stopping_patience={early_stopping_patience}\n'
                    f'cv={cv}\n'
                    f'cv_method={cv_method}\n'
                    f'standardized={standardize}\n'
                    f'random_state={random_state}'
                )

        # Save results to CSV
        if perc_tl_samples is not None:
            results.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_{np.round(perc_tl_samples, 1)}_r2.csv')
            results_rmse.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_{np.round(perc_tl_samples, 1)}_nrmse.csv')
        else:
            results.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_0.8_r2.csv')
            results_rmse.to_csv(f'{self.path_out}/{self.crop}_{lead_time}_0.8_nrmse.csv')

        if cv_method == 'loocv':
            results_values.to_csv(f'{self.path_out}/{self.crop}_values_{lead_time}_1.csv', index=False)

    def transferlearn_refactored(self, lead_time, field2reg, tl_train_layers, hidden_layer_sizes=[100, 50, 50, 1],
                                 epochs=100, batch_size: int = 10, dropout_perc: float = 0.3,
            optimizer: str = 'adam', early_stopping_patience: int = 3, cv: int = 30, meteo: bool = False,
            standardize: bool = False, random_state: int = 42):
        """
        Transfer learning for crop yield forecasting with 30-fold cross-validation

        This refactored version fixes all bugs while maintaining the exact same output format.

        Args:
            lead_time: Number of months before harvest for forecast
            field2reg: If True, transfer from field to regional (reversed)
            tl_train_layers: Number of layers to fine-tune during transfer learning
            hidden_layer_sizes: List of neurons per layer (default: [100, 50, 50, 1])
            epochs: Number of training epochs
            batch_size: Batch size for training
            dropout_perc: Dropout percentage (0-1)
            optimizer: 'adam' or 'sgd'
            early_stopping_patience: Early stopping patience
            cv: Number of cross-validation folds
            meteo: If True, use meteorological data
            standardize: If True, standardize features and targets
            random_state: Random seed for reproducibility

        Returns:
            None (saves results to CSV files)

        Output Files:
            - {crop}_{lead_time}_r2.csv: R² scores for all models
            - {crop}_{lead_time}_nrmse.csv: Normalized RMSE for all models
            - {crop}_{lead_time}_config.txt: Configuration used
        """
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [100, 50, 50, 1]

        print(f"Starting transfer learning experiment: {self.crop}, lead_time={lead_time}")

        # ===== Load Data =====
        if meteo:
            reg = pd.read_csv(f'Data/M/TL/meteo/{self.crop}_regional.csv')
            field = pd.read_csv(f'Data/M/TL/meteo/{self.crop}_field.csv')
        else:
            reg = pd.read_csv(f'Data/M/TL/cum/{self.crop}_regional.csv')
            field = pd.read_csv(f'Data/M/TL/cum/{self.crop}_field.csv')

        # Remove specific fields
        if self.crop == 'maize':
            reg = reg[reg.field_id != 'CZ064']
        else:
            reg = reg[reg.field_id != 'CZ0646000000']

        # Clean data
        reg = reg.dropna(axis=0)
        field = field.dropna(axis=0)

        # Select predictors based on lead time
        predictors = reg.columns[3:]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]

        X_regional = reg.loc[:, used_predictors].reset_index(drop=True)
        y_regional = reg.loc[:, 'yield'].reset_index(drop=True)
        X_field = field.loc[:, used_predictors].reset_index(drop=True)
        y_field = field.loc[:, 'yield'].reset_index(drop=True)

        print(f"Regional samples: {len(X_regional)}, Field samples: {len(X_field)}")
        print(f"Features: {len(used_predictors)}")

        # Define source and target domains
        if field2reg:
            X_source, y_source = X_field.copy(), y_field.copy()
            X_target, y_target = X_regional.copy(), y_regional.copy()
            print("Transfer direction: Field → Regional")
        else:
            X_source, y_source = X_regional.copy(), y_regional.copy()
            X_target, y_target = X_field.copy(), y_field.copy()
            print("Transfer direction: Regional → Field")

        # ===== Initialize Results Storage =====
        col_names = ['reg_train', 'reg_test', 'tl_train', 'tl_test', 'field_train',
                     'field_test', 'reg2reg_test', 'xgb_train', 'xgb_test']

        results_r2 = pd.DataFrame(index=range(cv), columns=col_names)
        results_nrmse = pd.DataFrame(index=range(cv), columns=col_names)

        # ===== Setup Cross-Validation =====
        kf_source = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        kf_target = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        # ===== Cross-Validation Loop =====
        for fold, ((source_train_idx, source_test_idx), (target_train_idx, target_test_idx)) in enumerate(
                zip(kf_source.split(X_source), kf_target.split(X_target))
        ):
            print(f"  Processing fold {fold + 1}/{cv}", end='\r')

            # Split source domain (regional)
            X_source_train = X_source.iloc[source_train_idx].copy()
            y_source_train = y_source.iloc[source_train_idx].copy()
            X_source_test = X_source.iloc[source_test_idx].copy()
            y_source_test = y_source.iloc[source_test_idx].copy()

            # Split target domain (field)
            X_target_train = X_target.iloc[target_train_idx].copy()
            y_target_train = y_target.iloc[target_train_idx].copy()
            X_target_test = X_target.iloc[target_test_idx].copy()
            y_target_test = y_target.iloc[target_test_idx].copy()

            # Store original targets for normalization
            y_source_train_orig = y_source_train.copy()
            y_source_test_orig = y_source_test.copy()
            y_target_train_orig = y_target_train.copy()
            y_target_test_orig = y_target_test.copy()

            # ===== Standardization (FIT ONLY ON SOURCE TRAINING DATA) =====
            if standardize:
                # Fit scalers on source training data
                scaler_x = StandardScaler().fit(X_source_train)
                scaler_y = StandardScaler().fit(y_source_train.values.reshape(-1, 1))

                # Transform all datasets
                X_source_train = pd.DataFrame(
                    scaler_x.transform(X_source_train),
                    columns=X_source_train.columns
                )
                X_source_test = pd.DataFrame(
                    scaler_x.transform(X_source_test),
                    columns=X_source_test.columns
                )
                X_target_train = pd.DataFrame(
                    scaler_x.transform(X_target_train),
                    columns=X_target_train.columns
                )
                X_target_test = pd.DataFrame(
                    scaler_x.transform(X_target_test),
                    columns=X_target_test.columns
                )

                y_source_train = pd.Series(
                    scaler_y.transform(y_source_train.values.reshape(-1, 1)).flatten()
                )
                y_source_test = pd.Series(
                    scaler_y.transform(y_source_test.values.reshape(-1, 1)).flatten()
                )
                y_target_train = pd.Series(
                    scaler_y.transform(y_target_train.values.reshape(-1, 1)).flatten()
                )
                y_target_test = pd.Series(
                    scaler_y.transform(y_target_test.values.reshape(-1, 1)).flatten()
                )

            # ===== MODEL 1: Regional/Source Model (reg_train, reg_test, reg2reg_test) =====
            model_source = _create_keras_regressor(
                X_source_train.shape[1], hidden_layer_sizes, epochs,
                batch_size, dropout_perc, optimizer, early_stopping_patience
            )
            model_source.fit(X_source_train, y_source_train)

            # Predictions
            y_pred_source_train = model_source.predict(X_source_train)
            y_pred_source_test = model_source.predict(X_source_test)
            y_pred_target_test_notl = model_source.predict(X_target_test)

            # Inverse transform if standardized
            if standardize:
                y_pred_source_train = scaler_y.inverse_transform(y_pred_source_train.reshape(-1, 1)).flatten()
                y_pred_source_test = scaler_y.inverse_transform(y_pred_source_test.reshape(-1, 1)).flatten()
                y_pred_target_test_notl = scaler_y.inverse_transform(y_pred_target_test_notl.reshape(-1, 1)).flatten()

            # Calculate metrics
            results_r2.loc[fold, 'reg_train'] = r2_score(y_source_train_orig, y_pred_source_train)
            results_r2.loc[fold, 'reg2reg_test'] = r2_score(y_source_test_orig, y_pred_source_test)
            results_r2.loc[fold, 'reg_test'] = r2_score(y_target_test_orig, y_pred_target_test_notl)

            results_nrmse.loc[fold, 'reg_train'] = _normalize_rmse(
                root_mean_squared_error(y_source_train_orig, y_pred_source_train),
                y_source_train_orig.values
            )
            results_nrmse.loc[fold, 'reg2reg_test'] = _normalize_rmse(
                root_mean_squared_error(y_source_test_orig, y_pred_source_test),
                y_source_test_orig.values
            )
            results_nrmse.loc[fold, 'reg_test'] = _normalize_rmse(
                root_mean_squared_error(y_target_test_orig, y_pred_target_test_notl),
                y_target_test_orig.values
            )

            # Save initial model (first fold only)
            if fold == 0:
                model_source.model_.save(
                    os.path.join(self.path_out, f'{self.crop}_{lead_time}_init_model.keras')
                )

            # ===== MODEL 2: Transfer Learning (tl_train, tl_test) =====
            model_tl = _create_keras_regressor(
                X_source_train.shape[1], hidden_layer_sizes, epochs,
                batch_size, dropout_perc, optimizer, early_stopping_patience
            )

            # Phase 1: Train on source
            model_tl.fit(X_source_train, y_source_train)

            # Phase 2: Freeze layers
            dense_layers = [
                i for i, layer in enumerate(model_tl.model_.layers)
                if isinstance(layer, Dense)
            ]
            n_layers_to_freeze = len(dense_layers) - tl_train_layers

            for i in range(n_layers_to_freeze):
                model_tl.model_.layers[dense_layers[i]].trainable = False

            # Recompile
            if optimizer == 'sgd':
                opt = SGD(learning_rate=0.01, momentum=0.9)
            else:
                opt = 'adam'

            model_tl.model_.compile(
                loss='mean_squared_error',
                optimizer=opt,
                metrics=[RootMeanSquaredError()]
            )

            # Phase 3: Fine-tune on target
            model_tl.fit(X_target_train, y_target_train)

            # Predictions
            y_pred_target_train_tl = model_tl.predict(X_target_train)
            y_pred_target_test_tl = model_tl.predict(X_target_test)

            # Inverse transform if standardized
            if standardize:
                y_pred_target_train_tl = scaler_y.inverse_transform(y_pred_target_train_tl.reshape(-1, 1)).flatten()
                y_pred_target_test_tl = scaler_y.inverse_transform(y_pred_target_test_tl.reshape(-1, 1)).flatten()

            # Calculate metrics
            results_r2.loc[fold, 'tl_train'] = r2_score(y_target_train_orig, y_pred_target_train_tl)
            results_r2.loc[fold, 'tl_test'] = r2_score(y_target_test_orig, y_pred_target_test_tl)

            results_nrmse.loc[fold, 'tl_train'] = _normalize_rmse(
                root_mean_squared_error(y_target_train_orig, y_pred_target_train_tl),
                y_target_train_orig.values
            )
            results_nrmse.loc[fold, 'tl_test'] = _normalize_rmse(
                root_mean_squared_error(y_target_test_orig, y_pred_target_test_tl),
                y_target_test_orig.values
            )

            # Save transfer model (first fold only)
            if fold == 0:
                model_tl.model_.save(
                    os.path.join(self.path_out, f'{self.crop}_{lead_time}_transfer_model.keras')
                )

            del model_tl

            # ===== MODEL 3: Field/Target Only (field_train, field_test) =====
            model_target = _create_keras_regressor(
                X_target_train.shape[1], hidden_layer_sizes, epochs,
                batch_size, dropout_perc, optimizer, early_stopping_patience
            )
            model_target.fit(X_target_train, y_target_train)

            # Predictions
            y_pred_target_train = model_target.predict(X_target_train)
            y_pred_target_test = model_target.predict(X_target_test)

            # Inverse transform if standardized
            if standardize:
                y_pred_target_train = scaler_y.inverse_transform(y_pred_target_train.reshape(-1, 1)).flatten()
                y_pred_target_test = scaler_y.inverse_transform(y_pred_target_test.reshape(-1, 1)).flatten()

            # Calculate metrics
            results_r2.loc[fold, 'field_train'] = r2_score(y_target_train_orig, y_pred_target_train)
            results_r2.loc[fold, 'field_test'] = r2_score(y_target_test_orig, y_pred_target_test)

            results_nrmse.loc[fold, 'field_train'] = _normalize_rmse(
                root_mean_squared_error(y_target_train_orig, y_pred_target_train),
                y_target_train_orig.values
            )
            results_nrmse.loc[fold, 'field_test'] = _normalize_rmse(
                root_mean_squared_error(y_target_test_orig, y_pred_target_test),
                y_target_test_orig.values
            )

            del model_target

            # ===== MODEL 4: XGBoost (xgb_train, xgb_test) =====
            # XGBoost uses original (unscaled) data
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.01,
                colsample_bytree=0.3,
                random_state=random_state
            )

            xgb_model.fit(X_target_train if not standardize else scaler_x.inverse_transform(X_target_train),
                          y_target_train_orig)

            y_pred_xgb_train = xgb_model.predict(
                X_target_train if not standardize else scaler_x.inverse_transform(X_target_train)
            )
            y_pred_xgb_test = xgb_model.predict(
                X_target_test if not standardize else scaler_x.inverse_transform(X_target_test)
            )

            # Calculate metrics
            results_r2.loc[fold, 'xgb_train'] = r2_score(y_target_train_orig, y_pred_xgb_train)
            results_r2.loc[fold, 'xgb_test'] = r2_score(y_target_test_orig, y_pred_xgb_test)

            results_nrmse.loc[fold, 'xgb_train'] = _normalize_rmse(
                root_mean_squared_error(y_target_train_orig, y_pred_xgb_train),
                y_target_train_orig.values
            )
            results_nrmse.loc[fold, 'xgb_test'] = _normalize_rmse(
                root_mean_squared_error(y_target_test_orig, y_pred_xgb_test),
                y_target_test_orig.values
            )

        print(f"\n  Completed {cv} folds")

        # ===== Save Results =====
        # Convert to numeric
        results_r2 = results_r2.apply(pd.to_numeric)
        results_nrmse = results_nrmse.apply(pd.to_numeric)

        # Save CSV files
        results_r2.to_csv(
            os.path.join(self.path_out, f'{self.crop}_{lead_time}_r2.csv'),
            index=False
        )
        results_nrmse.to_csv(
            os.path.join(self.path_out, f'{self.crop}_{lead_time}_nrmse.csv'),
            index=False
        )

        # Save configuration
        config_path = os.path.join(self.path_out, f'{self.crop}_{lead_time}_config.txt')
        with open(config_path, 'w') as f:
            f.write("=== Transfer Learning Configuration ===\n")
            f.write(f"Crop: {self.crop}\n")
            f.write(f"Lead time: {lead_time}\n")
            f.write(f"Transfer direction: {'Field → Regional' if field2reg else 'Regional → Field'}\n")
            f.write(f"Layers to fine-tune: {tl_train_layers}\n")
            f.write(f"Hidden layer sizes: {hidden_layer_sizes}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Dropout: {dropout_perc}\n")
            f.write(f"Optimizer: {optimizer}\n")
            f.write(f"Early stopping patience: {early_stopping_patience}\n")
            f.write(f"Cross-validation folds: {cv}\n")
            f.write(f"Standardized: {standardize}\n")
            f.write(f"Meteorological data: {meteo}\n")
            f.write(f"Random state: {random_state}\n\n")

            f.write("=== Results Summary ===\n")
            f.write("\nR² Scores (mean ± std):\n")
            for col in col_names:
                f.write(f"  {col}: {results_r2[col].mean():.4f} ± {results_r2[col].std():.4f}\n")

            f.write("\nNormalized RMSE (mean ± std):\n")
            for col in col_names:
                f.write(f"  {col}: {results_nrmse[col].mean():.4f} ± {results_nrmse[col].std():.4f}\n")

        # Print summary
        print("\n=== Results Summary ===")
        print("\nR² Scores:")
        print(results_r2.mean().to_string())
        print("\nNormalized RMSE:")
        print(results_nrmse.mean().to_string())

        print(f"\nResults saved to:")
        print(f"  - {self.crop}_{lead_time}_r2.csv")
        print(f"  - {self.crop}_{lead_time}_nrmse.csv")
        print(f"  - {self.crop}_{lead_time}_config.txt")

    def feature_importance_tl(self):
        reg = pd.read_csv(f'Data/M/TL/{self.crop}_regional.csv')
        field = pd.read_csv(f'Data/M/TL/{self.crop}_field.csv')

        preds = ('sig40', 'evi', 'ndwi', 'nmdi')

        predictors = reg.columns[3:]
        used_predictors = [a for a in predictors if a.startswith(preds)]

        reg = reg.dropna(axis=0)
        field = field.dropna(axis=0)

        X_field = field.loc[:, used_predictors]
        X_reg = reg.loc[:, used_predictors]
        y_field, y_reg = field.loc[:, 'yield'], reg.loc[:, 'yield']

        X_train, y_train, X_test_o, y_test_o = X_reg, y_reg, X_field, y_field
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8)
        X_transfer, X_test, y_transfer, y_test = train_test_split(X_test_o, y_test_o, train_size=0.8)
        X_transfer, X_transfer_cal, y_transfer, y_transfer_cal = train_test_split(X_transfer, y_transfer, train_size=0.8)

        y_train = y_train.values.reshape(-1, 1)
        y_val = y_val.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        y_transfer = y_transfer.values.reshape(-1, 1)
        y_transfer_cal = y_transfer_cal.values.reshape(-1, 1)

        scaler_xtrain = StandardScaler().fit(X_train)
        scaler_xtest = StandardScaler().fit(X_test)
        scaler_xval = StandardScaler().fit(X_val)
        scaler_xtransfer = StandardScaler().fit(X_transfer)
        scaler_xtransfer_cal = StandardScaler().fit(X_transfer_cal)

        scaler_ytrain = StandardScaler().fit(y_train)
        scaler_yval = StandardScaler().fit(y_val)
        scaler_ytest = StandardScaler().fit(y_test)
        scaler_ytransfer = StandardScaler().fit(y_transfer)
        scaler_ytransfer_cal = StandardScaler().fit(y_transfer_cal)

        X_train = scaler_xtrain.transform(X_train)
        X_val = scaler_xval.transform(X_val)
        X_test = scaler_xtest.transform(X_test)
        X_transfer = scaler_xtransfer.transform(X_transfer)
        X_transfer_cal = scaler_xtransfer_cal.transform(X_transfer_cal)

        y_train = scaler_ytrain.transform(y_train).flatten()
        y_val = scaler_yval.transform(y_val).flatten()
        y_test = scaler_ytest.transform(y_test).flatten()
        y_transfer = scaler_ytransfer.transform(y_transfer).flatten()
        y_transfer_cal = scaler_ytransfer_cal.transform(y_transfer_cal).flatten()

        feature_names = [a.replace('mean_daily_', '') for a in used_predictors]

        # Keras transfer learning example
        keras_model_builder = create_keras_model(X_train.shape[1], 'regression')

        transfer_cv = TransferLearningPermutationCV(
            model_builder=keras_model_builder,
            cv=30,  # Reduced for faster demo
            n_repeats=5,
            random_state=42,
            task_type='regression'
        )

        # Fit with transfer learning
        transfer_cv.fit_transfer_learning(
            X_train, y_train,
            X_transfer, y_transfer,
            feature_names=feature_names,
            original_fit_kwargs={'epochs': 50, 'batch_size': 32, 'verbose': 0},
            transfer_fit_kwargs={'epochs': 30, 'batch_size': 16, 'verbose': 0}  # Fine-tuning with fewer epochs
        )

        # Get comparison results
        comparison = transfer_cv.get_feature_ranking('both')
        print("\nTop 10 Features Comparison:")
        print(comparison[['feature', 'original_importance', 'transfer_importance',
                          'importance_change', 'rank_change']].head(10))

        comparison.to_csv(f'Results/Validation/dl/202505_revisions/FI_analysis/{self.crop}_FI.csv')

        # Plot comparisons
        # transfer_cv.plot_transfer_comparison(top_k=12, crop=self.crop)
        # transfer_cv.plot_stability_comparison(top_k=8)

        # Analyze the transfer learning effect
        print("\nTransfer Learning Analysis:")
        print(f"Features with increased importance: {(comparison['importance_change'] > 0).sum()}")
        print(f"Features with decreased importance: {(comparison['importance_change'] < 0).sum()}")
        print(f"Average importance change: {comparison['importance_change'].mean():.4f}")

        # Most improved/declined features
        most_improved = comparison.nlargest(3, 'importance_change')[['feature', 'importance_change']]
        most_declined = comparison.nsmallest(3, 'importance_change')[['feature', 'importance_change']]

        print(f"\nMost improved features after transfer learning:")
        for _, row in most_improved.iterrows():
            print(f"  {row['feature']}: +{row['importance_change']:.4f}")

        print(f"\nMost declined features after transfer learning:")
        for _, row in most_declined.iterrows():
            print(f"  {row['feature']}: {row['importance_change']:.4f}")

        # # Initialize the SHAP explainer
        # explainer = shap.Explainer(model, X_train)
        #
        # # Calculate SHAP values for the validation data
        # shap_values = explainer(X_val)
        # # print(shap_values)
        #
        # # Plot the feature importance
        # shap.summary_plot(shap_values, X_val, feature_names=feature_names)

    def plot_feature_imp(self):
        """
        Run the function self.feature_importance_tl() first to esatblish the files required for this function
        :return: plot showing the feature importance of the model
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fontsize = 16
        for i, crop in enumerate(['maize', 'winter_wheat', 'spring_barley']):
            # Read the CSV file
            df = pd.read_csv(f'Results/Validation/dl/202505_revisions/FI_analysis/{crop}_FI.csv').iloc[:10,:]

            # First barplot: original_importance and transfer_importance with error bars
            ax1 = axes[0, i]
            x = np.arange(len(df))
            x_label = [a.replace('sig40_', '').upper() for a in df.feature]
            width = 0.35

            ax1.bar(x - width / 2, df['original_importance'], width,
                    # yerr=df['original_std'],
                    label='Original Importance',
                    capsize=5, alpha=0.8)
            ax1.bar(x + width / 2, df['transfer_importance'], width,
                    # yerr=df['transfer_std'],
                    label='Transfer Importance',
                    capsize=5, alpha=0.8)
            ax1.set_title(crop.replace('_', ' ').capitalize(), fontsize=fontsize+2)
            if i==0:
                ax1.set_ylabel('Average Feature Importance', fontsize=fontsize)
            else:
                ax1.set_yticks([])
            ax1.set_xticks(x, x_label, rotation=45)
            ax1.set_ylim([-0.22, 0.55])
            ax1.tick_params(labelsize=fontsize)
            ax1.legend(fontsize=fontsize)
            ax1.grid(True, alpha=0.3)

            # Second barplot: difference between original_importance and transfer_importance
            ax2 = axes[1, i]
            diff = df['original_importance'] - df['transfer_importance']
            colors = ['red' if d < 0 else 'blue' for d in diff]
            ax2.bar(x, diff, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            if i==0:
                ax2.set_ylabel('Difference (Original - Transfer)', fontsize=fontsize)
            else:
                ax2.set_yticks([])
            ax2.set_xticks(x, x_label, rotation=45)
            ax2.set_ylim([-0.14, 0.06])
            ax2.tick_params(labelsize=fontsize)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.show()
        plt.savefig('Results/Validation/dl/202505_revisions/FI_analysis/FI_no_err.png', dpi=300)

    # --------------------------------------------- Science Case 2 - Ukraine -----------------------------------------
    def run_eracast(self, lead_time, loocv=None):
        """
        :param lead_time: int from 1 to 4. How many months before harvest the forecast is calculated
        :param loocv: str based on which col splitting for loocv is done. Either c_year or country
        :return: y_test and y_test_pred. list of observed and forecasted crop yields in test dataset
        """
        # Define parameters required for deep learning
        hidden_layer_sizes = [100, 50, 50, 50, 50, 1]
        epochs = 100
        batch_size = 10
        dropout_perc = 0.3
        optimizer = 'adam'
        early_stopping_patience = 3

        # Reading and preparing data
        file = pd.read_csv(f'Data/SC2/{self.temp_res}/{self.crop}_era_eu_abs.csv', index_col=0)
        file = file.dropna(axis=0, how='any')
        file.loc[:, 'country'] = [f[:2] for f in file.field_id]
        file.index = range(len(file.index))

        # Encoding the country information to binary rows
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded_array = encoder.fit_transform(file.loc[:, 'country'].values.reshape(-1,1))
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded_array, columns=[a+'4' for a in np.unique(file.loc[:, 'country'])])
        file = pd.concat([file, one_hot_encoded_df], axis=1).drop('country', axis=1)
        file.loc[:, 'country'] = [f[:2] for f in file.field_id]

        predictors = file.columns[3:-1]

        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        predictor_file = file.loc[:, used_predictors]

        scaler = StandardScaler()
        scaler.fit(predictor_file)
        predictor_file = pd.DataFrame(index=predictor_file.index, columns=predictor_file.columns, data=scaler.transform(predictor_file))

        if loocv is None:
            X, X_test, y, y_test = train_test_split(predictor_file, file.loc[:, 'yield_anom'], test_size=0.2)

            pipe = dl(predictor_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer,
                      early_stopping_patience)
            pipe.fit(X, y)
            y_test_pred = pipe.predict(X_test)
            y_train_pred = pipe.predict(X)
            print(f'train perf for {self.crop} lead_time:{lead_time} R: {pearsonr(y_test_pred, y_test)}')
            print(f'train perf: {mean_absolute_error(y_train_pred, y)/np.mean(y)}, test perf: {mean_absolute_error(y_test_pred, y_test)/np.mean(y_test)}')

            return y_test, y_test_pred
        else:
            years = np.unique(file.loc[:,loocv])
            ret_vals = pd.DataFrame(columns=['year', 'forecasted', 'observed'])
            for year in years:
                pipe = dl(predictor_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer,
                          early_stopping_patience)

                # estimator = self.get_default_regressions('XGB')
                # pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])

                inds = np.where(file.loc[:,loocv] == year)[0]
                inds_not = np.where(file.loc[:,loocv] != year)[0]
                X_test, y_test = predictor_file.iloc[inds, :], file.iloc[inds, 2]
                X, y = predictor_file.iloc[inds_not, :], file.iloc[inds_not, 2]

                pipe.fit(X, y)
                y_test_pred = pipe.predict(X_test)
                y_train_pred = pipe.predict(X)
                print(f'{year} train: {pearsonr(y_train_pred, y)[0]}test perf R: {pearsonr(y_test_pred, y_test)[0]}')

                dict = {'year': [year] * len(y_test),
                        'observed': y_test,
                        'forecasted': y_test_pred
                        }

                df = pd.DataFrame(dict)
                ret_vals = pd.concat([ret_vals, df])
                pipe = []
            ret_vals.to_csv(f'Results/SC2/loocv/{loocv}/{self.crop}_{lead_time}_infoc.csv')

    def eracast_tl(self, path_out, lead_time=1, tl_train_layers=2, standardize=False, abs=False):
        """
        :param lead_time: int from 1 to 4. How many months before harvest the forecast is calculated
        :param loocv: str based on which col splitting for loocv is done. Either c_year or country
        :return: y_test and y_test_pred. list of observed and forecasted crop yields in test dataset
        """
        # Define parameters required for deep learning
        hidden_layer_sizes = [100, 50, 50, 50, 50, 1]
        epochs = 100
        batch_size = 10
        dropout_perc = 0.3
        optimizer = 'adam'
        early_stopping_patience = 3
        preds = ('t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr', 'evi')


        # Reading and preparing data
        if abs:
            file = pd.read_csv(f'Data/SC2/{self.temp_res}/final/{self.crop}_all_abs_fin.csv', index_col=0)
        else:
            file = pd.read_csv(f'Data/SC2/{self.temp_res}/final/{self.crop}_all_fin.csv', index_col=0)
        file = file.dropna(axis=0, how='any')
        file.loc[:, 'country'] = [f[:2] for f in file.field_id]
        predictors = file.columns[3:-1]

        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        used_predictors = [a for a in used_predictors if a.startswith(preds)]
        met_pred = [a for a in used_predictors if not a.startswith('evi')]
        eo_pred = [a for a in used_predictors if a.startswith('evi')]
        predictor_file = file.loc[:, used_predictors]
        # print(predictor_file)
        # print(file.iloc[:, 2])
        if standardize:
            scaler = MinMaxScaler()
            scaler.fit(predictor_file)
            predictor_file = pd.DataFrame(index=predictor_file.index, columns=predictor_file.columns, data=scaler.transform(predictor_file))
        # print(predictor_file)
        # print(file.iloc[:, 2])
        eo_pred_file = predictor_file.loc[:, eo_pred]
        met_pred_file = predictor_file.loc[:, met_pred]

        countries = np.unique(file.loc[:, 'country'])
        ret_vals = pd.DataFrame(columns=['year', 'region', 'forecasted', 'forecasted_tl', 'forecasted_no_tl',
                                         'forecasted_eo', 'forecasted_met', 'observed'])
        # print(file)
        for country in countries:
            years = np.unique(file.iloc[np.where(file.loc[:, 'country'] == country)[0], 1])
            print(years)
            for year in years:
                pipe = dl(predictor_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer,
                          early_stopping_patience)
                pipe_no_tl = dl(predictor_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer,
                          early_stopping_patience)
                pipe_eo = dl(eo_pred_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc,
                                optimizer, early_stopping_patience)
                pipe_met = dl(met_pred_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc,
                                optimizer, early_stopping_patience)

                inds_train = np.where((file.loc[:, 'c_year'] != year) & (file.loc[:, 'country'] != country))[0]
                inds_transfer = np.where((file.loc[:, 'c_year'] != year) & (file.loc[:, 'country'] == country))[0]
                inds_no_transfer = np.where(file.loc[:, 'c_year'] != year)[0]
                inds_test = np.where((file.loc[:, 'c_year'] == year) & (file.loc[:, 'country'] == country))[0]

                X_train, y_train = predictor_file.iloc[inds_train, :], file.iloc[inds_train, 2]
                X_transfer, y_transfer = predictor_file.iloc[inds_transfer, :], file.iloc[inds_transfer, 2]
                X_no_transfer, y_no_transfer = predictor_file.iloc[inds_no_transfer, :], file.iloc[inds_no_transfer, 2]
                X_test, y_test = predictor_file.iloc[inds_test, :], file.iloc[inds_test, 2]
                regions = file.iloc[inds_test, 0]

                X_train_met, X_train_eo = met_pred_file.iloc[inds_train, :], eo_pred_file.iloc[inds_train, :]
                X_transfer_met, X_transfer_eo = met_pred_file.iloc[inds_transfer, :], eo_pred_file.iloc[inds_transfer, :]
                X_test_met, X_test_eo = met_pred_file.iloc[inds_test, :], eo_pred_file.iloc[inds_test, :]

                print(X_train.shape, X_transfer.shape, X_test.shape)
                print(X_train_eo.shape, X_test_eo.shape, X_train_met.shape, X_test_met.shape)

                pipe.fit(X_train, y_train)
                y_test_pred = pipe.predict(X_test)
                y_train_pred = pipe.predict(X_train)

                pipe_no_tl.fit(X_no_transfer, y_no_transfer)
                y_test_pred_no_tl = pipe_no_tl.predict(X_test)
                y_train_pred_no_tl = pipe_no_tl.predict(X_no_transfer)

                pipe_eo.fit(X_train_eo, y_train)
                pipe_met.fit(X_train_met, y_train)

                for i in range(len(hidden_layer_sizes)-tl_train_layers):
                    pipe['mlp'].model.layers[i].trainable = False
                    pipe_eo['mlp'].model.layers[i].trainable = False
                    pipe_met['mlp'].model.layers[i].trainable = False

                pipe.fit(X_transfer, y_transfer)
                y_tl_pred = pipe.predict(X_test)

                pipe_eo.fit(X_transfer_eo, y_transfer)
                pipe_met.fit(X_transfer_met, y_transfer)
                y_eo_pred = pipe_eo.predict(X_test_eo)
                y_met_pred = pipe_met.predict(X_test_met)

                print(f'{year} train: {pearsonr(y_train_pred, y_train)[0]:.3f}test perf R: {pearsonr(y_test_pred, y_test)[0]:.3f} tl: {pearsonr(y_tl_pred, y_test)[0]:.3f}')

                dict = {'year': [year] * len(y_test),
                        'region': regions,
                        'observed': y_test,
                        'forecasted': y_test_pred,
                        'forecasted_no_tl': y_test_pred_no_tl,
                        'forecasted_tl': y_tl_pred,
                        'forecasted_eo': y_eo_pred,
                        'forecasted_met': y_met_pred
                        }

                df = pd.DataFrame(dict)
                ret_vals = pd.concat([ret_vals, df])
                pipe = []
                pipe_no_tl = []
                pipe_eo = []
                pipe_met = []
            ret_vals.to_csv(f'{path_out}/{self.crop}_{lead_time}.csv')

            if not os.path.exists(os.path.join(path_out, 'configs.txt')):
                with open(os.path.join(path_out, 'configs.txt'), 'w') as file_w:
                    file_w.write(
                        f'used configs: tl_train_layers={tl_train_layers}\n hidden_layer_sizes={hidden_layer_sizes}\n '
                        f'epochs={epochs}\n batch_size=' f'{batch_size}\n dropout_perc={dropout_perc}\n '
                        f'optimizer={optimizer}\n early_stopping_patience=' f'{early_stopping_patience}\n'
                        f'standardized={standardize}\n abs={abs}')

    def eracast_tl_ua(self, path_out, lead_time=1, tl_train_layers=5, hidden_layer_sizes=[64, 32, 1], dropout_perc = 0.1):
        """
        :param lead_time: int from 1 to 4. How many months before harvest the forecast is calculated
        :param loocv: str based on which col splitting for loocv is done. Either c_year or country
        :return: y_test and y_test_pred. list of observed and forecasted crop yields in test dataset
        """
        # Define parameters required for deep learning
        # hidden_layer_sizes = [64, 32, 1]
        epochs = 300
        batch_size = 32
        # dropout_perc = 0.1
        optimizer = 'adam'
        early_stopping_patience = 15
        preds = ('t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr', 'evi', 'sm', 'VODCA_CXKu')


        # Reading and preparing data
        file = pd.read_csv(f'Data/SC2/2W/final/{self.crop}_all_abs_fin_year_det.csv', index_col=0)
        file = file.dropna(axis=0, how='any')
        file.loc[:, 'country'] = [f[:2] for f in file.field_id]
        predictors = file.columns[3:-1]

        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        used_predictors = [a for a in used_predictors if a.startswith(preds)]
        met_pred = [a for a in used_predictors if not a.startswith(('evi', 'sm', 'VODCA'))]
        eo_pred = [a for a in used_predictors if a.startswith(('evi', 'sm', 'VODCA'))]
        print(met_pred, eo_pred)
        predictor_file = file.loc[:, used_predictors]

        eo_pred_file = predictor_file.loc[:, eo_pred]
        met_pred_file = predictor_file.loc[:, met_pred]

        countries = ['UA']
        ret_vals = pd.DataFrame(columns=['year', 'region', 'forecasted', 'forecasted_tl', 'forecasted_no_tl',
                                         'forecasted_eo', 'forecasted_met', 'forecasted_xgb', 'forecasted_xgb_all',
                                         'observed'])

        for country in countries:
            years = np.unique(file.iloc[np.where(file.loc[:, 'country'] == country)[0], 1])
            train_perf = []
            train_perf_xgb = []
            for year in years:
                pipe = dl(predictor_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer,
                          early_stopping_patience)
                pipe_no_tl = dl(predictor_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc, optimizer,
                          early_stopping_patience)
                pipe_eo = dl(eo_pred_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc,
                                optimizer, early_stopping_patience)
                pipe_met = dl(met_pred_file.shape[1:], hidden_layer_sizes, epochs, batch_size, dropout_perc,
                                optimizer, early_stopping_patience)

                estimator = self.get_default_regressions('XGB')
                pipe_xgb = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
                pipe_xgb_all = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])

                inds_train = np.where((file.loc[:, 'c_year'] != year) & (file.loc[:, 'country'] != country))[0]
                inds_transfer = np.where((file.loc[:, 'c_year'] != year) & (file.loc[:, 'country'] == country))[0]
                inds_no_transfer = np.where(file.loc[:, 'c_year'] != year)[0]
                inds_test = np.where((file.loc[:, 'c_year'] == year) & (file.loc[:, 'country'] == country))[0]

                X_train, y_train = predictor_file.iloc[inds_train, :].to_numpy(), file.iloc[inds_train, 2].values.reshape(-1, 1)
                X_transfer, y_transfer = predictor_file.iloc[inds_transfer, :].to_numpy(), file.iloc[inds_transfer, 2].values.reshape(-1, 1)
                X_no_transfer, y_no_transfer = predictor_file.iloc[inds_no_transfer, :].to_numpy(), file.iloc[inds_no_transfer, 2].values.reshape(-1, 1)
                X_test, y_test = predictor_file.iloc[inds_test, :].to_numpy(), file.iloc[inds_test, 2].values.reshape(-1, 1)
                regions = file.iloc[inds_test, 0]

                X_train_met, X_train_eo = met_pred_file.iloc[inds_train, :].to_numpy(), eo_pred_file.iloc[inds_train, :].to_numpy()
                X_transfer_met, X_transfer_eo = met_pred_file.iloc[inds_transfer, :].to_numpy(), eo_pred_file.iloc[inds_transfer, :].to_numpy()
                X_test_met, X_test_eo = met_pred_file.iloc[inds_test, :].to_numpy(), eo_pred_file.iloc[inds_test, :].to_numpy()

                print(X_train.shape, X_transfer.shape, X_test.shape)
                print(X_train_eo.shape, X_test_eo.shape, X_train_met.shape, X_test_met.shape)

                # Scale datasets
                base_scaler_X = StandardScaler()
                base_scaler_X_eo = StandardScaler()
                base_scaler_X_met = StandardScaler()
                base_scaler_X_no_trans = StandardScaler()
                base_scaler_y = StandardScaler()
                base_scaler_y_all = StandardScaler()

                y_base_scaled = base_scaler_y.fit_transform(y_train)
                y_base_all_scaled = base_scaler_y_all.fit_transform(y_no_transfer)
                base_scaler_X, X_base_scaled = scale_x(scaler=base_scaler_X, X=X_train, fit=True)
                base_scaler_X_eo, X_base_eo_scaled = scale_x(scaler=base_scaler_X_eo, X=X_train_eo, fit=True)
                base_scaler_X_met, X_base_met_scaled = scale_x(scaler=base_scaler_X_met, X=X_train_met, fit=True)
                base_scaler_X_all, X_base_all_scaled = scale_x(scaler=base_scaler_X_no_trans, X=X_no_transfer, fit=True)

                #Scale test data
                y_test_scaled = base_scaler_y.transform(y_test)
                _, X_test_scaled = scale_x(scaler=base_scaler_X, X=X_test, fit=False)
                _, X_test_scaled_all = scale_x(scaler=base_scaler_X_no_trans, X=X_test, fit=False)
                _, X_test_scaled_eo = scale_x(scaler=base_scaler_X_eo, X=X_test_eo, fit=False)
                _, X_test_scaled_met = scale_x(scaler=base_scaler_X_met, X=X_test_met, fit=False)

                # Scale fine-tuning data using base scalers (important for transfer learning)
                y_finetune_scaled = base_scaler_y.transform(y_transfer)
                _, X_finetune_scaled = scale_x(scaler=base_scaler_X, X=X_transfer, fit=False)
                _, X_finetune_scaled_met = scale_x(scaler=base_scaler_X_met, X=X_transfer_met, fit=False)
                _, X_finetune_scaled_eo = scale_x(scaler=base_scaler_X_eo, X=X_transfer_eo, fit=False)

                pipe.fit(X_base_scaled, y_base_scaled)
                pipe_no_tl.fit(X_base_all_scaled, y_base_all_scaled)
                pipe_xgb.fit(X_finetune_scaled, y_finetune_scaled)
                pipe_xgb_all.fit(X_base_all_scaled, y_base_all_scaled)
                pipe_eo.fit(X_base_eo_scaled, y_base_scaled)
                pipe_met.fit(X_base_met_scaled, y_base_scaled)

                # Make predictions
                predictions_scaled_nft = pipe.predict(X_test_scaled)
                y_test_pred = base_scaler_y.inverse_transform(predictions_scaled_nft)
                # y_test_original_nft = base_scaler_y.inverse_transform(y_test_scaled)

                # Make predictions train
                predictions_scaled_nft_train = pipe.predict(X_base_scaled)
                y_train_pred = base_scaler_y.inverse_transform(predictions_scaled_nft_train)
                # y_train_original_nft = base_scaler_y.inverse_transform(y_train)

                # Make predictions all
                predictions_scaled_all = pipe_no_tl.predict(X_test_scaled_all)
                y_test_pred_no_tl = base_scaler_y_all.inverse_transform(predictions_scaled_all)

                # Same for xgb -------------------------------
                y_xgb_pred = pipe_xgb.predict(X_test)
                y_xgb_all_pred = pipe_xgb_all.predict(X_test)
                y_xgb_pred_train = pipe_xgb.predict(X_transfer)

                if len(hidden_layer_sizes)<tl_train_layers:
                    tl_train_layers=len(hidden_layer_sizes)
                for i in range(len(hidden_layer_sizes)-tl_train_layers):
                    pipe['mlp'].model.layers[i].trainable = False
                    pipe_eo['mlp'].model.layers[i].trainable = False
                    pipe_met['mlp'].model.layers[i].trainable = False

                pipe['mlp'].model.compile(optimizer=Adam(learning_rate=0.00001), loss='mse')
                pipe_eo['mlp'].model.compile(optimizer=Adam(learning_rate=0.00001), loss='mse')
                pipe_met['mlp'].model.compile(optimizer=Adam(learning_rate=0.00001), loss='mse')

                pipe.set_params(mlp__epochs=50, mlp__batch_size=16)
                pipe_eo.set_params(mlp__epochs=50, mlp__batch_size=16)
                pipe_met.set_params(mlp__epochs=50, mlp__batch_size=16)

                pipe.fit(X_finetune_scaled, y_finetune_scaled)
                pipe_eo.fit(X_finetune_scaled_eo, y_finetune_scaled)
                pipe_met.fit(X_finetune_scaled_met, y_finetune_scaled)

                # Make predictions all
                predictions_scaled_nft = pipe.predict(X_test_scaled)
                y_tl_pred = base_scaler_y.inverse_transform(predictions_scaled_nft)

                # Make predictions eo
                predictions_scaled_nft_train = pipe_eo.predict(X_test_scaled_eo)
                y_eo_pred = base_scaler_y.inverse_transform(predictions_scaled_nft_train)

                # Make predictions met
                predictions_scaled_all = pipe_met.predict(X_test_scaled_met)
                y_met_pred = base_scaler_y.inverse_transform(predictions_scaled_all)

                print(f'{year} train: {pearsonr(y_train_pred.ravel(), y_train.ravel())[0]:.3f}test perf R: '
                      f'{pearsonr(y_test_pred.ravel(), y_test.ravel())[0]:.3f} tl: '
                      f'{pearsonr(y_tl_pred.ravel(), y_test.ravel())[0]:.3f}')
                print(f'{year} train: {r2_score(y_train_pred.ravel(), y_train.ravel()):.3f}test perf R: '
                      f'{r2_score(y_test_pred.ravel(), y_test.ravel()):.3f} tl: '
                      f'{r2_score(y_tl_pred.ravel(), y_test.ravel()):.3f}')

                # train_perf.append(pearsonr(y_train_pred, y_train)[0])
                # train_perf_xgb.append(pearsonr(y_xgb_pred_train, y_transfer)[0])

                dict = {'year': [year] * len(y_test),
                        'region': regions,
                        'observed': y_test.ravel(),
                        'forecasted': y_test_pred.ravel(),
                        'forecasted_no_tl': y_test_pred_no_tl.ravel(),
                        'forecasted_tl': y_tl_pred.ravel(),
                        'forecasted_eo': y_eo_pred.ravel(),
                        'forecasted_met': y_met_pred.ravel(),
                        'forecasted_xgb': y_xgb_pred.ravel(),
                        'forecasted_xgb_all': y_xgb_all_pred.ravel()
                        }

                df = pd.DataFrame(dict)
                ret_vals = pd.concat([ret_vals, df])
                pipe = []
                pipe_no_tl = []
                pipe_eo = []
                pipe_met = []
            ret_vals.to_csv(f'{path_out}/{self.crop}_{lead_time}.csv')

            if not os.path.exists(os.path.join(path_out, 'configs.txt')):
                with open(os.path.join(path_out, 'configs.txt'), 'w') as file_w:
                    file_w.write(
                        f'used configs: tl_train_layers={tl_train_layers}\n hidden_layer_sizes={hidden_layer_sizes}\n '
                        f'epochs={epochs}\n batch_size=' f'{batch_size}\n dropout_perc={dropout_perc}\n '
                        f'optimizer={optimizer}\n early_stopping_patience=' f'{early_stopping_patience}')

    def eracast_xgb(self, lead_time=1, standardize=False, abs=True):
        """
        :param lead_time: int from 1 to 4. How many months before harvest the forecast is calculated
        :param loocv: str based on which col splitting for loocv is done. Either c_year or country
        :return: y_test and y_test_pred. list of observed and forecasted crop yields in test dataset
        """
        # Define parameters required for deep learning
        preds = ('t2m', 'tp', 'swvl1', 'pev', 'evavt', 'ssr', 'evi')

        # Reading and preparing data
        if abs:
            file = pd.read_csv(f'Data/SC2/{self.temp_res}/final/{self.crop}_all_abs_fin.csv', index_col=0)
        else:
            file = pd.read_csv(f'Data/SC2/{self.temp_res}/final/{self.crop}_all_fin.csv', index_col=0)
        file = file.dropna(axis=0, how='any')
        file.loc[:, 'country'] = [f[:2] for f in file.field_id]
        predictors = file.columns[3:-1]

        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        used_predictors = [a for a in used_predictors if a.startswith(preds)]
        met_pred = [a for a in used_predictors if not a.startswith('evi')]
        eo_pred = [a for a in used_predictors if a.startswith('evi')]
        predictor_file = file.loc[:, used_predictors]

        if standardize:
            scaler = MinMaxScaler()
            scaler.fit(predictor_file)
            predictor_file = pd.DataFrame(index=predictor_file.index, columns=predictor_file.columns, data=scaler.transform(predictor_file))

        eo_pred_file = predictor_file.loc[:, eo_pred]
        met_pred_file = predictor_file.loc[:, met_pred]

        countries = np.unique(file.loc[:, 'country'])
        ret_vals = pd.DataFrame(columns=['year', 'region', 'forecasted', 'forecasted_eo', 'forecasted_met', 'observed'])
        # print(file)
        for country in countries:
            years = np.unique(file.iloc[np.where(file.loc[:, 'country'] == country)[0], 1])

            for year in years:
                estimator = self.get_default_regressions('XGB')
                pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
                pipe_eo = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
                pipe_met = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])

                inds_train = np.where(file.loc[:, 'c_year'] != year)[0]
                inds_test = np.where((file.loc[:, 'c_year'] == year) & (file.loc[:, 'country'] == country))[0]

                X_train, y_train = predictor_file.iloc[inds_train, :], file.iloc[inds_train, 2]
                X_test, y_test = predictor_file.iloc[inds_test, :], file.iloc[inds_test, 2]
                regions = file.iloc[inds_test, 0]

                X_train_met, X_train_eo = met_pred_file.iloc[inds_train, :], eo_pred_file.iloc[inds_train, :]
                X_test_met, X_test_eo = met_pred_file.iloc[inds_test, :], eo_pred_file.iloc[inds_test, :]

                pipe.fit(X_train, y_train)
                y_test_pred = pipe.predict(X_test)
                y_train_pred = pipe.predict(X_train)

                pipe_met.fit(X_train_met, y_train)
                y_met_pred = pipe_met.predict(X_test_met)

                pipe_eo.fit(X_train_eo, y_train)
                y_eo_pred = pipe_eo.predict(X_test_eo)

                print(f'{self.crop} {country} {year} train: {pearsonr(y_train_pred, y_train)[0]:.3f}test perf R: '
                      f'{pearsonr(y_test_pred, y_test)[0]:.3f} met: {pearsonr(y_met_pred, y_test)[0]:.3f}'
                      f' eo: {pearsonr(y_eo_pred, y_test)[0]:.3f}'
                      )

                # scores_kf_fe = cross_val_score(pipe, X_train, y_train, cv=20, scoring="explained_variance")
                # print(f'{self.crop} {country} {year} R^2: {np.median(scores_kf_fe)}')
                # hp_tuned_vals = self.hyper_tune(X_train, y_train, model='XGB')
                # estimator.set_params(**hp_tuned_vals)
                #
                # scores_kf_hp = cross_val_score(pipe, X_train, y_train, cv=20, scoring="explained_variance")
                #
                # print(f'R^2 hp tuned: {np.median(scores_kf_hp)}')


            #     dict = {'year': [year] * len(y_test),
            #             'region': regions,
            #             'observed': y_test,
            #             'forecasted': y_test_pred,
            #             'forecasted_eo': y_eo_pred,
            #             'forecasted_met': y_met_pred
            #             }
            #
            #     df = pd.DataFrame(dict)
            #     ret_vals = pd.concat([ret_vals, df])
            #     pipe = []
            #     pipe_eo = []
            #     pipe_met = []
            #
            # ret_vals.to_csv(f'{path_out}/{self.crop}_{lead_time}.csv')
            #
            # if not os.path.exists(os.path.join(path_out, 'configs.txt')):
            #     with open(os.path.join(path_out, 'configs.txt'), 'w') as file_w:
            #         file_w.write(
            #             f'used configs: standardized={standardize}\n abs={abs}, model=XGB')

    def plot_loocv_dl(self, c=True, error=False):
        if c:
            # a = pd.read_csv(f'Results/SC2/loocv/country/{self.crop}_1_infoc.csv', index_col=0)
            a = pd.read_csv(f'Results/SC2/loocv/UA/{self.crop}_1.csv', index_col=0)
        else:
            a = pd.read_csv(f'Results/SC2/loocv/country/{self.crop}_1.csv', index_col=0)

        res = pd.DataFrame(index=np.unique(a.year), columns=['test', 'tl'])

        # plt.scatter(a.forecasted, a.observed)
        # line = np.linspace(0, 20, 400)
        # plt.plot(line, line, ':k', alpha=0.8)
        # min, max = np.min(a.observed), np.max(a.observed)
        # plt.xlim(min, max)
        # plt.ylim(min, max)
        # plt.xlabel('Forcasted yield [t/ha]')
        # plt.ylabel('Observed yield [t/ha]')
        # plt.title('Maize')
        # plt.show()
        for year in np.unique(a.year):
            inds = np.where(a.year == year)[0]
            b = a.copy()
            b = b.iloc[inds, :]
            if error:
                res.loc[year, 'test'] = root_mean_squared_error(b.forecasted, b.observed)/np.mean(b.observed)
                res.loc[year, 'tl'] = root_mean_squared_error(b.forecasted_tl, b.observed)/ np.mean(b.observed)
            else:
                res.loc[year, 'test'] = pearsonr(b.forecasted, b.observed)[0]
                res.loc[year, 'tl'] = pearsonr(b.forecasted_tl, b.observed)[0]

        plt.bar(height=res.test, x=res.index-0.15, width=0.3)
        plt.bar(height=res.tl, x=res.index+0.15, width=0.3)
        plt.legend(['test_perf', 'test_perf_tl'])
        plt.title(self.crop.capitalize())
        plt.ylabel('R between observed and predicted yields')
        plt.show()

    #--------------------------------------------- Run and evaluate models -------------------------------------------
    def write_results(self):
        cols = ['lead_time', 'default_run', 'FE', 'HPT', 'FE&HPT']
        if self.temp_res=='2W':
            lead_times = [8, 7, 6, 5, 4, 3, 2, 1]
        elif self.temp_res=='M':
            lead_times = [4, 3, 2, 1]
        csv_file = pd.DataFrame(data=None, index=lead_times, columns=cols)
        csv_file.loc[:, 'lead_time'] = lead_times
        for lt,lead_time in enumerate(lead_times):
            csv_file.iloc[lt, 1:] = self.runforecast(lead_time=lead_time, model='XGB', feature_select=True, hyper_tune=True)
        csv_file.to_pickle(f'Results/Validation/FE_HPT_{self.temp_res}_unmerged.csv')

    def s1_vs_s2(self, model='XGB'):
        cols = ['s1', 's2', 'both']
        if self.temp_res=='2W':
            lead_times = [8, 7, 6, 5, 4, 3, 2, 1]
        elif self.temp_res=='M':
            lead_times = [4, 3, 2, 1]
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
        lead_times = [4, 3, 2, 1]
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
            XGB = xgb.XGBRegressor(
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100
            )
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

    def get_train_test(self, lead_time, merge_months=False):
        if self.farm:
            file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.farm}_{self.crop}_s1s2.csv', index_col=0)
        else:
            file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.crop}_s1s2.csv', index_col=0)
        file = file.dropna(axis=0)
        predictors = [p for p in file.columns[3:] if not p in ['date_last_obs', 'prev_year_crop']]
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        # used_predictors.append('prev_year_crop')
        if merge_months:
            predictor_file = self.merge_previous_months(file.loc[:, used_predictors])
        else:
            predictor_file = file.loc[:, used_predictors]

        X, X_test, y, y_test = train_test_split(predictor_file, file.loc[:, 'yield'], test_size=0.2, random_state=5)
        return X, X_test, y, y_test


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

def ua_dl_run(path_out, ua_only=False, temp_res='M', hidden_layer_sizes=[64, 32, 1], dropout_perc = 0.1):
    # if not os.path.exists(path_out): os.makedirs(path_out)
    # folders = os.listdir(path_out)
    # if len(folders) == 0:
    #     new_folder = f'run1'
    # else:
    #     f = [int(fold[-1:]) for fold in folders]
    #     new_folder = f'run{np.nanmax(f) + 1}'

    # path_out = os.path.join(path_out, new_folder)
    if not os.path.exists(path_out): os.makedirs(path_out)
    for crop in ['maize', 'spring_barley', 'winter_wheat']:
        a = ml(crop=crop, country='UA', temp_res=temp_res)
        if ua_only:
            a.eracast_tl_ua(lead_time=1, path_out=path_out, hidden_layer_sizes=hidden_layer_sizes, dropout_perc=dropout_perc)
        else:
            a.eracast_tl(lead_time=1, path_out=path_out)

def run_write_reg():
    dss = ['s1s2']
    for country in ['Czechia', 'Austria']:
        # write_res_reg(country=country, method='cross_trained', dataset=ds)
        for ds in dss:
            write_res_reg(country=country, method='reg_trained', dataset=ds)

def add_rows(results_values, cv, y_fore, y_obs):
    a_name = results_values.columns[0]
    new_rows = {a_name: [cv] * len(y_fore), 'yield_forecast': y_fore, 'yield_obs': y_obs}
    results_values = pd.concat([results_values, pd.DataFrame(new_rows)])
    return results_values

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

def dl(input_shape, hidden_layer_sizes=[128, 64, 1], epochs=300, batch_size=32,
                dropout_perc=0.1, optimizer='adam', learning_rate=0.001, early_stopping_patience=15):
    """
    Improved MLP regression pipeline with better defaults
    """
    model = Sequential()

    for i, neur in enumerate(hidden_layer_sizes):
        if i == 0:  # First layer
            model.add(Dense(neur, input_shape=input_shape, activation='relu',
                            kernel_initializer='he_normal'))  # Better initialization
            if dropout_perc > 0:
                model.add(Dropout(dropout_perc))
        elif i < len(hidden_layer_sizes) - 1:  # Hidden layers
            model.add(Dense(neur, activation='relu', kernel_initializer='he_normal'))
            if dropout_perc > 0:
                model.add(Dropout(dropout_perc))
        else:  # Output layer
            model.add(Dense(neur, activation=None, kernel_initializer='normal'))

    # Better optimizer configuration
    if optimizer == 'adam':
        print(learning_rate)
        opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    else:
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # Enhanced callbacks
    callbacks = []
    if early_stopping_patience:
        callbacks.append(EarlyStopping(monitor='loss', patience=early_stopping_patience,
                                       restore_best_weights=True, verbose=0))
        callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                           min_lr=1e-7, verbose=0))

    estimators = []
    if callbacks:
        estimators.append(('mlp', KerasRegressor(model=model, epochs=epochs,
                                                 batch_size=batch_size, callbacks=callbacks, verbose=0)))
    else:
        estimators.append(('mlp', KerasRegressor(model=model, epochs=epochs,
                                                 batch_size=batch_size, verbose=0)))

    return Pipeline(estimators)

def dl_old_orig(input_shape, hidden_layer_sizes=[100, 50, 50, 1], epochs=100, batch_size=10, dropout_perc=0.1, optimizer='adam',
       early_stopping_patience=None):
    """
    Create MLP regression pipeline
    :param input_shape: tuple or int - input shape (number of features)
    :param hidden_layer_sizes: list of model architecture. list entries are number of neurons per hidden layer
    :param epochs: int number of epochs over which model is fitted
    :param batch_size: int number of samples to work through before updating the internal model parameters
    :param dropout_perc: float (0,1) percentage of neurons used for dropout in hidden layers
    :param optimizer: str either adam or sgd. SGD is pretuned with learning rate=0.01 and momentum=0.9.
    :param early_stopping_patience: int number of epochs without loss reduction until model stops calibrating
    :return: scikit pipeline
    """
    model = Sequential()

    for i, neur in enumerate(hidden_layer_sizes):
        if i == 0:  # First layer
            model.add(Dense(neur, input_shape=input_shape, activation='relu'))
            if dropout_perc:
                model.add(Dropout(dropout_perc))
        elif i < len(hidden_layer_sizes) - 1:  # Hidden layers
            model.add(Dense(neur, activation='relu'))
            if dropout_perc:
                model.add(Dropout(dropout_perc))
        else:  # Output layer
            model.add(Dense(neur, activation=None))  # No activation for regression

    if optimizer == 'sgd':
        sgd = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[RootMeanSquaredError()])
    elif optimizer == 'adam':
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[RootMeanSquaredError()])
    else:
        raise ValueError(f'optimizer {optimizer} not available')

    estimators = []
    # Removed internal StandardScaler since you're handling scaling externally

    if early_stopping_patience:
        estimators.append(('mlp', KerasRegressor(model=model, epochs=epochs, batch_size=batch_size,
                                                 callbacks=[
                                                     EarlyStopping(monitor='loss', patience=early_stopping_patience)],
                                                 verbose=0)))
    else:
        estimators.append(('mlp', KerasRegressor(model=model, epochs=epochs, batch_size=batch_size, verbose=0)))

    pipe = Pipeline(estimators)
    return pipe


def dl_old(input_shape, hidden_layer_sizes=[100, 50, 50, 1], epochs=100, batch_size=10, dropout_perc=0.1,
           optimizer='adam',
           early_stopping_patience=None):
    """
    Create MLP regression pipeline
    """

    def create_model():
        model = Sequential()

        # Explicit Input layer (no more warning!)
        model.add(Input(shape=input_shape))

        for i, neur in enumerate(hidden_layer_sizes):
            if i < len(hidden_layer_sizes) - 1:  # Hidden layers
                model.add(Dense(neur, activation='relu'))
                if dropout_perc:
                    model.add(Dropout(dropout_perc))
            else:  # Output layer
                model.add(Dense(neur, activation=None))

        if optimizer == 'sgd':
            sgd = SGD(learning_rate=0.01, momentum=0.9)
            model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[RootMeanSquaredError()])
        elif optimizer == 'adam':
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=[RootMeanSquaredError()])
        else:
            raise ValueError(f'optimizer {optimizer} not available')

        return model

    if early_stopping_patience:
        regressor = KerasRegressor(
            model=create_model,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='loss', patience=early_stopping_patience)],
            verbose=0
        )
    else:
        regressor = KerasRegressor(
            model=create_model,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

    return regressor

def plotting(basepath):
    """
    Paper_plot
    :return: Plots the results generated by self.write_results to a boxplot comparing the performance of the model
                using S-1, S-2, and all data
    """
    lead_times = [4,3,2,1]
    lt_n = lead_times.copy()
    lt_n.reverse()
    fig = plt.figure(figsize=(18, 9))

    for pt, crop in enumerate(['maize', 'winter_wheat', 'spring_barley']):
        crop_name = crop.replace('_', ' ').capitalize()
        crop_data = pd.read_csv(f'Data/M/TL/{crop}_field.csv')
        crop_mean = np.mean(crop_data.loc[:, 'yield'])
        outer = gridspec.GridSpec(2, 3, wspace=0.12, hspace=0.07)
        seaborn.set(font_scale=1.4, style='white')

        for v, val in enumerate(['r2', 'nrmse']):
            for lt in lead_times:
                file = pd.read_csv(f'{basepath}/{crop}_{lt}_1_{val}.csv', index_col=0)
                file.loc[:,'lead_time'] = [lt]*len(file.index)
                reg2reg = file.loc[:, ['reg2reg_test', 'lead_time']]
                reg2field = file.loc[:, ['reg_test', 'lead_time']]
                tl = file.loc[:, ['tl_test', 'lead_time']]
                field = file.loc[:, ['field_test', 'lead_time']]
                xgb = file.loc[:, ['xgb_test', 'lead_time']]

                reg2reg = reg2reg.rename(columns={'reg2reg_test': 'perf'})
                reg2field = reg2field.rename(columns={'reg_test': 'perf'})
                tl = tl.rename(columns={'tl_test': 'perf'})
                field = field.rename(columns={'field_test': 'perf'})
                xgb = xgb.rename(columns={'xgb_test': 'perf'})

                reg2reg.loc[:, 'predictors'] = ['reg2reg'] * len(reg2reg.lead_time)
                reg2field.loc[:, 'predictors'] = ['reg2fld'] * len(reg2field.lead_time)
                tl.loc[:, 'predictors'] = ['reg2fld_ft'] * len(tl.lead_time)
                field.loc[:, 'predictors'] = ['fld2fld'] * len(field.lead_time)
                xgb.loc[:, 'predictors'] = ['xgb_fld'] * len(xgb.lead_time)

                final_lt = pd.concat([reg2reg, field, reg2field, tl, xgb])
                if lt==np.max(lead_times):
                    final = final_lt
                else:
                    final = pd.concat([final, final_lt])

            if v==1:
                final.perf = final.perf/crop_mean
            ax = plt.Subplot(fig, outer[v, pt])

            final.index = range(len(final.index))
            s = seaborn.boxplot(data=final, x='lead_time', y='perf', hue='predictors', zorder=2, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

            lw = 0.5
            [s.axvline(x+0.5, color='k', linewidth=lw*4, zorder=1) for x in s.get_xticks()[:-1]]
            # [s.axvspan(x-0.4, x-0.24, facecolor='blue', alpha=0.1, zorder=1) for x in s.get_xticks()]

            if v==0:
                s.set_title(crop_name)
                s.set_ylim([0, 1])
                ax.set_xticks([])
                s.set_xlabel("")
                minors = .1
            else:
                s.set_xticklabels(lt_n)
                s.set_xlabel('Lead Time [months]')
                s.set_ylim([0, 0.5])
                minors = .01

            lin_col = 'gray'
            alpha = 0.5
            # s.legend_.set_title(None)
            [[s.axhline(x + minors*m, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()] for m in range(10)]
            [s.axhline(x, color='k', linewidth=lw, zorder=1) for x in s.get_yticks()]

            if pt==0:
                if v==0:
                    s.set_ylabel("$\mathregular{R^{2}}$")
                else:
                    s.set_ylabel("nRMSE")
            else:
                ax.set_yticks([])
                s.set_ylabel("")
            fig.add_subplot(ax)
            if (pt==1) and (v==1):
                ax.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.35))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.13)
    # plt.show()
    fig_path = os.path.join(os.path.dirname(basepath), 'plots')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, 'tl_comparison_202409_Fig_5.png'), dpi=300)
    plt.close()

def plotting_cum_compare(basepath):
    """
    Paper_plot
    :return: Plots the results generated by self.write_results to a boxplot comparing the performance of the model
                using S-1, S-2, and all data
    """
    lead_times = [4,3,2,1]
    lt_n = lead_times.copy()
    lt_n.reverse()
    fig = plt.figure(figsize=(18, 9))

    for pt, crop in enumerate(['maize', 'winter_wheat', 'spring_barley']):
        crop_name = crop.replace('_', ' ').capitalize()
        crop_data = pd.read_csv(f'Data/M/TL/{crop}_field.csv')
        crop_mean = np.mean(crop_data.loc[:, 'yield'])
        outer = gridspec.GridSpec(2, 3, wspace=0.12, hspace=0.07)
        seaborn.set(font_scale=1.4, style='white')

        for v, val in enumerate(['r2', 'nrmse']):
            for lt in lead_times:
                file_old = pd.read_csv(f'{basepath}/run1_random_noncum/{crop}_{lt}_1_{val}.csv', index_col=0)
                file_new = pd.read_csv(f'{basepath}/run2_random/{crop}_{lt}_1_{val}.csv', index_col=0)
                file = file_new - file_old
                file.loc[:,'lead_time'] = [lt]*len(file.index)
                reg2reg = file.loc[:, ['reg2reg_test', 'lead_time']]
                reg2field = file.loc[:, ['reg_test', 'lead_time']]
                tl = file.loc[:, ['tl_test', 'lead_time']]
                field = file.loc[:, ['field_test', 'lead_time']]
                xgb = file.loc[:, ['xgb_test', 'lead_time']]

                reg2reg = reg2reg.rename(columns={'reg2reg_test': 'perf'})
                reg2field = reg2field.rename(columns={'reg_test': 'perf'})
                tl = tl.rename(columns={'tl_test': 'perf'})
                field = field.rename(columns={'field_test': 'perf'})
                xgb = xgb.rename(columns={'xgb_test': 'perf'})

                reg2reg.loc[:, 'predictors'] = ['reg2reg'] * len(reg2reg.lead_time)
                reg2field.loc[:, 'predictors'] = ['reg2fld'] * len(reg2field.lead_time)
                tl.loc[:, 'predictors'] = ['reg2fld_ft'] * len(tl.lead_time)
                field.loc[:, 'predictors'] = ['fld2fld'] * len(field.lead_time)
                xgb.loc[:, 'predictors'] = ['xgb_fld'] * len(xgb.lead_time)

                final_lt = pd.concat([reg2reg, field, reg2field, tl, xgb])
                if lt==np.max(lead_times):
                    final = final_lt
                else:
                    final = pd.concat([final, final_lt])

            if v==1:
                final.perf = final.perf
            ax = plt.Subplot(fig, outer[v, pt])

            final.index = range(len(final.index))
            s = seaborn.boxplot(data=final, x='lead_time', y='perf', hue='predictors', zorder=2, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

            lw = 0.5
            [s.axvline(x+0.5, color='k', linewidth=lw*4, zorder=1) for x in s.get_xticks()[:-1]]
            # [s.axvspan(x-0.4, x-0.24, facecolor='blue', alpha=0.1, zorder=1) for x in s.get_xticks()]

            if v==0:
                s.set_title(crop_name)
                s.set_ylim([-1, 1])
                ax.set_xticks([])
                s.set_xlabel("")
                minors = .1
            else:
                s.set_xticklabels(lt_n)
                s.set_xlabel('Lead Time [months]')
                s.set_ylim([-0.1, 0.1])
                minors = .01

            lin_col = 'gray'
            alpha = 0.5
            # s.legend_.set_title(None)
            [[s.axhline(x + minors*m, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()] for m in range(10)]
            [s.axhline(x, color='k', linewidth=lw, zorder=1) for x in s.get_yticks()]

            if pt==0:
                if v==0:
                    s.set_ylabel("Δ $\mathregular{R^{2}}$")
                else:
                    s.set_ylabel("Δ nRMSE")
            else:
                ax.set_yticks([])
                s.set_ylabel("")
            fig.add_subplot(ax)
            if (pt==1) and (v==1):
                ax.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.35))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.13)
    # plt.show()
    fig_path = os.path.join(basepath, 'plots')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, 'cumsum_vs_ind.png'), dpi=300)
    plt.close()

def plot_loocv(basepath, stat='r2'):
    """
    :return: Paper_plot
    """
    fig = plt.figure(figsize=(12, 6))
    seaborn.set(font_scale=1.2, style='ticks')

    crops = ['maize', 'winter_wheat', 'spring_barley']
    for pt, crop in enumerate(crops):
        crop_data = pd.read_csv(f'Data/M/TL/{crop}_field.csv')
        crop_mean = np.mean(crop_data.loc[:, 'yield'])

        crop_name = crop.replace('_', ' ').capitalize()
        a = pd.read_csv(f'{basepath}/{crop}_values_1_1.csv', index_col=0)
        if stat=='r2':
            corrs = pd.read_csv(f'{basepath}/{crop}_1_1_r2.csv', index_col=0)
        else:
            corrs = pd.read_csv(f'{basepath}/{crop}_1_1_nrmse.csv', index_col=0)
        corrs.loc[:, 'year'] = corrs.index
        mini, maxi = np.nanmin([a.yield_forecast, a.yield_obs]), np.nanmax([a.yield_forecast, a.yield_obs])

        outer = gridspec.GridSpec(1, len(crops)+1, wspace=0.2, width_ratios=[0.3, 0.3, 0.3, 0.1])
        inner = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=outer[pt], hspace=0.3, height_ratios=[0.4, 0.6])

        ax1 = plt.Subplot(fig, inner[1])

        seaborn.scatterplot(data=a, x='yield_forecast', y='yield_obs', hue='year', palette=cmr.managua, ax=ax1)
        line = np.linspace(0, 20, 100)
        ax1.plot(line, line, ':k', alpha=0.8)
        ax1.legend([],[], frameon=False)
        ax1.legend_.set_title("$\mathregular{R^{2}}$"+f"={np.round(pearsonr(a.yield_forecast, a.yield_obs)[0]**2,2)}")
        ax1.set_xlabel('Forecasted yield [t/ha]')
        if pt==0:
            ax1.set_ylabel('Observed yield [t/ha]')
            ax1.text(0.3, 18,'B)', fontsize=20, fontweight='bold')
        else:
            ax1.set_ylabel('')
        ax1.set_ylim(mini, maxi)
        ax1.set_xlim(mini, maxi)
        ha, le = ax1.get_legend_handles_labels()
        fig.add_subplot(ax1)

        ax0 = plt.Subplot(fig, inner[0])
        if stat=='r2':
            corrs_n = expand_df(corrs)
            print(corrs_n)
            s1 = seaborn.barplot(data=corrs_n, x='year', y='validation', hue='val_method', ax=ax0)
            [s1.axvline(x + .5, color='k') for x in s1.get_xticks()]
            # seaborn.barplot(corrs.tl_test, ax=ax0)
            ha1, le1 = ax0.get_legend_handles_labels()
            ax0.set_ylim(-1, 1)
        else:
            seaborn.barplot(corrs.tl_test/crop_mean, ax=ax0)
            ax0.set_ylim(0, 0.5)
        ax0.set_title(crop_name, fontsize=16)

        ax0.grid(axis='y', linestyle='--', color='gray', alpha=0.5)

        ax0.legend([], [], frameon=False)

        if pt==0:
            if stat == 'r2':
                ax0.set_ylabel("$\mathregular{R^{2}}$")
            else:
                ax0.set_ylabel("nRMSE")
            ax0.text(-2.2, 0.8, 'A)', fontsize=20, fontweight='bold')
        else:
            ax0.set_ylabel('')
        fig.add_subplot(ax0)

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[-1], hspace=0.2, height_ratios=[0.4, 0.6])

    ax2 = plt.Subplot(fig, inner[0])
    ax2.axis('off')
    # ax2.legend(ha,le,ncol=9, loc='center', fontsize=12)
    ax2.legend(ha1, le1, ncol=1, bbox_to_anchor=(1.25, 1.05))
    fig.add_subplot(ax2)

    ax3 = plt.Subplot(fig, inner[-1])
    ax3.axis('off')
    # ax2.legend(ha,le,ncol=9, loc='center', fontsize=12)
    ax3.legend(ha, le, ncol=1, bbox_to_anchor=(1.02, 1))
    fig.add_subplot(ax3)


    plt.subplots_adjust(left=0.1, right=0.98, top=0.95)
    # plt.show()

    fig_path = os.path.join(os.path.dirname(basepath), 'plots')
    os.makedirs(fig_path, exist_ok=True)
    # plt.show()
    plt.savefig(os.path.join(fig_path, f'loocv_202410_{stat}_all.png'), dpi=300)

    plt.close()
    #

def plot_tlvsfield(basepath, stats=['r2']):
    samples = np.unique([float(a.split('_')[-2]) for a in os.listdir(basepath)
                         if (a.endswith('csv')) and not (a.endswith('rmse.csv'))])
    tl = pd.DataFrame(index=samples, columns=['perc_samples', 'median', 'mean', 'std'])
    field = pd.DataFrame(index=samples, columns=['perc_samples', 'median', 'mean', 'std'])
    fontsize=16
    crops = ['maize', 'winter_wheat', 'spring_barley']
    fig = plt.figure(figsize=(12, 6))
    outer = gridspec.GridSpec(2, 1, hspace=0.5, height_ratios=[0.9, 0.05])
    inner = gridspec.GridSpecFromSubplotSpec(1, len(crops), subplot_spec=outer[0], wspace=0.18, width_ratios=[0.3, 0.3, 0.3])
    samples = [1 if a==1.0 else a for a in samples]
    for s, stat in enumerate(stats):
        for j, crop in enumerate(crops):
            crop_data = pd.read_csv(f'Data/M/TL/{crop}_field.csv')
            crop_mean = np.mean(crop_data.loc[:, 'yield'])
            crop_name = crop.replace('_', ' ').capitalize()
            for i in samples:
                a = pd.read_csv(f'{basepath}/{crop}_1_{i:.1f}_{stat}.csv')
                tl_t = a.tl_test
                field_t = a.field_test
                tl.loc[i, :] = [i * 0.8, np.nanmedian(tl_t), np.nanmean(tl_t), np.nanstd(tl_t)]
                field.loc[i, :] = [i * 0.8, np.nanmedian(field_t), np.nanmean(field_t), np.nanstd(field_t)]

            perc_samples = np.array(list(tl['perc_samples']))*100
            tl_med = np.array(list(tl['median']))
            tl_er = np.array(list(tl['std']))
            field_med = np.array(list(field['median']))
            field_er = np.array(list(field['std']))
            if stat=='rmse':
                tl_med = tl_med / crop_mean
                tl_er = tl_er / crop_mean
                field_med = field_med / crop_mean
                field_er = field_er / crop_mean

            ax = plt.Subplot(fig, inner[j])

            # Plot the first dataset with error bands
            ax.plot(perc_samples, tl_med, color='blue', label='reg2fld_ft')
            ax.fill_between(perc_samples, tl_med - tl_er, tl_med + tl_er, alpha=0.2, color='blue')

            # Plot the second dataset with error bands
            ax.plot(perc_samples, field_med, color='red', label='fld2fld')
            ax.fill_between(perc_samples, field_med - field_er, field_med + field_er, alpha=0.2, color='red')

            ax.set_ylim([0,1])

            # Add labels and title
            ax.set_xlabel('Used samples [%]', fontsize=fontsize)
            if j==0:
                if stat=='r2':
                    ax.set_ylabel("$\mathregular{R^{2}}$", fontsize=fontsize)
                else:
                    ax.set_ylabel("nRMSE", fontsize=fontsize)
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

    fig_path = os.path.join(os.path.dirname(basepath), 'plots')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, 'tlvsfield_202409_Fig_6.png'), dpi=300)

    plt.close()

def series2list(series, temp_step):
    vals = series.values
    if temp_step=='M':
        ls = [np.nan] * 4
        for i in range(len(ls)):
            ind_start, ind_end = -4*(4-i), -4*(3-i)
            if ind_end==0:
                ls[i] = np.nanmean(vals[ind_start:])
            else:
                ls[i] = np.nanmean(vals[ind_start:ind_end])
    elif temp_step=='2W':
        ls = [np.nan] * 8
        for i in range(len(ls)):
            ind_start, ind_end = -2 * (8 - i), -2 * (7 - i)
            if ind_end == 0:
                ls[i] = np.nanmean(vals[ind_start:])
            else:
                ls[i] = np.nanmean(vals[ind_start:ind_end])
    else:
        raise ValueError('temp_step not available. Choose either M for monthly or 2W for biweekly')
    return ls

def expand_df(df):
    valids = ['reg2reg_test', 'field_test', 'reg_test', 'tl_test', 'xgb_test']
    valids_name = {'reg2reg_test': 'reg2reg',
                   'field_test': 'fld2fld',
                   'reg_test': 'reg2fld',
                   'tl_test': 'reg2fld_tl',
                   'xgb_test': 'xgb_fld'}
    corrs_vals = pd.DataFrame(columns=['validation', 'year', 'val_method'])
    new_cols_o = corrs_vals.copy()
    for v in valids:
        new_cols = new_cols_o.copy()
        new_cols.loc[:, 'validation'] = df.loc[:, v]
        new_cols.loc[:, 'year'] = df.index
        new_cols.loc[:, 'val_method'] = [valids_name[v]] * df.shape[0]
        corrs_vals = pd.concat([corrs_vals, new_cols])

    return corrs_vals

def plot_loocv_tl(path, var='UA', old=False):
    crops = ['maize', 'winter_wheat', 'spring_barley']
    fontsize = 22

    fig = plt.figure(figsize=(15, 10))
    outer = gridspec.GridSpec(2, len(crops)+1, wspace=0.03, width_ratios=[0.3, 0.3, 0.3, 0.3], height_ratios=[0.95, 0.05], hspace=0.02)
    # outer = gridspec.GridSpec(1, len(crops), wspace=0.03, width_ratios=[0.3, 0.3, 0.3])

    for i, crop in enumerate(crops):
        a = pd.read_csv(f'{path}/{crop}_1.csv', index_col=0)
        if old:
            b = pd.read_csv(f'Data/SC2/M/{crop}_all_abs.csv', index_col=0)
            b.loc[:, 'country'] = [i[:2] for i in b.field_id]
            regs = b.iloc[np.where((b.country == 'UA') & (b.yield_anom > 0))[0], :]
            regs = regs.dropna(axis=0, how='any')
            if regs.shape[0] != a.shape[0]:
                raise ValueError("TILT")
            a.loc[:, 'region'] = regs.loc[:, 'field_id'].values
        a.loc[:, 'country'] = [sf[:2] for sf in a.loc[:, 'region']]
        years = np.unique(a.year)
        if var=='country':
            years = np.unique(a.country)
        for j, error in enumerate([True, False]):
            res = pd.DataFrame(index=years, columns=['test', 'tl', 'no_tl', 'eo', 'met', 'crop'])
            if old:
                res = pd.DataFrame(index=years, columns=['test', 'tl', 'crop'])

            for year in years:
                if var=='country':
                    inds = np.where(a.country == year)[0]
                else:
                    inds = np.where(a.year == year)[0]
                b = a.copy()
                b = b.iloc[inds, :]
                if error:
                    # res.loc[year, 'test'] = mean_squared_error(b.forecasted, b.observed, squared=False) / np.mean(b.observed)
                    # res.loc[year, 'tl'] = mean_squared_error(b.forecasted_tl, b.observed, squared=False) / np.mean(b.observed)
                    # if not old:
                        # res.loc[year, 'no_tl'] = mean_squared_error(b.forecasted_no_tl, b.observed, squared=False) / np.mean(b.observed)
                        # res.loc[year, 'eo'] = mean_squared_error(b.forecasted_eo, b.observed, squared=False) / np.mean(b.observed)
                        # res.loc[year, 'met'] = mean_squared_error(b.forecasted_met, b.observed, squared=False) / np.mean(b.observed)
                    #
                    res.loc[year, 'test'] = mean_absolute_percentage_error(b.forecasted, b.observed)
                    res.loc[year, 'tl'] = mean_absolute_percentage_error(b.forecasted_tl, b.observed)
                    if not old:
                        res.loc[year, 'no_tl'] = mean_absolute_percentage_error(b.forecasted_no_tl, b.observed)
                        res.loc[year, 'eo'] = mean_absolute_percentage_error(b.forecasted_eo, b.observed)
                        res.loc[year, 'met'] = mean_absolute_percentage_error(b.forecasted_met, b.observed)

                else:
                    # res.loc[year, 'test'] = pearsonr(b.forecasted, b.observed)[0]
                    # res.loc[year, 'tl'] = pearsonr(b.forecasted_tl, b.observed)[0]
                    # if not old:
                        # res.loc[year, 'no_tl'] = pearsonr(b.forecasted_no_tl, b.observed)[0]
                        # res.loc[year, 'eo'] = pearsonr(b.forecasted_eo, b.observed)[0]
                        # res.loc[year, 'met'] = pearsonr(b.forecasted_met, b.observed)[0]

                    res.loc[year, 'test'] = explained_variance_score(b.forecasted, b.observed)
                    res.loc[year, 'tl'] = explained_variance_score(b.forecasted_tl, b.observed)
                    if not old:
                        res.loc[year, 'no_tl'] = explained_variance_score(b.forecasted_no_tl, b.observed)
                        res.loc[year, 'eo'] = explained_variance_score(b.forecasted_eo, b.observed)
                        res.loc[year, 'met'] = explained_variance_score(b.forecasted_met, b.observed)

            res.crop = [crop.replace('_', ' ').capitalize()] * res.shape[0]
            if error:
                if i == 0:
                    all_err = res
                else:
                    all_err = pd.concat([all_err, res])
            else:
                if i == 0:
                    all_cor = res
                else:
                    all_cor = pd.concat([all_cor, res])

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, i], hspace=0.1, height_ratios=[0.5, 0.5])
            ax1 = plt.Subplot(fig, inner[j])

            col1 = '#1b9e77'
            col2 = '#d95f02'
            col3 = '#7570b3'
            col4 = '#e7298a'
            col5 = '#66a61e'

            wid = 0.18
            # if var != 'UA':
            #     labels = res.index
            #     res.index = range(len(res.index))
            #     ax1.bar(height=res.test, x=res.index - wid, width=wid, color=col1, zorder=1, label='test')
            #     ax1.axhline(np.mean(res.test), xmin=0, xmax=2023, color=col1, zorder=0, label='test mean', linestyle='--')
            if var=='UA':
                # ax1.bar(height=res.tl, x=res.index, width=wid, color=col2, zorder=1, label='tl')
                # ax1.axhline(np.mean(res.tl), xmin=0, xmax=2023, color=col2, zorder=0, label='tl mean', linestyle='--')
                # ax1.bar(height=res.no_tl, x=res.index + wid, width=wid, color=col3, zorder=1, label='no tl')
                # ax1.axhline(np.mean(res.no_tl), xmin=0, xmax=2023, color=col3, zorder=0, label='no tl mean',
                #             linestyle='--')
                ax1.bar(height=res.test, x=res.index - 2*wid, width=wid, color=col1, zorder=1, label='test')
                ax1.bar(height=res.tl, x=res.index - wid, width=wid, color=col2, zorder=1, label='tl')
                if not old:
                    ax1.bar(height=res.no_tl, x=res.index, width=wid, color=col3, zorder=1, label='no_tl')
                    ax1.bar(height=res.eo, x=res.index + wid, width=wid, color=col4, zorder=1, label='tl_eo')
                    ax1.bar(height=res.met, x=res.index + 2*wid, width=wid, color=col5, zorder=1, label='tl_met')
            elif var=='country':
                ax1.bar(height=res.test, x=res.index, width=wid, color=col1, zorder=1, label='test')
                ax1.bar(height=res.tl, x=res.index + wid, width=wid, color=col2, zorder=1, label='tl')
                if not old:
                    ax1.bar(height=res.no_tl, x=res.index + wid, width=wid, color=col3, zorder=1, label='no_tl')
                    ax1.bar(height=res.eo, x=res.index + wid, width=wid, color=col4, zorder=1, label='tl_eo')
                    ax1.bar(height=res.met, x=res.index + wid, width=wid, color=col5, zorder=1, label='tl_met')
            ax1.tick_params(axis='both', labelsize=fontsize)
            ax1.grid(axis='y', linestyle='--', color='gray', alpha=0.5)
            ha, le = ax1.get_legend_handles_labels()
            if i==0:
                if error:
                    ax1.set_ylabel('MAPE', fontsize=fontsize)
                else:
                    ax1.set_ylabel('$\mathregular{R^{2}}$', fontsize=fontsize)
                    # ax1.set_ylabel('R', fontsize=fontsize)
            else:
                ax1.set_yticklabels([])
            if j==0:
                ax1.set_title(crop.replace('_', ' ').capitalize(), fontsize=fontsize + 2)
                ax1.set_xticks([])
                ax1.set_ylim(0, 1)
            else:
                ax1.set_ylim(0, 1)
            fig.add_subplot(ax1)

    all_cor = pd.melt(all_cor, id_vars=['crop'])
    all_err = pd.melt(all_err, id_vars=['crop'])
    print(all_cor)

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, -1], hspace=0.1, height_ratios=[0.5, 0.5])
    ax_box_cor = plt.Subplot(fig, inner[1])
    ax_box_err = plt.Subplot(fig, inner[0])
    col_palette = [col1, col2, col3, col4, col5]
    seaborn.boxplot(data=all_cor, x='crop', y='value', hue='variable', zorder=2, ax=ax_box_cor, palette=col_palette)
    seaborn.boxplot(data=all_err, x='crop', y='value', hue='variable', zorder=2, ax=ax_box_err, palette=col_palette)

    ax_box_err.set_xticks([])
    ax_box_cor.tick_params(axis='both', labelsize=fontsize-4, rotation=45)
    for ax in [ax_box_cor, ax_box_err]:
        ax.set_ylim(0, 1)
        ax.set_yticklabels([])
        ax.set_ylabel('')
        ax.grid(axis='y', linestyle='--', color='gray', alpha=0.5)
        ax.set_xlabel('')
        ax.legend([], [], frameon=False)
        fig.add_subplot(ax)

    ax2 = plt.Subplot(fig, outer[1,1])
    ax2.axis('off')
    ax2.legend(ha, le, ncol=5, bbox_to_anchor=(2.15, 0), fontsize=fontsize)
    fig.add_subplot(ax2)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.92)
    plt.show()

    # fig_path = os.path.join(os.path.dirname(path), 'figures0')
    # if not os.path.exists(fig_path): os.makedirs(fig_path)
    # plt.savefig(f'{fig_path}/Validation_tl_{var}.png', dpi=300)

def plot_loocv_country(path):
    crops = ['maize', 'winter_wheat', 'spring_barley']
    fontsize = 22

    fig = plt.figure(figsize=(15, 10))
    outer = gridspec.GridSpec(2, len(crops), wspace=0.03, width_ratios=[0.3]*len(crops), height_ratios=[0.95, 0.05], hspace=0.02)

    files = os.listdir(path)

    for i, crop in enumerate(crops):
        country_files = [a for a in files if a.startswith(crop)]
        countries = [a.split('_')[-2] for a in country_files]
        for country in countries:
            a = pd.read_csv(f'{path}/{crop}_1.csv', index_col=0)

            for j, error in enumerate([False, True]):
                res = pd.DataFrame(index=countries, columns=['test', 'tl', 'no_tl', 'eo', 'met'])

                res_test = []
                res_tl = []
                res_no_tl = []
                res_eo = []
                res_met = []

                for year in np.unique(a.year):
                    c = a.copy()
                    inds_y = np.where(c.year == year)[0]
                    c = c.iloc[inds_y, :]
                    if error:
                        res_test.append(
                            root_mean_squared_error(c.forecasted, c.observed) / np.mean(a.observed))
                        res_tl.append(
                            root_mean_squared_error(c.forecasted_tl, c.observed) / np.mean(a.observed))
                        res_no_tl.append(
                            root_mean_squared_error(c.forecasted_no_tl, c.observed) / np.mean(a.observed))
                        res_eo.append(
                            root_mean_squared_error(c.forecasted_eo, c.observed) / np.mean(a.observed))
                        res_met.append(
                            root_mean_squared_error(c.forecasted_met, c.observed) / np.mean(a.observed))
                    else:
                        res_test.append(explained_variance_score(c.forecasted, c.observed))
                        res_tl.append(explained_variance_score(c.forecasted_tl, c.observed))
                        res_no_tl.append(explained_variance_score(c.forecasted_no_tl, c.observed))
                        res_eo.append(explained_variance_score(c.forecasted_eo, c.observed))
                        res_met.append(explained_variance_score(c.forecasted_met, c.observed))

                res.loc[country, 'test'] = np.nanmean(res_test)
                res.loc[country, 'tl'] = np.nanmean(res_tl)
                res.loc[country, 'no_tl'] = np.nanmean(res_no_tl)
                res.loc[country, 'eo'] = np.nanmean(res_eo)
                res.loc[country, 'met'] = np.nanmean(res_met)

            res_melted = res.reset_index().melt(id_vars='index', var_name='Group', value_name='Value')
            res_melted.rename(columns={'index': 'Category'}, inplace=True)

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, i], hspace=0.1, height_ratios=[0.5, 0.5])
            ax1 = plt.Subplot(fig, inner[j])

            seaborn.barplot(data=res_melted, x='Category', y='Value', hue='Group', palette='Set2', ax=ax1)

            fig.add_subplot(ax1)


    # ax2 = plt.Subplot(fig, outer[1,1])
    # ax2.axis('off')
    # ax2.legend(ha, le, ncol=4, bbox_to_anchor=(1.4, 0.1), fontsize=fontsize)
    # fig.add_subplot(ax2)

    # plt.subplots_adjust(left=0.1, right=0.98, top=0.92)
    plt.show()

    # fig_path = os.path.join(os.path.dirname(path), 'figures0')
    # if not os.path.exists(fig_path): os.makedirs(fig_path)
    # plt.savefig(f'{fig_path}/Validation_tl_country_ex_var.png', dpi=300)

def plot_ua(path, war=False, old=False):
    """
    uses data from https://www.crisisgroup.org/content/ukraine-war-map-tracking-frontlines
    :return: paper_plot_ua
    """
    war_map = pd.read_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\war_map.csv').iloc[:,1:]
    # print(war_map)
    war_map_d = {a: b for a,b in zip(war_map.Region_ID, war_map.War)}
    fig = plt.figure(figsize=(12, 5))
    seaborn.set(font_scale=1.7)
    seaborn.set_style('white')
    seaborn.color_palette('Spectral')
    for pt, crop in enumerate(['maize', 'winter_wheat', 'spring_barley']):
        file = pd.read_csv(f'{path}/{crop}_1.csv', index_col=0)
        file.loc[:, 'country'] = [i[:2] for i in file.region]
        file = file.iloc[np.where(file.country == 'UA')[0], :]
        if old:
            b = pd.read_csv(f'Data/SC2/M/{crop}_all_abs.csv', index_col=0)
            b.loc[:, 'country'] = [i[:2] for i in b.field_id]
            regs = b.iloc[np.where((b.country == 'UA') & (b.yield_anom > 0))[0], :]
            regs = regs.dropna(axis=0, how='any')

        else:
            a = pd.read_csv(f'Data/SC2/2W/final/{crop}_all_abs_fin_year_det.csv', index_col=0)
            a.loc[:,'country'] = [i[:2] for i in a.field_id]
            regs = a.iloc[np.where(a.country=='UA')[0], :]

        if regs.shape[0] != file.shape[0]:
            raise ValueError("TILT")
        file.loc[:, 'region'] = regs.loc[:, 'field_id'].values
        file.loc[:, 'war'] = [war_map_d[a] for a in file.loc[:, 'region']]

        crop_name = crop.replace('_', ' ').capitalize()

        outer = gridspec.GridSpec(1, 3, wspace=0.12)
        ax = plt.Subplot(fig, outer[pt])

        if pt == 1:
            if not war:
                s = seaborn.scatterplot(x=file.forecasted_tl, y=file.observed, hue=file.year, ax=ax)
            else:
                file = file.iloc[np.where(file.year==2022)[0], :]
                s = seaborn.scatterplot(x=file.forecasted_tl, y=file.observed, hue=file.war, ax=ax,
                                        palette=dict(No="#4b4453", Regained="#ff8066", Occupied="#e41a1c"), s=80)
            seaborn.move_legend(s, loc='lower center', bbox_to_anchor=(.5, -.4), ncol=6, title=None)
        else:
            if not war:
                s = seaborn.scatterplot(x=file.forecasted_tl, y=file.observed, hue=file.year, ax=ax, legend=False)
            else:
                file = file.iloc[np.where(file.year == 2022)[0], :]
                s = seaborn.scatterplot(x=file.forecasted_tl, y=file.observed, hue=file.war, ax=ax, legend=False,
                                        palette=dict(No="#4b4453", Regained="#ff8066", Occupied="#e41a1c"), s=80)
        s.set_xlabel('forecasted yield [t/ha]')
        s.set_title(crop_name)

        if pt == 0:
            s.set_ylabel("Observed yield [t/ha]")
        else:
            # ax.set_yticks([])
            s.set_ylabel("")

        # Draw a line of x=y
        min0, min1 = np.min(file.observed), np.min(file.forecasted)
        max0, max1 = np.max(file.observed), np.max(file.forecasted)
        # if crop=='winter_wheat': min0, min1, max0, max1 = 2,2,6,6
        lims = [min(min0, min1)-0.1, max(max0, max1)+0.1]
        s.plot(lims, lims, '--', color='#999999')
        s.set_ylim(lims)
        s.set_xlim(lims)

        fig.add_subplot(ax)
    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.25)
    plt.show()
    # path_out = os.path.join(os.path.dirname(path), 'figures0', 'scatter_war_lstm.png')
    # plt.savefig(path_out, dpi=300)
    # plt.close()

def ua_line(path, old=False):
    """
    :return: paper_plot
    """
    war_map = pd.read_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\war_map.csv').iloc[:, 1:]
    crops = ['maize', 'winter_wheat', 'spring_barley']
    fontsize=20
    ymax = {a: b for a, b in zip(crops, [10, 6, 4])}
    war_map_d = {a: b for a, b in zip(war_map.Region_ID, war_map.War)}
    war_zones = np.unique(war_map.iloc[np.where(war_map.War=='Occupied')[0],0])
    fig = plt.figure(figsize=(12, 10))
    seaborn.set(font_scale=1.7)
    seaborn.set_style('white')
    seaborn.color_palette('Spectral')
    outer = gridspec.GridSpec(len(crops)+1, 1, height_ratios=[0.3, 0.3, 0.3, 0.05])
    for c, crop in enumerate(crops):
        inner = gridspec.GridSpecFromSubplotSpec(1, len(war_zones), subplot_spec=outer[c])
        file = pd.read_csv(f'{path}/{crop}_1.csv', index_col=0)
        file.loc[:, 'country'] = [i[:2] for i in file.region]
        file = file.iloc[np.where(file.country == 'UA')[0], :]
        if old:
            b = pd.read_csv(f'Data/SC2/M/{crop}_all_abs.csv', index_col=0)
            b.loc[:, 'country'] = [i[:2] for i in b.field_id]
            regs = b.iloc[np.where((b.country == 'UA') & (b.yield_anom > 0))[0], :]
            file_ua = regs.dropna(axis=0, how='any')

        else:
            file_ref = pd.read_csv(f'Data/SC2/2W/final/{crop}_all_abs_fin_year_det.csv', index_col=0)
            file_ref.loc[:, 'country'] = [field_id[:2] for field_id in file_ref.field_id]
            file_ua = file_ref.iloc[np.where(file_ref.country == 'UA')[0], :]
        # if file_ua.shape[0] != file.shape[0]:
        #     raise ValueError("TILT")

        # file.region = file_ua.field_id.values
        file.loc[:, 'war'] = [war_map_d[a] for a in file.region]

        errors = {'year': [],
                  'errors': []}
        for year in np.unique(file.year):
            errors['year'].append(year)
            subfile = file.iloc[np.where(file.year==year)[0],:]
            errors['errors'].append(root_mean_squared_error(subfile.prediction, subfile.observed))
        errors_df = pd.DataFrame(errors)

        crop_name = crop.replace('_', ' ').capitalize()

        for pt, war_zone in enumerate(war_zones):
            this_file = file.iloc[np.where(file.region==war_zone)[0],:]
            ax = plt.Subplot(fig, inner[pt])
            time = this_file.year
            vals = this_file.prediction
            ers = [errors_df.iloc[np.where(errors_df.year==t)[0],1].values[0] for t in time]
            ax.plot(time, vals, label='estimated yield')
            ax.fill_between(time, vals-ers, vals+ers, alpha=0.2)
            ax.scatter(time, this_file.observed, marker='x', label='observed yield', linewidths=4)
            ax.set_ylim(0, ymax[crop])

            if c<(len(crops)-1):
                ax.set_xticklabels([])
            else:
                ax.set_xticks(time)
                ax.set_xticklabels(time, rotation=45, fontsize=fontsize-2)
            if c==0:
                ax.set_title(war_zone, fontsize=fontsize+2)
            if pt==0:
                ax.set_ylabel(f'{crop_name} \nyields [t/ha]', fontsize=fontsize)
            else:
                ax.set_yticklabels([])

            ha, le = ax.get_legend_handles_labels()

            fig.add_subplot(ax)

    ax2 = plt.Subplot(fig, outer[3])
    ax2.axis('off')
    ax2.legend(ha, le, ncol=2, fontsize=fontsize, bbox_to_anchor=(0.82, 0.05))
    fig.add_subplot(ax2)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.06)
    # plt.show()
    fig_path = os.path.join(os.path.dirname(path), 'figures0')
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(f'{fig_path}/line_war_zones_xgb.png', dpi=300)

def run_dl(path_out_t, cv_method='random', lts=[1, 2], p_tls=[1], not_random=False, standardize=False, feature_importance=False):
    """
    :param cv_method: cross validation method. Check ml.transferlearn for more info
    :param lts: list of leadtimes for which the forecasts will be calculated in months
    :param p_tls: list of perc_tl_samples used for ml.transferlearn
    :return:
    """

    folders = [int(a[3]) for a in os.listdir(path_out_t) if a.startswith('run')]
    if len(folders)==0:
        new_folder = f'run1_{cv_method}'
    else:
        new_folder = f'run{np.nanmax(folders) + 1}_{cv_method}'
    #
    for crop in ['maize', 'spring_barley', 'winter_wheat']:  #'spring_barley', 'winter_wheat', 'maize'
        if feature_importance:
            a = ml(crop=crop, country='TL')
            # a.feature_importance_tl()
            a.plot_feature_imp()
        else:
            a = ml(crop=crop, country='TL', path_out=os.path.join(path_out_t, new_folder))
            for lt in lts:
                for p_tl_sa in p_tls:
                    a.transferlearn(lead_time=lt, field2reg=False, tl_train_layers=4, perc_tl_samples=p_tl_sa,
                              hidden_layer_sizes=[100, 50, 50, 1], epochs=100, batch_size=10, dropout_perc=0.3,
                              optimizer='adam', early_stopping_patience=3, cv=30, cv_method=cv_method,
                              preds=('sig40', 'evi', 'ndwi', 'nmdi'), not_random=not_random, standardize=standardize)
                    # preds=('sig40', 'ndvi', 'evi', 'ndwi', 'nmdi', 't2m', 'tp', 'swvl', 'pev', 'evavt', 'ssr')

def detrend_yield():
    for crop in ['spring_barley']:
        data = pd.read_csv(f'Data/M/TL/{crop}_field.csv')
        regs = np.unique(data.field_id)
        print(data)
        for reg in regs[:10]:
            reg_i = np.where(data.field_id==reg)[0]
            print(reg_i)
            if len(reg_i)>3:
                pass


        # data.to_csv(f'Data/M/TL/{crop}_s1s2_field_detrended.csv')

def plot_sample_random_vs_cluster(basepath):
    cluster_path = r'H:\Emanuel\Code_new\yipeeo\Results\Validation\dl\meteo_data\run4_randomhs'
    random_path = r'H:\Emanuel\Code_new\yipeeo\Results\Validation\dl\meteo_data\run3_random'

    fontsize=16
    crops = ['maize', 'winter_wheat', 'spring_barley']
    stats = ['pearson', 'rmse']

    seaborn.set(font_scale=1.6, style='whitegrid')

    fig = plt.figure(figsize=(12, 6))
    outer = gridspec.GridSpec(3, 1, hspace=0.8, height_ratios=[0.45, 0.45, 0.02])

    for s, stat in enumerate(stats):
        inner = gridspec.GridSpecFromSubplotSpec(1, len(crops), subplot_spec=outer[s], wspace=0.18,
                                                 width_ratios=[0.3, 0.3, 0.3])
        for j, crop in enumerate(crops):
            crop_data = pd.read_csv(f'Data/M/TL/{crop}_field.csv')
            crop_mean = np.mean(crop_data.loc[:, 'yield'])
            crop_name = crop.replace('_', ' ').capitalize()
            cluster = pd.read_csv(f'{cluster_path}/{crop}_1_0.2_{stat}.csv')
            randoms = pd.read_csv(f'{random_path}/{crop}_1_0.2_{stat}.csv')
            tl_random = randoms.tl_test.dropna()
            tl_cluster = cluster.tl_test.dropna()
            field_t = randoms.field_test.dropna()
            tl_random = tl_random.rename('finetune_random')
            tl_cluster = tl_cluster.rename('finetune_cluster')
            field_t = field_t.rename('no_finetune')

            if stat=='rmse':
                tl_random = tl_random / crop_mean
                tl_cluster = tl_cluster / crop_mean
                field_t = field_t / crop_mean
            else:
                tl_random = tl_random**2
                tl_cluster = tl_cluster ** 2
                field_t = field_t ** 2

            ax = plt.Subplot(fig, inner[j])

            # Plot the first dataset with error bands
            seaborn.kdeplot(data=[field_t, tl_cluster, tl_random], ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            if (s == 1) and (j == 1):
                seaborn.move_legend(ax, 'lower left', bbox_to_anchor=(-0.7,-0.75), ncol=3)
            else:
                ax.get_legend().remove()
    #         # Add labels and title

            if stat=='pearson':
                ax.set_xlabel("$\mathregular{R^{2}}$", fontsize=fontsize)
            else:
                ax.set_xlabel("nRMSE", fontsize=fontsize)

            if j>0:
                ax.set_ylabel('')

            if s==0:
                ax.set_title(crop_name, weight='bold')
            fig.add_subplot(ax)

    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.05)

    # plt.show()
    fig_path = os.path.join(basepath, 'plots')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, 'random_vs_cluster.png'), dpi=300)

    plt.close()

def plot_dist_crop_preds(basepath, crop='winter_wheat', standardize=False):
    """
    :return: Paper_plot
    """
    splits = ['Random', 'L1YOCV']
    regs = ['field', 'regional']
    year = 2020
    seaborn.set(font_scale=1.4, style='ticks')

    fig = plt.figure(figsize=(20, 8))
    outermost = gridspec.GridSpec(2, 1, wspace=0.25, height_ratios=[0.02, 0.98], hspace=0.1)
    title_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outermost[0])
    outer = gridspec.GridSpecFromSubplotSpec(2, len(splits)+len(regs) + 1, subplot_spec=outermost[1], wspace=0.25, width_ratios=[0.2, 0.2, 0.05, 0.2, 0.2], hspace=0.4, height_ratios=[0.6, 0.4])

    for r, reg in zip([0, 3], regs):
        for pt, split in enumerate(splits):
            crop_data = pd.read_csv(f'Data/M/TL/{crop}_{reg}.csv')
            crop_data['sig40_cr_mean_daily_LT1'].where(crop_data['sig40_cr_mean_daily_LT1'] < -3.5, np.nan, inplace=True)
            years_field = crop_data.c_year
            X = crop_data.loc[:, ['sig40_cr_mean_daily_LT1', 'evi_LT1']]
            y = crop_data.loc[:, ['yield']]

            if split == 'Random':
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
            elif split == 'L1YOCV':
                year_test, year_train = np.where(years_field == year)[0], np.where(years_field != year)[0]
                X_train, y_train = X.iloc[year_train, :], y.iloc[year_train]
                X_test, y_test = X.iloc[year_test, :], y.iloc[year_test]

            if standardize:
                scaler_xtr = StandardScaler().fit(X_train)
                scaler_xte = StandardScaler().fit(X_test)
                scaler_ytr = StandardScaler().fit(y_train)
                scaler_yte = StandardScaler().fit(y_test)
                X_train = pd.DataFrame(scaler_xtr.transform(X_train), columns=X_train.columns)
                X_test = pd.DataFrame(scaler_xte.transform(X_test), columns=X_test.columns)
                y_train = pd.DataFrame(scaler_ytr.transform(y_train), columns=y_train.columns)
                y_test = pd.DataFrame(scaler_yte.transform(y_test), columns=y_test.columns)

            ax1 = plt.Subplot(fig, outer[1, pt+r])
            seaborn.kdeplot(data=y_train.loc[:, 'yield'], label='Train', fill=True, ax=ax1)
            seaborn.kdeplot(data=y_test.loc[:, 'yield'], label='Test', fill=True, ax=ax1)

            ax1.set_xlabel('Yield [t/ha]')
            if (split=='L1YOCV') & (reg==regs[0]):
                ax1.legend(bbox_to_anchor=(1.5, 1.5), fontsize=16)
            if pt>0:
                ax1.set_ylabel('')
            fig.add_subplot(ax1)

            ax0 = plt.Subplot(fig, outer[0, pt+r])
            seaborn.kdeplot(x=X_train.sig40_cr_mean_daily_LT1, y=X_train.evi_LT1, ax=ax0)
            seaborn.kdeplot(x=X_test.sig40_cr_mean_daily_LT1, y=X_test.evi_LT1, ax=ax0)

            ax0.set_xlabel('Sig40 CR LT1')
            if pt>0:
                ax0.set_ylabel('')
            else:
                ax0.set_ylabel('EVI LT1')
            if split=='L1YOCV':
                ax0.set_title(f'{split} {year}', fontdict={'fontsize':20})
            else:
                ax0.set_title(split, fontdict={'fontsize':20})
            fig.add_subplot(ax0)

    axt = plt.Subplot(fig, title_grid[0])
    axt.axis('off')
    axt.set_title('Field', fontdict={'fontweight': 'bold', 'fontsize':24})
    fig.add_subplot(axt)

    axt = plt.Subplot(fig, title_grid[1])
    axt.axis('off')
    axt.set_title('Regional', fontdict={'fontweight': 'bold', 'fontsize':24})
    fig.add_subplot(axt)

    line = plt.Line2D((.5, .5), (0, .39), color="k", linewidth=5)
    fig.add_artist(line)
    line = plt.Line2D((.5, .5), (.49, 1), color="k", linewidth=5)
    fig.add_artist(line)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.94)
    # plt.show()
    plt.savefig(os.path.join(basepath,'plots/sampling.png'), dpi=300)

def tab_diff_sample(crop):
    """
    :return: Paper_plot
    """
    regs = ['field', 'regional']
    years = range(2017, 2023)
    year = 2020
    preds = ['sig40_cr_mean_daily_LT1', 'evi_LT1']

    for reg in regs:
        res = pd.DataFrame(index=['yield'] + preds)

        crop_data = pd.read_csv(f'Data/M/TL/{crop}_{reg}.csv')
        crop_data['sig40_cr_mean_daily_LT1'].where(crop_data['sig40_cr_mean_daily_LT1'] < -3.5, np.nan, inplace=True)
        crop_data = crop_data.dropna(axis=0)
        years_field = crop_data.c_year
        X = crop_data.loc[:, ['sig40_cr_mean_daily_LT1', 'evi_LT1']]
        y = crop_data.loc[:, ['yield']]

        #Random split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
        res.loc['yield', 'Random'] = ttest_ind(y_train, y_test).pvalue

        for pred in preds: res.loc[pred, 'Random'] = ttest_ind(X_train.loc[:, pred], X_test.loc[:, pred]).pvalue

        for year in years:
            year_test, year_train = np.where(years_field == year)[0], np.where(years_field != year)[0]
            X_train, y_train = X.iloc[year_train, :], y.iloc[year_train]
            X_test, y_test = X.iloc[year_test, :], y.iloc[year_test]
            res.loc['yield', str(year)] = ttest_ind(y_train, y_test).pvalue
            for pred in preds: res.loc[pred, str(year)] = ttest_ind(X_train.loc[:, pred], X_test.loc[:, pred]).pvalue
        res.round(2).to_csv(f'Results/explore_data/ttest_{crop}_{reg}.csv')
    #
    #
    #         X_train.sig40_cr_mean_daily_LT1
    #         X_train.evi_LT1
    #


class TransferLearningPermutationCV:
    def __init__(self, model_builder=None, model=None, cv=5, n_repeats=10, random_state=42,
                 task_type='regression', metric=None):
        """
        Permutation Feature Importance with Cross-Validation for Transfer Learning

        Parameters:
        -----------
        model_builder : callable, optional
            Function that returns a new compiled Keras model (for Keras models)
        model : sklearn estimator, optional
            Pre-built sklearn model
        cv : int or cross-validation generator
            Number of folds or CV strategy
        n_repeats : int
            Number of times to permute each feature
        random_state : int
            Random state for reproducibility
        task_type : str
            'regression' or 'classification'
        metric : callable, optional
            Custom metric function (y_true, y_pred) -> float
        """
        if model_builder is None and model is None:
            raise ValueError("Either model_builder or model must be provided")

        self.model_builder = model_builder
        self.model = model
        self.cv = cv if hasattr(cv, 'split') else KFold(n_splits=cv, shuffle=True, random_state=random_state)
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.task_type = task_type
        self.is_keras = model_builder is not None or (KERAS_AVAILABLE and isinstance(model, keras.Model))

        # Set default metric based on task type
        if metric is None:
            self.metric = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)  # Negative MSE (higher is better)
        else:
            self.metric = metric

        # Storage for results
        self.original_results_ = None
        self.transfer_results_ = None

    def fit_transfer_learning(self, X_train, y_train, X_transfer, y_transfer,
                              feature_names=None, original_fit_kwargs=None, transfer_fit_kwargs=None):
        """
        Fit models with transfer learning and compute feature importance for both stages

        Parameters:
        -----------
        X_train : array-like
            Original training features
        y_train : array-like
            Original training targets
        X_transfer : array-like
            Transfer learning features
        y_transfer : array-like
            Transfer learning targets
        feature_names : list, optional
            Names of features
        original_fit_kwargs : dict, optional
            Fitting parameters for original training
        transfer_fit_kwargs : dict, optional
            Fitting parameters for transfer learning
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        self.feature_names_ = feature_names

        # Set default fitting parameters
        original_fit_kwargs = original_fit_kwargs or {}
        transfer_fit_kwargs = transfer_fit_kwargs or {}

        if self.is_keras:
            default_keras_kwargs = {'epochs': 100, 'batch_size': 32, 'verbose': 0, 'validation_split': 0.1}
            original_fit_kwargs = {**default_keras_kwargs, **original_fit_kwargs}
            transfer_fit_kwargs = {**default_keras_kwargs, **transfer_fit_kwargs}

        print("Starting Transfer Learning Feature Importance Analysis...")
        print("=" * 60)

        # Stage 1: Original model training and feature importance
        print("Stage 1: Training original models and computing feature importance...")
        self.original_results_ = self._compute_importance_stage(
            X_train, y_train, X_train, y_train,
            "Original", original_fit_kwargs
        )

        # Stage 2: Transfer learning and feature importance
        print(f"\nStage 2: Transfer learning and computing feature importance...")
        self.transfer_results_ = self._compute_importance_stage(
            X_train, y_train, X_transfer, y_transfer,
            "Transfer", transfer_fit_kwargs, original_fit_kwargs
        )

        print(f"\nTransfer Learning Analysis Complete!")
        print(self.original_results_
            # f"Original Model CV Score: {self.original_results_['cv_scores_mean']:.4f} ± {self.original_results_['cv_scores_std']:.4f}"
        )
        print(self.transfer_results_
            # f"Transfer Model CV Score: {self.transfer_results_['cv_scores_mean']:.4f} ± {self.transfer_results_['cv_scores_std']:.4f}"
        )

        return self

    def _compute_importance_stage(self, X_pretrain, y_pretrain, X_eval, y_eval,
                                  stage_name, eval_fit_kwargs, pretrain_fit_kwargs=None):
        """Compute feature importance for one stage of transfer learning"""

        fold_importances = []
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X_eval, y_eval)):
            print(f"  Processing {stage_name} fold {fold_idx + 1}...")

            # Split evaluation data
            X_train_fold, X_val_fold = X_eval[train_idx], X_eval[val_idx]
            y_train_fold, y_val_fold = y_eval[train_idx], y_eval[val_idx]

            # Create and train model
            if self.is_keras:
                model_fold = self.model_builder() if self.model_builder else self._clone_keras_model()

                # For transfer learning: first train on original data, then on transfer data
                if stage_name == "Transfer":
                    # Pre-train on original data
                    model_fold.fit(X_pretrain, y_pretrain, **pretrain_fit_kwargs)

                # Train/fine-tune on current fold data
                model_fold.fit(X_train_fold, y_train_fold, **eval_fit_kwargs)

                # Get predictions and compute baseline score
                y_pred = model_fold.predict(X_val_fold, verbose=0)
                if self.task_type == 'classification' and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                elif self.task_type == 'classification':
                    y_pred = (y_pred > 0.5).astype(int)
                else:
                    y_pred = y_pred.ravel()

                baseline_score = self.metric(y_val_fold, y_pred)

                # Compute permutation importance
                importances = self._compute_permutation_importance_keras(
                    model_fold, X_val_fold, y_val_fold, baseline_score
                )

            else:
                # sklearn model
                model_fold = self._clone_sklearn_model()

                # For transfer learning with sklearn (simulate by training on combined data)
                if stage_name == "Transfer":
                    X_combined = np.vstack([X_pretrain, X_train_fold])
                    y_combined = np.hstack([y_pretrain, y_train_fold])
                    model_fold.fit(X_combined, y_combined)
                else:
                    model_fold.fit(X_train_fold, y_train_fold)

                baseline_score = model_fold.score(X_val_fold, y_val_fold)

                # Use sklearn's permutation importance
                perm_result = permutation_importance(
                    model_fold, X_val_fold, y_val_fold,
                    n_repeats=self.n_repeats,
                    random_state=self.random_state,
                    scoring='neg_mean_squared_error' if self.task_type == 'regression' else 'accuracy'
                )
                importances = perm_result.importances_mean

            fold_importances.append(importances)
            fold_scores.append(baseline_score)

        # Compile results
        importances_by_fold = np.array(fold_importances)

        return {
            'feature_importances_': np.mean(importances_by_fold, axis=0),
            'feature_importances_std_': np.std(importances_by_fold, axis=0),
            'importances_by_fold_': importances_by_fold,
            'cv_scores_': fold_scores,
            'cv_scores_mean_': np.mean(fold_scores),
            'cv_scores_std_': np.std(fold_scores)
        }

    def _compute_permutation_importance_keras(self, model, X_val, y_val, baseline_score):
        """Compute permutation importance for Keras models"""
        importances = []

        for feature_idx in range(X_val.shape[1]):
            feature_scores = []

            for _ in range(self.n_repeats):
                # Create copy of validation data
                X_permuted = X_val.copy()

                # Permute the feature
                np.random.seed(self.random_state + feature_idx)
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

                # Get predictions on permuted data
                y_pred_permuted = model.predict(X_permuted, verbose=0)

                if self.task_type == 'classification' and y_pred_permuted.shape[1] > 1:
                    y_pred_permuted = np.argmax(y_pred_permuted, axis=1)
                elif self.task_type == 'classification':
                    y_pred_permuted = (y_pred_permuted > 0.5).astype(int)
                else:
                    y_pred_permuted = y_pred_permuted.ravel()

                # Compute score with permuted feature
                permuted_score = self.metric(y_val, y_pred_permuted)
                feature_scores.append(baseline_score - permuted_score)

            importances.append(np.mean(feature_scores))

        return np.array(importances)

    def fit(self, X, y, feature_names=None, **fit_kwargs):
        """
        Compute permutation importance across CV folds

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list, optional
            Names of features
        **fit_kwargs : dict
            Additional arguments for model fitting (e.g., epochs, batch_size for Keras)
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self.feature_names_ = feature_names
        n_features = X.shape[1]

        # Store importances for each fold
        fold_importances = []
        fold_scores = []

        print(
            f"Computing permutation importance across {self.cv.n_splits if hasattr(self.cv, 'n_splits') else 'CV'} folds...")

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            print(f"Processing fold {fold_idx + 1}...")

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and fit model
            if self.is_keras:
                model_fold = self.model_builder() if self.model_builder else self._clone_keras_model()

                # Set default Keras fitting parameters
                keras_kwargs = {
                    'epochs': 100,
                    'batch_size': 32,
                    'verbose': 0,
                    'validation_split': 0.1
                }
                keras_kwargs.update(fit_kwargs)

                # Fit the model
                model_fold.fit(X_train, y_train, **keras_kwargs)

                # Get predictions and compute baseline score
                y_pred = model_fold.predict(X_val, verbose=0)
                if self.task_type == 'classification' and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                elif self.task_type == 'classification':
                    y_pred = (y_pred > 0.5).astype(int)
                else:
                    y_pred = y_pred.ravel()

                baseline_score = self.metric(y_val, y_pred)

            else:
                # sklearn model
                model_fold = self._clone_sklearn_model()
                model_fold.fit(X_train, y_train)
                baseline_score = model_fold.score(X_val, y_val)

            fold_scores.append(baseline_score)

            # Compute permutation importance manually for Keras models
            if self.is_keras:
                importances = self._compute_permutation_importance_keras(
                    model_fold, X_val, y_val, baseline_score
                )
            else:
                # Use sklearn's permutation_importance
                perm_result = permutation_importance(
                    model_fold, X_val, y_val,
                    n_repeats=self.n_repeats,
                    random_state=self.random_state,
                    scoring=self.scoring
                )
                importances = perm_result.importances_mean

            fold_importances.append(importances)

        # Convert to numpy array for easier manipulation
        self.importances_by_fold_ = np.array(fold_importances)

        # Calculate mean and std across folds
        self.feature_importances_ = np.mean(self.importances_by_fold_, axis=0)
        self.feature_importances_std_ = np.std(self.importances_by_fold_, axis=0)

        # Store additional info
        self.cv_scores_ = fold_scores
        self.cv_scores_mean_ = np.mean(fold_scores)
        self.cv_scores_std_ = np.std(fold_scores)

        print(f"Done! CV Score: {self.cv_scores_mean_:.4f} ± {self.cv_scores_std_:.4f}")

        return self

    def _compute_permutation_importance_keras(self, model, X_val, y_val, baseline_score):
        """Compute permutation importance for Keras models"""
        importances = []

        for feature_idx in range(X_val.shape[1]):
            feature_scores = []

            for _ in range(self.n_repeats):
                # Create copy of validation data
                X_permuted = X_val.copy()

                # Permute the feature
                np.random.seed(self.random_state + feature_idx)
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

                # Get predictions on permuted data
                y_pred_permuted = model.predict(X_permuted, verbose=0)

                if self.task_type == 'classification' and y_pred_permuted.shape[1] > 1:
                    y_pred_permuted = np.argmax(y_pred_permuted, axis=1)
                elif self.task_type == 'classification':
                    y_pred_permuted = (y_pred_permuted > 0.5).astype(int)
                else:
                    y_pred_permuted = y_pred_permuted.ravel()

                # Compute score with permuted feature
                permuted_score = self.metric(y_val, y_pred_permuted)
                feature_scores.append(baseline_score - permuted_score)

            importances.append(np.mean(feature_scores))

        return np.array(importances)

    def _clone_keras_model(self):
        """Clone a Keras model"""
        if self.model_builder:
            return self.model_builder()
        else:
            # Clone existing model (this is tricky with Keras)
            model_config = self.model.get_config()
            new_model = keras.Model.from_config(model_config)
            new_model.compile(
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics
            )
            return new_model

    def _clone_sklearn_model(self):
        """Create a copy of sklearn model with same parameters"""
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()
            return type(self.model)(**params)
        else:
            return self.model

    def get_feature_ranking(self, stage='both'):
        """Get features ranked by importance for original, transfer, or both stages"""
        if self.original_results_ is None or self.transfer_results_ is None:
            raise ValueError("Must call fit_transfer_learning() first")

        if stage == 'original':
            return self._create_ranking_df(self.original_results_)
        elif stage == 'transfer':
            return self._create_ranking_df(self.transfer_results_)
        elif stage == 'both':
            # Create comparison dataframe
            orig_ranking = self._create_ranking_df(self.original_results_)
            transfer_ranking = self._create_ranking_df(self.transfer_results_)

            comparison_df = pd.DataFrame({
                'feature': orig_ranking['feature'],
                'original_importance': orig_ranking['importance_mean'],
                'original_std': orig_ranking['importance_std'],
                'transfer_importance': transfer_ranking['importance_mean'],
                'transfer_std': transfer_ranking['importance_std'],
                'importance_change': transfer_ranking['importance_mean'] - orig_ranking['importance_mean'],
                'original_rank': orig_ranking['rank'],
                'transfer_rank': transfer_ranking['rank'],
                'rank_change': orig_ranking['rank'] - transfer_ranking['rank']  # Positive means improved rank
            })

            # Sort by transfer importance
            comparison_df = comparison_df.sort_values('transfer_importance', ascending=False).reset_index(drop=True)

            return comparison_df
        else:
            raise ValueError("stage must be 'original', 'transfer', or 'both'")

    def _create_ranking_df(self, results):
        """Create ranking dataframe from results"""
        sorted_idx = np.argsort(results['feature_importances_'])[::-1]

        return pd.DataFrame({
            'feature': [self.feature_names_[i] for i in sorted_idx],
            'importance_mean': results['feature_importances_'][sorted_idx],
            'importance_std': results['feature_importances_std_'][sorted_idx],
            'rank': np.arange(1, len(sorted_idx) + 1)
        })

    def plot_transfer_comparison(self, crop, top_k=15, figsize=(15, 8)):
        """Plot comparison of feature importance before and after transfer learning"""
        if self.original_results_ is None or self.transfer_results_ is None:
            raise ValueError("Must call fit_transfer_learning() first")

        comparison_df = self.get_feature_ranking('both').head(top_k)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Side-by-side comparison
        x = np.arange(len(comparison_df))
        width = 0.35

        ax1.barh(x - width / 2, comparison_df['original_importance'], width,
                 label='Original', alpha=0.7, xerr=comparison_df['original_std'], capsize=3)
        ax1.barh(x + width / 2, comparison_df['transfer_importance'], width,
                 label='Transfer', alpha=0.7, xerr=comparison_df['transfer_std'], capsize=3)

        ax1.set_yticks(x)
        ax1.set_yticklabels(comparison_df['feature'])
        ax1.set_xlabel('Feature Importance')
        ax1.set_title(f'Feature Importance: Original vs Transfer {crop}')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()

        # Plot 2: Importance change
        colors = ['red' if x < 0 else 'green' for x in comparison_df['importance_change']]
        ax2.barh(x, comparison_df['importance_change'], color=colors, alpha=0.7)
        ax2.set_yticks(x)
        ax2.set_yticklabels(comparison_df['feature'])
        ax2.set_xlabel('Importance Change (Transfer - Original)')
        ax2.set_title('Change in Feature Importance')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()

        # Plot 3: Rank change
        rank_colors = ['red' if x < 0 else 'green' for x in comparison_df['rank_change']]
        ax3.barh(x, comparison_df['rank_change'], color=rank_colors, alpha=0.7)
        ax3.set_yticks(x)
        ax3.set_yticklabels(comparison_df['feature'])
        ax3.set_xlabel('Rank Change (Positive = Improved Rank)')
        ax3.set_title('Change in Feature Ranking')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(axis='x', alpha=0.3)
        ax3.invert_yaxis()

        # Plot 4: Scatter plot of original vs transfer importance
        ax4.scatter(comparison_df['original_importance'], comparison_df['transfer_importance'],
                    alpha=0.7, s=60)

        # Add diagonal line
        min_val = min(comparison_df['original_importance'].min(), comparison_df['transfer_importance'].min())
        max_val = max(comparison_df['original_importance'].max(), comparison_df['transfer_importance'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal Importance')

        ax4.set_xlabel('Original Importance')
        ax4.set_ylabel('Transfer Importance')
        ax4.set_title('Original vs Transfer Importance')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # Add feature labels to scatter plot
        for i, txt in enumerate(comparison_df['feature'][:10]):  # Label top 10 only
            ax4.annotate(txt, (comparison_df['original_importance'].iloc[i],
                               comparison_df['transfer_importance'].iloc[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_stability_comparison(self, top_k=10, figsize=(15, 6)):
        """Plot stability comparison between original and transfer learning"""
        if self.original_results_ is None or self.transfer_results_ is None:
            raise ValueError("Must call fit_transfer_learning() first")

        comparison_df = self.get_feature_ranking('both').head(top_k)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Get feature indices for top features
        top_features = comparison_df['feature'].tolist()
        feature_indices = [self.feature_names_.index(feat) for feat in top_features]

        # Plot 1: Original model stability
        orig_fold_data = self.original_results_['importances_by_fold_'][:, feature_indices].T
        ax1.boxplot(orig_fold_data, labels=top_features)
        ax1.set_title('Original Model: Importance Distribution Across Folds')
        ax1.set_ylabel('Permutation Importance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(alpha=0.3)

        # Plot 2: Transfer model stability
        transfer_fold_data = self.transfer_results_['importances_by_fold_'][:, feature_indices].T
        ax2.boxplot(transfer_fold_data, labels=top_features)
        ax2.set_title('Transfer Model: Importance Distribution Across Folds')
        ax2.set_ylabel('Permutation Importance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_k=None, figsize=(10, 6)):
        """Plot feature importance with error bars"""
        if self.feature_importances_ is None:
            raise ValueError("Must call fit() first")

        ranking_df = self.get_feature_ranking()

        if top_k is not None:
            ranking_df = ranking_df.head(top_k)

        plt.figure(figsize=figsize)

        # Create horizontal bar plot with error bars
        y_pos = np.arange(len(ranking_df))
        plt.barh(y_pos, ranking_df['importance_mean'],
                 xerr=ranking_df['importance_std'],
                 alpha=0.7, capsize=5)

        plt.yticks(y_pos, ranking_df['feature'])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance (Cross-Validation)\nError bars show ± 1 std across folds')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_importance_stability(self, top_k=10, figsize=(12, 6)):
        """Plot importance across different CV folds to assess stability"""
        if self.importances_by_fold_ is None:
            raise ValueError("Must call fit() first")

        # Get top k features
        ranking_df = self.get_feature_ranking().head(top_k)
        top_features = ranking_df['feature'].tolist()

        # Get indices of top features
        feature_indices = [self.feature_names_.index(feat) for feat in top_features]

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Importance across folds
        fold_data = self.importances_by_fold_[:, feature_indices].T

        ax1.boxplot(fold_data, labels=top_features)
        ax1.set_title('Feature Importance Distribution Across CV Folds')
        ax1.set_ylabel('Permutation Importance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(alpha=0.3)

        # Plot 2: Coefficient of variation (stability metric)
        cv_scores = self.feature_importances_std_[feature_indices] / np.abs(self.feature_importances_[feature_indices])

        ax2.bar(range(len(top_features)), cv_scores)
        ax2.set_title('Feature Importance Stability\n(Lower = More Stable)')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_xticks(range(len(top_features)))
        ax2.set_xticklabels(top_features, rotation=45)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

def create_keras_model(input_dim, task_type='regression'):
    """Create a Keras model builder function"""

    def model_builder():
        model = keras.Sequential([
            keras.layers.Dense(100, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1 if task_type == 'regression' else 1,
                               activation='linear' if task_type == 'regression' else 'sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='mse' if task_type == 'regression' else 'binary_crossentropy',
            metrics=['mae'] if task_type == 'regression' else ['accuracy']
        )
        return model

    return model_builder

def scale_x(scaler, X, fit=False):
    X_reshaped = X.reshape(-1, X.shape[-1])
    if fit:
        X_scaled = scaler.fit_transform(X_reshaped)
    else:
        X_scaled = scaler.transform(X_reshaped)
    return scaler, X_scaled.reshape(X.shape)

def evi2cum():
    for crop in ['spring_barley', 'winter_wheat', 'maize']:
        field_file = pd.read_csv(f'Data/M/TL/{crop}_field.csv')
        reg_file = pd.read_csv(f'Data/M/TL/{crop}_regional.csv')
        evi_cols = [a for a in reg_file.columns if a.startswith('evi')]
        for i in range(1, len(evi_cols)):
            field_file[evi_cols[i]] = field_file[evi_cols[:i + 1]].sum(axis=1)
            reg_file[evi_cols[i]] = reg_file[evi_cols[:i + 1]].sum(axis=1)
        field_file.to_csv(f'Data/M/TL/cum/{crop}_field.csv', index=False)
        reg_file.to_csv(f'Data/M/TL/cum/{crop}_regional.csv', index=False)



if __name__ == '__main__':
    #Later: deep learn, other numbers of features for FS-> self optimize number of features
    pd.set_option('display.max_columns', None)
    warnings.filterwarnings('ignore')
    start_pro = datetime.now()
    # crop = 'spring_barley'
    # tab_diff_sample(crop)
    # ua_line('Results/SC2/202506/loocv/country/lstm_test2')

    # for country in ['EU']:
    #     for crop in ['spring_barley', 'winter_wheat', 'maize']:
    #         a = nc2table(country=country, crop=crop)
    #         data = 'sm'
    #         # a.resample_data_EU(data_source=data, temp_step='2W', anoms=True)
    #         # a.resample_data_EU(data_source=data, temp_step='2W', anoms=False)
    #         a.merge_EU(temp_step='2W', abs=False)
    #         a.merge_EU(temp_step='2W', abs=True)

    # detrend_yield()

    print(f'calculation stopped and took {datetime.now() - start_pro}')



    # Further ideas:
    # Find optimal number of features
    # always merge monthly but starting every two weeks
    # Global model???
    # feature_selection based on crosscors




    

