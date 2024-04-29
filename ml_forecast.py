#%%
import os
import itertools
import csv
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
from cmcrameri import cm as cmr
from pandas.tseries.offsets import DateOffset
from glob import glob
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
# from tensorflow import keras
import inspect
#%%
#ToDo: remove farms and just use ukr_horod as country name
class nc2table:
    """
    The Crop yield data and predictors are saved as nc files. This class extracts the predictors for the four months
    before harvest either in biweekly or monthly timesteps and saves them as csv files.
    """
    def __init__(self, country, crop, yield_data_file, farm=None):
        self.country = country
        self.crop = crop
        self.farm = farm
        # Harvest dates in CZR around 25 July for winter wheat, 10 Oct for Maize, and 20 Jul Spring barley
        self.harvest_date = {'common winter wheat': [7,25], 'grain maize and corn-cob-mix': [10,10], 'spring barley': [7,20]}   #ToDo: dont hardcode harvest dates
        self.crop_data = gpd.read_file(yield_data_file)
        self.lead_times = ['_LT4','_LT3','_LT2','_LT1']
        
    def resample_s1(self, s1_path, temp_step='M'):
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
        pred_file_path = s1_path
        self.crop_data = self.crop_data.iloc[inds,:]
        
        pipeline_df = self.crop_data.iloc[:,[1,5,10]] #seems to be columns number for field id, year and yield in the shape file
        pipeline_df.index = range(len(pipeline_df.index))
        col_names = [[a+b for a in params] for b in self.lead_times]
        col_names = list(itertools.chain(*col_names))
        pipeline_df.loc[:,col_names] = np.nan
        
        for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id,pipeline_df.c_year):
            #field = field.split('_')[-1] #field id for other countries not in same format
            #path_nc = os.path.join(pred_file_path, f'{self.country}_{field}_{year}_cleaned_cr_agg_daily.nc')
            path_nc = os.path.join(pred_file_path, f'{field}_{year}_cleaned_cr_agg_daily.nc')
            if not os.path.exists(path_nc):
                #print(f'file {self.country}_{field}_{year}_cleaned_cr_agg_daily.nc does not exist')
                print(f'file {field}_{year}_cleaned_cr_agg_daily.nc does not exist')
                continue
            s1 = xr.open_dataset(path_nc)
            for param in params:
                this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
                vals = list(itertools.chain(*s1[param].values))
                ndvi = pd.Series(vals, s1.time)
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

    def resample_s2(self, s2_path, temp_step='M'):
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
        pred_file_path = s2_path
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
    '''
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
        pred_file_path = os.path.join(ecostress_file_path, self.country, 'nc')
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
    '''
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
        files = [a for a in glob(path) if a.endswith(tuple(['s1.csv', 's2.csv']))]
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
    '''
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
    '''
class ml:
    def __init__(self, crop, country, temp_res='M', farm=None):
        self.crop = crop
        self.country = country
        self.farm = farm
        self.temp_res = temp_res

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
        file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.crop}_all.csv', index_col=0).iloc[:,2:-1]
        file = self.select_preds(file=file, lt=lt, preds=preds)
        ticknames = [i.replace('_mean_daily', '') for i in file.columns]
        file.columns = ticknames
        matrix = file.corr(method='pearson')
        plt.figure(figsize=(15, 15), dpi=300)
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        ax = seaborn.heatmap(matrix, annot=True, mask=mask, cmap=cmr.vik_r, vmin=-0.9, vmax=0.9)
        plt.yticks(rotation=0, fontsize=20)
        plt.xticks(rotation=45, fontsize=20)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        plt.show()
        # if lt:
        #     plt.savefig(f'Figures/Predictor_analysis/cor_matrix_{lt}_{self.temp_res}.png', dpi=300)
        # if preds:
        #     plt.savefig(f'Figures/Predictor_analysis/cor_matrix_fewpreds_{self.temp_res}.png', dpi=300)
        # if not lt and not preds:
        #     plt.savefig(f'Figures/Predictor_analysis/cor_matrix_all_{self.temp_res}.png', dpi=300)

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
            selected_features = [p for p in X.columns if p.startswith(tuple(['sig40_vh','evi']))]

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
            years_test = years_obs[years_obs > min_obs].index #Are we taking less data to train and more data to test?

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

    def rundeepcast(self, lead_time):
        pass
        # file = pd.read_csv(f'Data/2W/{self.country}_{self.crop}.csv', index_col=0)
        # file = file.dropna(axis=0)
        # predictors = file.columns[3:]
        # used_predictors = [a for a in predictors if int(a[-1])>=lead_time]
        #
        # X, X_test, y, y_test = train_test_split(file.loc[:,used_predictors], file.loc[:,'yield'], test_size=0.2, random_state=5)
        """
        mnist = keras.datasets.mnist
        (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
        # plt.imshow(x_train_full[0])
        # plt.show()
        x_train_norm = x_train_full / 255.
        x_test_norm = x_test / 255.

        x_valid, x_train = x_train_norm[:5000], x_train_norm[5000:]
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

        x_test = x_test_norm
        print(x_train.shape)
        """
        # model = keras.models.Sequential()
        #
        # model.add(keras.layers.Flatten(input_shape=[28, 28]))
        # model.add(keras.layers.Dense(300, activation="relu"))
        # model.add(keras.layers.Dense(100, activation="relu"))
        # model.add(keras.layers.Dense(10, activation="softmax"))
        #
        # model.summary()
        #
        # model.compile(loss="sparse_categorical_crossentropy",
        #               optimizer="sgd",
        #               metrics=["accuracy"])
        #
        # model_history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))
        # print(model.evaluate(x_test,y_test))

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
                    selected_features = [a for a in X.columns if not a.startswith(tuple(['ndvi','evi','ndwi','nmdi']))]
                elif predictor=='s2':
                    selected_features = [a for a in X.columns if a.startswith(tuple(['ndvi', 'evi', 'ndwi', 'nmdi']))]
                else:
                    selected_features = X.columns

                X, X_test = X.loc[:, selected_features], X_test.loc[:, selected_features]
                # hp_tuned_vals = self.hyper_tune(lead_time=lead_time, model=model, selected_features=selected_features)
                hp_tuned_vals = {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.3, 'colsample_bytree': 0.5}
                estimator = self.get_default_regressions(model)
                estimator.set_params(**hp_tuned_vals)
                pipe = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
                print(X.columns)
                scores_kf_hp = cross_val_score(pipe, X, y, cv=30, scoring="explained_variance")
                print(f'train perf for lead_time:{lead_time} R^2: {np.median(scores_kf_hp)}')
                csv_file.loc[lead_time, predictor] = scores_kf_hp
        csv_file.to_pickle(f'Results/Validation/{self.crop}_{model}_s1_s2_{self.temp_res}_unmerged.csv')
    '''
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
                    selected_features = [a for a in X.columns if a.startswith(tuple(['ndvi', 'evi', 'ndwi', 'nmdi']))]
                elif predictor=='s1-s2':
                    selected_features = [a for a in X.columns if a.startswith(tuple(['sig','ndvi', 'evi', 'ndwi', 'nmdi']))]
                elif predictor=='eco':
                    selected_features = [a for a in X.columns if a.startswith('ECO2LSTE')]
                else:
                    selected_features = [a for a in X.columns if a.startswith(tuple(['sig','ndvi', 'evi', 'ndwi', 'nmdi', 'ECO']))]

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
    '''
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
        s = seaborn.boxplot(data=final.explode('perf'), x='lead_time', y='perf', hue='predictors', zorder=2)
        if self.temp_res=='2W':
            s.set_xticklabels(np.linspace(2, 16, 8))
        elif self.temp_res=='M':
            s.set_xticklabels(np.linspace(1, 4, 4))
        s.set_xlabel('Lead Time [months]')
        s.set_ylabel('explained variance')
        s.set_title(self.crop)
        s.set_ylim([-2,1])
        lw=0.5
        lin_col = 'gray'
        alpha = 0.5
        s.legend_.set_title(None)
        [s.axhline(x + .1, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        [s.axhline(x + .2, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        [s.axhline(x + .3, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        [s.axhline(x + .4, color=lin_col, linewidth=lw, alpha=alpha, zorder=1) for x in s.get_yticks()]
        [s.axhline(x, color='k', linewidth=lw, zorder=1) for x in s.get_yticks()]
        # plt.show()
        plt.savefig(fr'M:\Projects\YIPEEO\04_deliverables_documents\03_ATBD\Figs\ml_validation/{comp}_{self.crop}_{self.temp_res}-2.png', dpi=300)
        plt.close()


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
            LASSO = Lasso(eps=0.001,
                          n_alphas=100,
                          alphas=None,
                          fit_intercept=True,
                          normalize=True,
                          precompute='auto',
                          max_iter=1000,
                          tol=0.0001,
                          copy_X=True,
                          cv=5,
                          verbose=False,
                          n_jobs=-1,
                          positive=False,
                          random_state=None,
                          selection='cyclic')
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
            file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.farm}_{self.crop}_all.csv', index_col=0)
        else:
            file = pd.read_csv(f'Data/{self.temp_res}/{self.country}/{self.crop}_all.csv', index_col=0)
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


#%%
if __name__ == '__main__':
    #Ideally i want to put filenames right in the beginning of the code instead of changing them in the classes
    
    #specify the working directory as well because the files would flood the code folder
    working_dir = '/data/yipeeo_wd'
    #and change the directory in the system to make it effective else it does not change anything
    os.chdir(working_dir)
    
    parent_dir = '/home/nluintel/shares/climers/Projects/YIPEEO/07_data'
    yield_data_file = os.path.join(parent_dir, 'Crop yield', 'Database', 'field_scale_lleida.shp')
    s1_file_path = os.path.join(parent_dir, 'Predictors', 'eo_ts', 's1','daily')
    s2_file_path = os.path.join(parent_dir, 'Predictors', 'eo_ts', 's2', 'Spain', 'lleida', 'nc')
    #ecostress_file_path = os.path.join(parent_dir, 'Predictors', 'eo_ts', 'ECOSTRESS')
    
    

    #Later: deep learn, ECOSTRESS, other numbers of features for FS-> self optimize number of features
    pd.set_option('display.max_columns', 15)
    warnings.filterwarnings('ignore')
    start_pro = datetime.now()

    # Predictors resampling
    crops = ['common winter wheat','grain maize and corn-cob-mix','spring barley']
    # crop_summary_stats()
    # plot_crop_summary()
    for crop in crops[:1]:
        a = nc2table(country='es', crop=crop, yield_data_file=yield_data_file)
        #a.resample_ecostress(temp_step='M')
        a.resample_s1(s1_file_path, temp_step='M')
        a.resample_s2(s2_file_path, temp_step='M')
        a.previous_crop()
        a.merge_s1_s2(temp_step='M')
        #a.merge_all(temp_step='M')


    # # Forecasting
    # for crop in crops[:1]:
    #     a = ml(crop=crop, country='cz', farm='rost', temp_res='2W')
    #     # a.s1_vs_s2()
    #     a.plot_res(comp='s1s2')
    #     # a.s1_vs_s2()
    #     # a.cross_cor_predictors(lt=1)
    # #     a.feature_imp(lead_time=1)
    #     # a = ml(crop=crop, country='cz', temp_res='M')
    #     # for model in ['XGB','RF']:
    #     #     a.runforecast_country(model=model, optimize=True)

    #     # a.s1_vs_s2_eco(model='XGB')
    #     # a.plot_res(comp='eco')
    # # for country in ['ua', 'nl']:
    # #     a = ml(crop=crops[1], country=country, temp_res='M')
    # #     a.runforecast_country(model='RF', optimize=True)
    # # for lead_time in [4,3,2,1]:
    # #     a.runforecast(lead_time=lead_time, model='XGB', feature_select=True, hyper_tune=True)


    # # a.write_results()
    # # a.plot_res()
    # # a.runforecast_loocv(lead_time=1, preds=['sig40_vh','ndvi'])
    # # for lead_time in [2,1]:
    # #     a = ml()
    #     # a.runforecast(country='cz',crop=crops[1], lead_time=lead_time)
    #     # a.rundeepcast(farm='rost',crop=crop[1], lead_time=1)
    # print(f'calculation stopped and took {datetime.now() - start_pro}')

    # #ToDo: today:
    # # check test / train performance

    # #ToDo: later:
    # # Find optimal number of features
    # # LSTM
    # # always merge monthly but starting every two weeks
    # # Global model???
    # # feature_selection based on crosscors


# %%
