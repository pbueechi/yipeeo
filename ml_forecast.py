import os
import itertools
import csv
import seaborn
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import tensorflow as tf
from cmcrameri import cm as cmr
from pandas.tseries.offsets import DateOffset
from glob import glob
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.dummy import DummyRegressor
# from tensorflow import keras

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
        self.harvest_date = {'common winter wheat': [7,25], 'grain maize and corn-cob-mix': [10,10], 'spring barley': [7,20]}   #ToDo: dont hardcode harvest dates
        self.crop_data = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale.shp')
        self.lead_times = ['_LT4','_LT3','_LT2','_LT1']

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

        if not os.path.exists(f'Data/{temp_step}'):
            os.makedirs(f'Data/{temp_step}')
        if self.farm:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.farm}_{self.crop}_s1.csv')
        else:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}_{self.crop}_s1.csv')

    def resample_s2(self, temp_step='M'):
        """
        Extracts Sentinel-2 data from nc files and saves them as csv
        :param temp_step: str either M or 2W for aggregating the data monthly or Biweekly
        :return: saves a csv file which will be used for the ML
        """
        if temp_step=='2W':
            self.lead_times = ['_LT8', '_LT7', '_LT6', '_LT5','_LT4', '_LT3', '_LT2', '_LT1']
        params = ['ndvi', 'evi', 'ndwi', 'nmdi']
        if self.farm:
            inds = np.where((self.crop_data.farm_code==self.farm)&(self.crop_data.crop_type==self.crop)&(self.crop_data.c_year>2015))[0]
            pred_file_path = rf'D:\DATA\yipeeo\Predictors\S2_L2A\{self.country}\{self.farm}\nc'
        else:
            inds = np.where((self.crop_data.country_co==self.country)&(self.crop_data.crop_type==self.crop)&(self.crop_data.c_year>2015))[0]
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

        if not os.path.exists(f'Data/{temp_step}'):
            os.makedirs(f'Data/{temp_step}')
        if self.farm:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.farm}_{self.crop}_s2.csv')
        else:
            pipeline_df.to_csv(f'Data/{temp_step}/{self.country}_{self.crop}_s2.csv')

    def merge_tabs(self,temp_step):
        """
        :param temp_res: str either M or 2W for aggregating the data monthly or Biweekly
        :return: merges all predictor csv file to one single csv file. Needs to be used if both S-1 and S-2 data should
        bes used as predictors at the same time
        """
        if self.farm:
            path = f'Data/{temp_step}/{self.farm}_{self.crop}*.csv'
        else:
            path = f'Data/{temp_step}/{self.country}_{self.crop}*.csv'
        files = glob(path)
        for i, file in enumerate(files):
            if i==0:
                csv_fin = pd.read_csv(file, index_col=0)
            else:
                csv_next = pd.read_csv(file, index_col=0)
                if not (len(csv_fin.c_year)==np.sum(csv_fin.c_year==csv_next.c_year)) and (len(csv_fin.field_id)==np.sum(csv_fin.field_id==csv_next.field_id)):
                    raise ValueError('the files do not correspond')
                csv_fin = csv_fin.merge(csv_next)
        if self.farm:
            csv_fin.to_csv(f'Data/{temp_step}/{self.farm}_{self.crop}_all.csv')
        else:
            csv_fin.to_csv(f'Data/{temp_step}/{self.country}_{self.crop}_all.csv')


class ml:
    def __init__(self, crop, country, farm=None):
        self.crop = crop
        self.country = country
        self.farm = farm

    #---------------------------------------------- Exploratory data analysis ----------------------------------------
    def calc_corrs(self):
        """
        :return: calculates correlations between predictors and crop yields
        """
        file = pd.read_csv(f'Data/M/{self.farm}_{self.crop}.csv', index_col=0)
        predictors = file.columns[3:]
        crop_yield = file.loc[:, 'yield']
        for predictor in predictors:
            cor_file = pd.DataFrame([crop_yield, file.loc[:, predictor]]).transpose()
            cor_file = cor_file.dropna(axis=0)
            cor = spearmanr(cor_file.iloc[:, 0], cor_file.iloc[:, 1])
            print(f'{predictor} correlation with yield: {cor[0]} with p: {cor[1]}')

    def cross_cor_predictors(self, lt=None, preds=None):
        file = pd.read_csv(f'Data/2W/{self.country}_{self.crop}_all.csv', index_col=0).iloc[:,3:-1]
        file = self.select_preds(file=file, lt=lt, preds=preds)
        matrix = file.corr(method='pearson')
        plt.figure(figsize=(15, 15), dpi=300)
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        seaborn.heatmap(matrix, annot=True, mask=mask, cmap=cmr.vik_r, vmin=-0.9, vmax=0.9)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45)
        # plt.show()
        if lt:
            plt.savefig(f'Figures/Predictor_analysis/cor_matrix_{lt}.png', dpi=300)
        if preds:
            plt.savefig(f'Figures/Predictor_analysis/cor_matrix_small.png', dpi=300)
        if not lt or preds:
            plt.savefig('Figures/Predictor_analysis/cor_matrix_all.png', dpi=300)

    #---------------------------------------------- Set up models ----------------------------------------------------
    def runforecast(self, lead_time, predictors, model='RF'):
        file = pd.read_csv(f'Data/2W/{self.country}_{self.crop}_{predictors}.csv', index_col=0)
        file = file.dropna(axis=0)
        predictors = [p for p in file.columns[3:] if not p == 'date_last_obs']
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        predictor_file = self.merge_previous_months(file.loc[:, used_predictors])
        # predictor_file = file.loc[:, used_predictors]
        X, X_test, y, y_test = train_test_split(predictor_file, file.loc[:, 'yield'], test_size=0.2, random_state=5)

        pipe = Pipeline([('scalar', StandardScaler()), ('clf', self.get_default_regressions(model))])
        scores_kf = cross_val_score(pipe, X, y, cv=10, scoring="explained_variance")
        print(f'train perf for lead_time:{lead_time}, R^2: {np.nanmean(scores_kf)}')
        return scores_kf

    def runforecast_loocv(self, lead_time, preds=None, model='RF'):
        min_obs = 15
        file = pd.read_csv(f'Data/2W/{self.country}_{self.crop}_all.csv', index_col=0)
        file = file.dropna(axis=0)
        if preds:
            predictors = [p for p in file.columns if p.startswith(tuple(preds))]
        else:
            predictors = [p for p in file.columns[3:] if not p == 'date_last_obs']
        used_predictors = [a for a in predictors if int(a[-1]) >= lead_time]
        predictor_file = self.merge_previous_months(file.loc[:, used_predictors])
        years = file.c_year
        # value_counts()
        years_obs = years.value_counts()
        years_test = years_obs[years_obs>min_obs].index

        # predictor_file = file.loc[:, used_predictors]

        # X, X_test, y, y_test = train_test_split(predictor_file, file.loc[:, 'yield'], test_size=0.2, random_state=5)
        #
        # pipe = Pipeline([('scalar', StandardScaler()), ('clf', self.get_default_regressions(model))])
        #
        # pipe.fit(X, y)
        # y_test_pred = pipe.predict(X_test)
        # y_train_pred = pipe.predict(X)
        # print(f'test perf: {pearsonr(y_test_pred,y_test)[0]}')
        # print(f'train perf: {pearsonr(y_train_pred, y)[0]}')

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
        cols = ['lead_time', 's1', 's2', 'all']
        lead_times = [8, 7, 6, 5, 4, 3, 2, 1]
        csv_file = pd.DataFrame(data=None, index=lead_times, columns=cols)
        csv_file.loc[:, 'lead_time'] = lead_times
        for lead_time in lead_times:
            for pred in cols[1:]:
                csv_file.loc[lead_time, pred] = self.runforecast(country=self.country, crop=self.crop,
                                                                 lead_time=lead_time, predictors=pred)
        csv_file.to_pickle('Results/Validation/s1_vs_s2.csv')

    def plot_res(self):
        """
        :return: Plots the results generated by self.write_results to a boxplot comparing the performance of the model
                    using S-1, S-2, and all data
        """
        file = pd.read_pickle('Results/Validation/s1_vs_s2.csv')
        s1 = file.loc[:, ['lead_time', 's1']]
        s2 = file.loc[:, ['lead_time', 's2']]
        all = file.loc[:, ['lead_time', 'all']]
        s1 = s1.rename(columns={'s1': 'perf'})
        s2 = s2.rename(columns={'s2': 'perf'})
        all = all.rename(columns={'all': 'perf'})
        s1.loc[:, 'predictors'] = ['s1'] * len(s1.lead_time)
        s2.loc[:, 'predictors'] = ['s2'] * len(s2.lead_time)
        all.loc[:, 'predictors'] = ['all'] * len(all.lead_time)

        final = pd.concat([s1, s2, all])
        s = seaborn.boxplot(data=final.explode('perf'), x='lead_time', y='perf', hue='predictors')
        s.set_xticklabels(np.linspace(2, 16, 8))
        s.set_xlabel('Lead Time [weeks]')
        s.set_ylabel('explained variance')
        plt.savefig('Figures/performance_wheat_s1_2.png', dpi=300)

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




if __name__ == '__main__':
    pd.set_option('display.max_columns', 15)
    start_pro = datetime.now()
    crops = ['grain maize and corn-cob-mix','common winter wheat','spring barley']
    # for crop in crops:
    #     a = nc2table(country='cz', crop=crop)
    #     # a.resample_s1(temp_step='M')
    #     # a.resample_s2(temp_step='M')
    #     a.merge_tabs(temp_res='2W')
    #     a.merge_tabs(temp_res='M')
    # write_results()
    # plot_res()
    # for lead_time in [8, 7, 6, 5, 4, 3, 2, 1]:
    a = ml(crop=crops[1], country='cz')
    a.runforecast_loocv(lead_time=1, preds=['sig40_vh','ndvi'], )
    # a.cross_cor_predictors(preds=['sig40_vh','ndvi'])
    # for lead_time in [2,1]:
    #     a = ml()
        # a.runforecast(country='cz',crop=crops[1], lead_time=lead_time)
        # a.rundeepcast(farm='rost',crop=crop[1], lead_time=1)
    print(f'calculation stopped and took {datetime.now() - start_pro}')

    
    """
    Cloud mask:
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
    https://github.com/sentinel-hub/sentinel2-cloud-detector
    ECOSTRESS:
    https://cmr.earthdata.nasa.gov/search/
    """

