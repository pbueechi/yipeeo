import os
import itertools
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.dummy import DummyRegressor
from tensorflow import keras


def flag_clouds():
    pass

def load_data(country, farm=None, crop='wheat'):
    harvest_month = {'common winter wheat':7,'grain maize and corn-cob-mix':10, 'spring barley':7}
    crop_data = gpd.read_file(r'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale.shp')
    if farm:
        inds = np.where((crop_data.farm_code==farm)&(crop_data.crop_type==crop)&(crop_data.c_year>2015))[0]
        pred_file_path = rf'D:\DATA\yipeeo\Predictors\S2_L2A\{country}\{farm}\nc'
    else:
        inds = np.where((crop_data.country_co==country)&(crop_data.crop_type==crop)&(crop_data.c_year>2015))[0]
        pred_file_path = rf'D:\DATA\yipeeo\Predictors\S2_L2A\{country}\nc'
    crop_data = crop_data.iloc[inds,:]

    pipeline_df = crop_data.iloc[:,[1,5,10]]
    pipeline_df.index = range(len(pipeline_df.index))
    params = ['ndvi', 'evi', 'ndwi', 'nmdi']
    lead_times = ['_LT3','_LT2','_LT1','_LT0']
    col_names = [[a+b for a in params] for b in lead_times]
    col_names = list(itertools.chain(*col_names))
    pipeline_df.loc[:,col_names] = np.nan

    for df_ind, field, year in zip(pipeline_df.index, pipeline_df.field_id,pipeline_df.c_year):
        for param in params:
            this_cols = [a for a in pipeline_df.columns if a.startswith(param)]
            s2 = xr.open_dataset(os.path.join(pred_file_path, f'{field}.nc'))
            ndvi = s2[param].to_series()
            ndvi_m = ndvi.resample('M').mean()
            #ToDo dont hardcode range of selected months
            year_ind = np.where((ndvi_m.index.year==year)&(ndvi_m.index.month<harvest_month[crop]+1)&(ndvi_m.index.month>harvest_month[crop]-4))[0]
            pipeline_df.loc[df_ind, this_cols] = ndvi_m[year_ind].values

    pipeline_df.to_csv(f'Data/{farm}_{crop}.csv')

def calc_corrs(crop, farm):
    file = pd.read_csv(f'Data/{farm}_{crop}.csv', index_col=0)
    predictors = file.columns[3:]
    print(file)
    crop_yield = file.loc[:,'yield']
    for predictor in predictors:
        cor_file = pd.DataFrame([crop_yield,file.loc[:,predictor]]).transpose()
        cor_file = cor_file.dropna(axis=0)
        cor = pearsonr(cor_file.iloc[:,0],cor_file.iloc[:,1])
        print(f'{predictor} correlation with yield: {cor[0]} with p: {cor[1]}')

def runforecast(crop, farm, lead_time):
    file = pd.read_csv(f'Data/{farm}_{crop}.csv', index_col=0)
    file = file.dropna(axis=0)
    predictors = file.columns[3:]
    used_predictors = [a for a in predictors if int(a[-1])>=lead_time]

    X, X_test, y, y_test = train_test_split(file.loc[:,used_predictors], file.loc[:,'yield'], test_size=0.2, random_state=5)

    pipe = Pipeline([('scalar', StandardScaler()), ('clf', RandomForestRegressor(100))])
    # pipe = Pipeline([('scalar', StandardScaler()), ('clf', LinearRegression())])
    # pipe = Pipeline([('scalar', StandardScaler()), ('clf', DummyRegressor())])
    # pipe = Pipeline([('scalar', StandardScaler()), ('clf', xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, colsample_bytree=0.3))])
    # pipe.fit(X, y)
    # y_test_pred = pipe.predict(X_test)
    # y_train_pred = pipe.predict(X)
    # print(f'test perf: {pearsonr(y_test_pred,y_test)[0]}')
    # print(f'train perf: {pearsonr(y_train_pred, y)[0]}')

    scores_kf = cross_val_score(pipe, X, y, cv=5, scoring="explained_variance")
    print(f'train perf for lead_time:{lead_time}, R^2: {np.nanmean(scores_kf)}')

def rundeepcast(crop, farm, lead_time):
    # file = pd.read_csv(f'Data/{farm}_{crop}.csv', index_col=0)
    # file = file.dropna(axis=0)
    # predictors = file.columns[3:]
    # used_predictors = [a for a in predictors if int(a[-1])>=lead_time]
    #
    # X, X_test, y, y_test = train_test_split(file.loc[:,used_predictors], file.loc[:,'yield'], test_size=0.2, random_state=5)
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




if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    start_pro = datetime.now()
    # load_data(country='czr',farm='rost',crop='spring barley')
    # calc_corrs(farm='rost',crop='common winter wheat')
    # Crops with most datapoints: grain maize and corn-cob-mix, common winter wheat, spring barley
    crop = ['grain maize and corn-cob-mix','common winter wheat','spring barley']
    for lead_time in [3,2,1,0]:
        runforecast(farm='rost',crop=crop[1], lead_time=lead_time)
    # print(f'calculation stopped and took {datetime.now() - start_pro}')
    # rundeepcast(farm='rost',crop=crop[1], lead_time=1)
    
    """
    Cloud mask:
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
    https://github.com/sentinel-hub/sentinel2-cloud-detector
    ECOSTRESS:
    https://cmr.earthdata.nasa.gov/search/
    """

