import os
import math
import itertools
import random
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

def agg_s1(path, crop, country, subsample=False):
    path_from = os.path.join(path, country, crop, 'daily')
    path_write = r'D:\data-write\YIPEEO\predictors\S1'
    path_write = os.path.join(path_write, country, crop)
    if not os.path.exists(path_write):
        os.makedirs(path_write)
    files = os.listdir(path_from)
    if country=='AUT':
        region_ind = 1
    else:
        region_ind = 3
    regions = [a.split('_')[region_ind] for a in files]
    regions_unique = np.unique(regions)

    for region in regions_unique[26:]:
        used_files = [a for a in files if a.split('_')[region_ind]==region]
        if subsample:
            if len(used_files)>1500:
                used_files = random.sample(used_files,1500)
        print(region, len(used_files), datetime.now())
        variables = ['sig0_vv_mean_daily', 'sig0_vh_mean_daily', 'sig0_cr_mean_daily', 'sig40_vv_mean_daily', 'sig40_vh_mean_daily', 'sig40_cr_mean_daily']
        ts = {var: pd.DataFrame(data=None, index=pd.date_range(start='1/1/2016', end='31/12/2022')) for var in variables}
        for file in used_files:
            nc_file = xr.open_dataset(os.path.join(path_from, file))
            time_nc = nc_file.time.values
            for var in variables:
                vals = list(itertools.chain(*nc_file[var].values))
                nc_ser = pd.DataFrame(data=convert_db(vals, to=False), index=time_nc, columns=[file])
                ts[var] = ts[var].join(nc_ser)
        for v, var in enumerate(variables):
            means = pd.DataFrame(convert_db(ts[var].mean(axis=1), to=True), index=ts[var].index, columns=[var])
            if v==0:
                xr_file = xr.Dataset.from_dataframe(means)
            else:
                xr_file[var] = xr.DataArray.from_series(means.squeeze())

            xr_file[var].attrs = nc_file[var].attrs
        xr_file = xr_file.rename({'index': 'time'})
        xr_file.to_netcdf(path=os.path.join(path_write, f'{region}.nc'))

def convert_db(a_list, to):
    if to:
        a_conv = [10 * math.log(a, 10) for a in a_list]
    else:
        a_conv = [10 ** (a / 10) for a in a_list]
    return a_conv

if __name__=='__main__':
    path = r'D:\DATA\yipeeo\Predictors\S1'
    for crop in ['winter_wheat']:
        start = datetime.now()
        agg_s1(path, crop=crop, country='AUT', subsample=True)
        print(f'calculation for {crop} took {datetime.now()-start}')