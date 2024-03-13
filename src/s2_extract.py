import os, sys
import time
import gc
import pystac_client
import planetary_computer
import rasterio
import csv
import warnings
import glob
import cProfile
import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
import xarray as xr
from scipy import stats as st
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from rasterio.mask import mask
from pystac.extensions.eo import EOExtension as eo

#ToDo: this code needs to be run separately for the three crops winter wheat, maize and spring barley for the countries CZR and Austria
class s2:
    """
    Class to retrieve sentinel-2 L2A data and extract it to field level
    """
    # ToDo: There are several ways to save some computation time. First, change bands to the second list below.
    #  Afterwards, self.time_of_interest can be adjusted in def __init__
    bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']
    # bands = ['B02','B04','B08','B11','B12','SCL']

    def __init__(self, region, year, crop, data_dir, out_dir):
        """
        :param region: str of name of location. For the example dataset rost must be used. For NUTS3 level different name can be used
        """
        self.region = region
        self.crop = crop
        self.year = year
        #ToDo Replace the two following links with where the output should be stored and the link to the shapefile with the fields

        self.table_path = f'{out_dir}/predictors/S2_L2A/ts'
        self.fields = gpd.read_file(f'{data_dir}/{region}/nuts/{crop}_{year}.shp')
        #self.table_path = 'D:/data-write/YIPEEO/predictors/S2_L2A/ts'
        #self.fields = gpd.read_file(f'D:/DATA/yipeeo/Crop_data/Crop_class/{region}/nuts/{crop}_{year}.shp')
        
        self.fields = self.fields.set_crs(epsg=4326)
        self.csv_path = os.path.join(self.table_path, region, crop, str(year))
        if not os.path.exists(self.csv_path): os.makedirs(self.csv_path)
        self.fields = self.fields.drop_duplicates(subset='FS_KENNUNG')
        self.row_head = ['observation', 'cloud_cover[%]'] + [int(i) for i in list(self.fields.FS_KENNUNG.values)]
        self.time_of_interest = f'{self.year}-01-01/{self.year}-10-31'
        # Maize efficient version
        # time_of_interest = f'{self.year}-04-01/{self.year}-10-31'
        # wheat, spring barley efficient version
        # time_of_interest = f'{self.year}-01-01/{self.year}-07-31'

    def extract_s2(self, items, core=None):
        """
        :param items: list of s2 images as returned from self.find_s2_items
        :param core: int number of core on which code is run. if not parallelized use None
        :param new: Boolean leave false if there are already some results stored. True if its the first run -> potentially existing files will be removed!
        :return: writes csv files with the s2 information per band and core
        """
        # Start looping through all S-2 scenes and defined S-2 bands
        try:
            for it_num, item in enumerate(items):
                print(f'core={core}: {it_num}/{len(items)} from {item.datetime.date()} at {datetime.now()}')
                signed_item = planetary_computer.sign(item)

                # Define extent of current scene and reduce fields to the ones in this extent
                exts = pd.DataFrame(item.geometry['coordinates'][0], columns=['x', 'y'])
                xmin, ymin, xmax, ymax = exts.min(axis=0).x, exts.min(axis=0).y, exts.max(axis=0).x, exts.max(axis=0).y
                used_fields = self.fields.clip(mask=[xmin, ymin, xmax, ymax])
                for band in self.__class__.bands:
                    # Load S2 scenes

                    # scao
                    sign_href = planetary_computer.sign(signed_item.assets[band].href)
                    #src = rasterio.open(signed_item.assets[band].href)
                    src = rasterio.open(sign_href)

                    fields = used_fields.to_crs(src.crs)
                    # Per scene and band extract and summarize the information per field
                    fields_data = pd.DataFrame(data=None, index=range(2), columns=self.row_head)
                    fields_data.iloc[:, 0] = [f'{item.datetime.date()}_median', f'{item.datetime.date()}_std']
                    fields_data.iloc[:, 1] = [eo.ext(item).cloud_cover, eo.ext(item).cloud_cover]

                    if core:
                        file_path = os.path.join(self.csv_path, f'run_{band}_{core}.csv')
                    else:
                        file_path = os.path.join(self.csv_path, f'run_{band}_all.csv')

                    for ipol in range(len(fields)):
                        polygon = fields[ipol:ipol + 1]
                        try:
                            out_image, out_transform = mask(src, polygon.geometry, crop=True)
                            out_image = np.where(out_image == 0, np.nan, out_image)
                            if np.isnan(out_image).all():
                                fields_data.loc[:, polygon.FS_KENNUNG.values[0]] = [np.nan, np.nan]
                            else:
                                if band == 'SCL':
                                    arr_flat = out_image.flatten()
                                    arr_flat = arr_flat[~np.isnan(arr_flat)]
                                    fields_data.loc[:, polygon.FS_KENNUNG.values[0]] = [int(st.mode(arr_flat)[0][0]),
                                                                                    np.nanstd(out_image)]
                                else:
                                    fields_data.loc[:, polygon.FS_KENNUNG.values[0]] = [np.nanmedian(out_image),
                                                                                    np.nanstd(out_image)]
                        except:
                            fields_data.loc[:, polygon.FS_KENNUNG.values[0]] = [np.nan, np.nan]
                    with open(file_path, 'a') as f1:
                        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                        writer.writerow(list(fields_data.iloc[0, :]))
                        writer.writerow(list(fields_data.iloc[1, :]))
                    src.close()
                    gc.collect()
            
            return None
        except Exception as e:
            return e

    def find_s2_items(self):
        """
        :return: Finds all available S2 L2A images available for the considered timespan and region and returns them as list
        """
        # Define bounding box around fields where S2 data will be loaded from
        xmin, xmax = self.fields.bounds.min(0).minx, self.fields.bounds.max(0).maxx
        ymin, ymax = self.fields.bounds.min(0).miny, self.fields.bounds.max(0).maxy
        area_of_interest = {
            "type": "Polygon",
            "coordinates": [[
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin],
            ]],
        }

        time_of_interest = self.time_of_interest

        # Setup connections to planetary computer and search for available S-2 images
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=area_of_interest,
            datetime=time_of_interest,
            query={"eo:cloud_cover": {"lt": 80}},  # Only scenes with cloud_cover<80% are considered
        )
        items = search.item_collection()
        return items

    def createcsv(self, n_cores=1):
        """
        :param n_cores: int number of cores on which calculation is distributed
        :return: prepares the csv files required for self.file2tab
        """
        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)

        for collection in self.__class__.bands:
            for i in range(n_cores):
                if n_cores==1:
                    file_path = os.path.join(self.csv_path, f'run_{collection}_all.csv')
                else:
                    file_path = os.path.join(self.csv_path, f'run_{collection}_{i}.csv')
                
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f1:
                        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                        writer.writerow(self.row_head)

    def find_existing(self, search_s2, core=None):
        """
        Function needed in case the file2tab crashed during process. It searches for the s2 files
        that have not yet been processed.
        :param search_s2: list of s2 tiles from where to extract the data
        :param core: int for on which CPU core the processing is done
        :return: two updated lists of search_lst and search_geo including all files that do not have results yet.
        """
        if core is None:
            done_files_path = os.path.join(self.csv_path, 'run_SCL_all.csv')
            done_files = pd.read_csv(done_files_path, index_col=0)
            done_dates = [a.split('_')[0].replace('-','') for a in done_files.index]
        else:
            fnames =  glob.glob(os.path.join(self.csv_path, f'run_SCL_*.csv'))
            done_dates = []
            for done_files_path in fnames:
                done_files = pd.read_csv(done_files_path, index_col=0)
                done_dates.extend([a.split('_')[0].replace('-','') for a in done_files.index])
            done_dates = list(set(done_dates))

        search_s2_new = []
        for i in range(len(search_s2)):
            item_i = str(search_s2[i])
            date_time_a = item_i.split('_')[2][:8]
            if (done_files is None) or (not date_time_a in done_dates):
                search_s2_new.append(search_s2[i])
        return search_s2_new

    def run_extraction(self, n_cores=4):
        """
        :param n_cores: int number of cores on which parallel computation should be done
        :param new: boolean. True if the calculation starts. False, if calculation was interrupted and needs to be continued
        :return: runs the function self.extract_s2 in parallel.
        """
        status = "restart"
        while status == "restart":
            status = self.do_run_extraction(n_cores)
            time.sleep(10)

    def do_run_extraction(self, n_cores):
        # create csv file if not exists
        self.createcsv(n_cores=n_cores)

        search_s2 = self.find_s2_items()
        print(len(search_s2))
        search_s2 = self.find_existing(search_s2, n_cores)

        results = []
        if n_cores>1:
            futures = []
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
                for i in range(n_cores):
                    chunks = int(len(search_s2) / n_cores)
                    start, end = i * chunks, (i + 1) * chunks
                    if end > len(search_s2):
                        end = len(search_s2)
                    this_search_s2 = search_s2[start:end]
                    f = pool.submit(self.extract_s2, items=this_search_s2, core=i)
                    futures.append(f)
            #import pdb; pdb.set_trace()
            for f in futures:
                results.append(f.result())
        else:
            # Non-parallel computing use the following
            result = self.extract_s2(items=search_s2, core=None)
            results.append(result)
        
        # analyse results
        status = "succeed"
        for result in results:
            #import pdb; pdb.set_trace()
            if result is None: 
                continue
            elif (isinstance(result, rasterio.errors.RasterioIOError)
                    and str(result)=="HTTP response code: 403"):
                    # planetary computer access error 403
                    status = "restart"
                    break
            else: # other errors which should not occur
                status = "error"
                print ("Unknown Error:", str(reulst))

        # status: succeed, restart , or error (unknown error)
        return status

    def merge_files(self):
        """
        This function merges all tables that were established with self.run to one single table. Is only needed if
        s2 extraction is run in parallel
        :return: writes csv file
        """
        for collection in self.__class__.bands:
            files = os.path.join(self.csv_path, f"run_{collection}*.csv")
            files = glob.glob(files)
            df = pd.concat(map(pd.read_csv, files), ignore_index=True)
            df.to_csv(os.path.join(self.csv_path, f"run_{collection}_all.csv"), index=0)

    def clean_csv(self):
        """
        :return: removes nan rows from resulting csv files from output run_extraction or merge files. The updated files
        are stored as feather to make upcoming processing faster.
        """
        path = rf'{self.table_path}/{self.region}/{self.crop}'
        years = os.listdir(path)
        for year in years:
            path_new = os.path.join(path, year)
            bands = [a for a in os.listdir(path_new) if a.endswith('.csv')]
            for band in bands:
                file = pd.read_csv(os.path.join(path_new, band), index_col=0)
                file_new = file.dropna(axis=0, how='all', subset=file.columns[1:]).reset_index()
                file_new.to_feather(os.path.join(path_new, band.replace('.csv', '.feather')))

    def field2nuts(self):
        """
        :return: aggregates field level data to NUTS4 for Austria
        """
        #Prepare scene classification mask to only use values from fields without disturbances -> i.e. SCL classes 4,5,6
        scl_path = os.path.join(self.csv_path, f'run_SCL_all.feather')
        scl = pd.read_feather(scl_path)
        sub_scl = scl.iloc[:,2:].fillna(0).astype('int')
        cloud_mask = sub_scl.where(sub_scl >= 4)
        cloud_mask = cloud_mask.where(cloud_mask <= 6).notna()
        cloud_mask.index = scl.observation

        all_nuts = np.unique(self.fields.g_id)
        bands = [a for a in self.__class__.bands if not a=='SCL']
        for band in bands:
            file_path = os.path.join(self.csv_path, f'run_{band}_all.feather')
            file = pd.read_feather(file_path)
            file.iloc[:, 2:] = file.iloc[:, 2:][cloud_mask]
            nuts_file = file.iloc[:, :2]
            nuts_file = nuts_file.reindex(columns=['observation', 'cloud_cover[%]']+list(all_nuts))

            for nut in all_nuts:
                fields_in_nuts = [int(a) for a in self.fields.FS_KENNUNG[self.fields.g_id == nut].values]
                fields_in_nuts = [a for a in fields_in_nuts if a in list(file.columns)]
                sub_file = file.loc[:,fields_in_nuts]
                nuts_file.loc[:,nut] = sub_file.mean(axis=1)
            nuts_file.to_feather(os.path.join(self.csv_path, f'run_{band}_all_nuts.feather'))

    def table2nc(self):
        """
        :return:combines all csv files with the data of the individual S-2 L2A Bands to one netcdf file
        """
        #Load first file to get all fields names
        band = self.__class__.bands[0]
        #ToDo: link below can be adjusted f'run_{band}_all_nuts.feather' for converstion of nuts level data
        # or f'run_{band}_all.feather' for field level conversion
        file_path = os.path.join(self.csv_path, f'run_{band}_all_nuts.feather')
        file = pd.read_feather(file_path)
        fields_o = file.columns[2:]

        #Filter fields with only nan values
        nan_fields = file.iloc[:,2:].isna().all(axis=0)
        fields = fields_o[~nan_fields]
        #
        nc_path = os.path.join(self.csv_path, 'nc')
        if not os.path.exists(nc_path):
            os.makedirs(nc_path)

        if np.sum(nan_fields)>0:
            with open(os.path.join(nc_path, 'read_me.txt'), 'w') as file:
                file.write(f'There are fields with no pixels of S-2 L2A data inside. These are the fields:{fields_o[nan_fields].values}')

        #Loop through all fields to establish one nc field per field with all bands
        for f, field in enumerate(fields):
            #loop through all bands to collect information of all bands per field
            for b,band in enumerate(self.__class__.bands[:-1]):
                file_path = os.path.join(self.csv_path, f'run_{band}_all_nuts.feather')
                file = pd.read_feather(file_path)
                file.iloc[:,2:] = file.iloc[:,2:].round()

                #Split file into median and std
                med_f = file.iloc[::2, :]
                std_f = file.iloc[1::2, :]
                #Remove rows with only nan
                nan_fields = med_f.iloc[:, 2:].isna().all(axis=1)
                med_f = med_f[~nan_fields.values]
                std_f = std_f[~nan_fields.values]

                #check if there are only std values in the std_f file
                char = [a.split('_')[1] for a in std_f.iloc[:,0]]
                if not char[1:]==char[:-1]:
                    raise ValueError('There are not only std values in the std file')

                #Establish a pandas Series as target variable with daily timestep from 2016 to 2022
                dates_all = pd.date_range(start=f'1/1/{self.year}', end=f'31/12/{self.year}')
                target_var = pd.Series(data=None, index=dates_all)

                #Establish pandas series with sentinel-2 data as loaded in the file and merge it with the target series
                dates = [datetime.strptime(a.split('_')[0], '%Y-%m-%d') for a in med_f.iloc[:,0]]
                med_field = pd.Series(data=med_f.iloc[:, f+2].values, index=dates)
                std_field = pd.Series(data=std_f.iloc[:, f+2].values, index=dates)

                #resample to daily values for cases where there are several observations per day
                med_field = med_field.resample('D').mean()
                std_field = std_field.resample('D').mean()

                _, med_daily = target_var.align(med_field, axis=0)

                _, std_daily = target_var.align(std_field, axis=0)
                med_daily = med_daily.replace(np.nan, -9999)
                std_daily = std_daily.replace(np.nan, -9999)

                #Establish new xr Dataset in first loop. Afterwards add bands to this file.
                if b == 0:
                    #Add cloud cover info
                    cloud_field = pd.Series(data=med_f.iloc[:,1].values, index=dates)
                    cloud_field = cloud_field.resample('D').mean()
                    _, cloud_daily = target_var.align(cloud_field, axis=0)
                    xr_file = xr.Dataset.from_dataframe(pd.DataFrame(cloud_daily, columns=[f'cloud_cover']))
                    xr_file[f'cloud_cover'].attrs = dict(
                                            # FillValue=np.nan,
                                            units='%',
                                            long_name=f'mean cloud cover over whole S-2 scene, not per field',
                                            value_range='0-100'
                                        )

                    xr_file[f'{band}_median'] = xr.DataArray.from_series(med_daily).astype('int32')
                    xr_file[f'{band}_std'] = xr.DataArray.from_series(std_daily).astype('int32')

                else:
                    if band == 'SCL':
                        xr_file[f'{band}_mode'] = xr.DataArray.from_series(med_daily).astype('int32')
                    else:
                        xr_file[f'{band}_median'] = xr.DataArray.from_series(med_daily).astype('int32')
                        xr_file[f'{band}_std'] = xr.DataArray.from_series(std_daily).astype('int32')
                if band == 'SCL':
                    xr_file['SCL_mode'].attrs = dict(
                        FillValue=-9999,
                        units='-',
                        long_name='Sentinel-2 Scene Classification',
                        scl_values=[0,1,2,3,4,5,6,7,8,9,10,11],
                        scl_classes=['No_data','Saturated_or_defective_pixel','Topographic_casted_shadows','Cloud_shadows',
                                 'Vegetation','Non-vegetated','Water','Unclassified','Cloud_medium_probability',
                                 'Cloud_high_probability','Thin_cirrus','Snow_or_ice']
                    )
                else:
                    xr_file[f'{band}_median'].attrs = dict(
                        FillValue=-9999,
                        units='-',
                        long_name=f'median of all Sentinel-2 {band} pixels laying within field',
                        value_range='1-10000'
                    )
                    xr_file[f'{band}_std'].attrs = dict(
                        FillValue=-9999,
                        units='-',
                        long_name=f'std of all Sentinel-2 {band} pixels laying within field',
                        value_range='1-10000'
                    )
            xr_file = xr_file.rename({'index':'time'})
            xr_file.to_netcdf(path=os.path.join(nc_path, f'{field}.nc'))

    # The nc files are cleaned (outlier removed and cloud masked) with the following function
    def cleaning_s2(self, cloud_flag=False):
        """
        :param cloud_flag: boolean, true if data still needs to be cloud flagged. False if thats already done. The
        latter is the case if the fields are already aggregated to nuts level
        :return: Removes outliers and saves new nc file
        """
        path = os.path.join(self.csv_path, 'nc')
        fields = os.listdir(path)
        fields = [field for field in fields if field.endswith('.nc')]

        bands = self.__class__.bands[:-1]
        bands = [band+'_median' for band in bands]

        #Load required data
        for field in fields:
            #Load file and extract scene classification
            ds = xr.open_dataset(os.path.join(path, field))

            if cloud_flag:
                scl = ds.SCL_mode.values
                if np.sum(scl==[-9999]*len(scl))==len(scl):
                    print(f'file {field} has only nan values')
                    continue

                # Establish mask with valid observations. I.e. Scene classification without clouds, snow etc.
                # scl as in https://www.sciencedirect.com/science/article/pii/S0924271623002654
                ok_scene_class = [4, 5, 6]
                a_mask = np.in1d(scl, ok_scene_class)

            # Sentinel-2 processing changed on 2022-01-25 introducing an offset of 1000
            # https://github.com/stactools-packages/sentinel2/issues/44

            time = ds.time.values
            offset_loc = np.where(time>=pd.to_datetime('2022-01-25', format='%Y-%m-%d'))[0]

            #Load all bands, mask them with scl mask and remove outliers
            for band in bands:
                band_values = ds[band].values
                #remove offset of 1000 since Jan 25 2022
                band_values[offset_loc] = band_values[offset_loc]-1000
                if cloud_flag:
                    band_values_masked = band_values[a_mask]
                    #remove outliers
                    band_values_masked_no_ol = removeOutliers(band_values_masked)
                    ds[band].values = [-9999] * len(ds[band].values)
                    ds[band].values[a_mask] = band_values_masked_no_ol
                else:
                    band_values_masked_no_ol = removeOutliers(band_values)
                    ds[band].values = [-9999] * len(ds[band].values)
                    ds[band].values = band_values_masked_no_ol

            path_out = os.path.join(path, 'cleaned')
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            ds.to_netcdf(os.path.join(path_out, field))

    # As a last step, Vegetation indices are added to the nc files.
    def add_indices2nc(self):
        """
        :return: writes new nc file with different vegetation indices.
        """
        file_path = os.path.join(self.csv_path, 'nc', 'cleaned')
        files = os.listdir((file_path))
        files = [file for file in files if file.endswith('.nc')]
        for file in files:
            ds = xr.open_dataset(os.path.join(file_path, file))

            B2 = ds.B02_median
            B4 = ds.B04_median
            B8 = ds.B08_median
            B11 = ds.B11_median
            B12 = ds.B12_median
            ndvi,evi,ndwi,nmdi = indices_calc(B2, B4, B8, B11, B12)

            ds['ndvi'] = ndvi
            ds['ndvi'].attrs = dict(
                # FillValue=np.nan,
                units='-',
                long_name='Normalized Difference Vegetation Index',
                value_range='-1 to 1'
            )

            ds['evi'] = evi
            ds['evi'].attrs = dict(
                # FillValue=np.nan,
                units='-',
                long_name='Enhanced Vegetation Index',
                value_range='-1 to 1'
            )

            ds['ndwi'] = ndwi
            ds['ndwi'].attrs = dict(
                # FillValue=np.nan,
                units='-',
                long_name='Normalized Difference Water Index',
                value_range='-1 to 1'
            )

            ds['nmdi'] = nmdi
            ds['nmdi'].attrs = dict(
                # FillValue=np.nan,
                units='-',
                long_name='Normalized Multiband Drought Index',
                value_range='0 to ~1'
            )

            if not os.path.exists(os.path.join(file_path, 'new')):
                os.makedirs(os.path.join(file_path, 'new'))
            ds.to_netcdf(path=os.path.join(file_path, 'new', file))


# The next two functions are used before in add_indeces2nc and cleaning_s2
def indices_calc(B2,B4,B8,B11,B12):
    """
    :param B2-B12: np.array of the required S-2 bands to calculate the indices.
    :return: Calculates the indices ndvi, evi, ndwi (water index), and nmdi (normalized multiband drought index)
    based on common formulas as for example in cavalaris et al., 2021
    """
    ts = B2.time.values

    B2 = np.where(B2<=-9999, np.nan, B2/10000)
    B4 = np.where(B4<=-9999, np.nan, B4/10000)
    B8 = np.where(B8<=-9999, np.nan, B8/10000)
    B11 = np.where(B11<=-9999, np.nan, B11/10000)
    B12 = np.where(B12<=-9999, np.nan, B12/10000)

    ndvi = (B8-B4)/(B8+B4)
    evi = 2.5*(B8-B4)/((B8+6*B4-7.5*B2)+1)
    ndwi = (B8-B12)/(B8+B12)
    nmdi = (B8-(B11-B12))/(B8+(B11-B12))

    ndvi_ar = xr.DataArray.from_series(pd.Series(data=ndvi, index=ts))
    evi_ar = xr.DataArray.from_series(pd.Series(data=evi, index=ts))
    ndwi_ar = xr.DataArray.from_series(pd.Series(data=ndwi, index=ts))
    nmdi_ar = xr.DataArray.from_series(pd.Series(data=nmdi, index=ts))

    ndvi_ar = ndvi_ar.rename({'index': 'time'})
    evi_ar = evi_ar.rename({'index': 'time'})
    ndwi_ar = ndwi_ar.rename({'index': 'time'})
    nmdi_ar = nmdi_ar.rename({'index': 'time'})

    return ndvi_ar,evi_ar,ndwi_ar,nmdi_ar

def removeOutliers(x, outlierConstant=2):
    a = np.array(x)
    a_masked = a[np.where(a>-9999)]
    upper_quartile = np.percentile(a_masked, 75)
    lower_quartile = np.percentile(a_masked, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    a = np.where(((a>=quartileSet[0]) & (a<=quartileSet[1])),a,-9999)
    return a


def main():

    proc_step = sys.argv[1] # either download or postproc
    region = sys.argv[2]
    n_cores = int(sys.argv[3])

    data_dir = "/yipeeo/data/cloud-geo/Crop_data/Crop_class"
    out_dir = "/yipeeo/work/field_extraction"

    # collection all crop type of each year for given region
    crop_data = {}
    for fname in glob.glob(f"{data_dir}/{region}/nuts/*.shp"):
        bname = os.path.splitext(os.path.basename(fname))[0]
        crop_type, crop_year = bname[:-5], int(bname[-4:])
        if crop_type not in crop_data:
            crop_data[crop_type] = []
        crop_data[crop_type].append(crop_year)


    #ToDo: this code needs to be run separately for the three crops winter wheat, maize and spring barley for the countries CZR and Austria
    warnings.filterwarnings('ignore')

    for crop_type in sorted(crop_data.keys()):
        years = sorted(crop_data[crop_type])
        for year in years:
            try:
                
                print (f"{region}: crop={crop_type} year={year} started")
                start_pro = datetime.now()
                a = s2(region=region, year=year, crop=crop_type, 
                        data_dir=data_dir, out_dir=out_dir)
                
                if proc_step=="download":
                    if region=="Austria" and crop_type=="maize" and year==2016:
                        print (f"Skip: Austria {crop_type} {year}")
                        continue

                    print (f"{region}: crop={crop_type} year={year} start run_extraction")        
                    a.run_extraction(n_cores=n_cores)
                    print (f"{region}: crop={crop_type} year={year} start merge_files")
                    a.merge_files()
                    print (f"{region}: crop={crop_type} year={year} start clean_csv")
                    a.clean_csv()

                # ToDo field2nuts saves much time, as the data is directly merged to nuts before conversion to nc.
                #  Field data would of course by prefered, but if not feasible, aggregation can be done with the following:
                import pdb; pdb.set_trace()
                if proc_step=="postproc":
                    print (f"{region}: crop={crop_type} year={year} start field2nuts")
                    a.field2nuts()

                    print (f"{region}: crop={crop_type} year={year} start table2nc")
                    a.table2nc()
                    a.cleaning_s2()
                    a.add_indices2nc()

                print(f'{region}: crop={crop_type} year={year} completed in{datetime.now() - start_pro}')
            except Exception as e:
                print ("Exception:", e)



if __name__ == '__main__':
    main()