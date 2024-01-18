import csv
import datetime
import gc
import warnings
import os
import glob
import rasterio
import h5py
import earthaccess
import pyproj
import itertools
import rich.table
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pystac.extensions.eo import EOExtension as eo
from scipy import stats as st
from pystac_client import Client
from pyresample import geometry as geom
from pyresample import kd_tree as kdt
from rasterio.mask import mask
from rasterio.transform import Affine
# from osgeo import gdal, gdal_array, gdalconst, osr


class ecostress:
    """
    Class to retrieve ecostress data and extract it to field level
    """
    collections = ['ECO2CLD', 'ECO2LSTE']
    def __init__(self, region):
        self.region = region
        self.path_temp = 'D:/DATA/yipeeo/Predictors/ECOSTRESS/temp_files2'
        self.table_path = 'D:/data-write/YIPEEO/predictors/ECOSTRESS/ts'
        self.fields = gpd.read_file(f'D:/DATA/yipeeo/Crop_data/Crop_yield/all/field_scale_{region}.shp')
        self.fields = self.fields.drop_duplicates(subset='field_id')
        self.row_head = ['observation', 'day_time'] + list(self.fields.field_id.values)

        self.year_min, self.year_max = np.nanmin(self.fields.c_year.values), np.nanmax(self.fields.c_year.values)
        if self.year_max < 2018:
            raise ValueError('no ECOSTRESS data available for the given fields at the time of harvesting')
        if self.year_min < 2019:
            self.year_min = 2019
            warnings.warn(
                'There are harvest dates before 2018. No ECOSTRESS data is available by then. Hence, only ECOSTRESS data from 2018 onwards will be extracted')

        # Define bounding box around fields where ECOSTRESS data will be loaded from
        xmin, xmax = self.fields.bounds.min(0).minx, self.fields.bounds.max(0).maxx
        ymin, ymax = self.fields.bounds.min(0).miny, self.fields.bounds.max(0).maxy
        self.bbox = (xmin, ymin, xmax, ymax)

    def createcsv(self, n_cores=1):
        """
        :param n_cores: int number of cores on which calculation is distributed
        :return: prepares the csv files required for self.file2tab
        """
        if not os.path.exists(rf'{self.table_path}/{self.region}'):
            os.makedirs(rf'{self.table_path}/{self.region}')

        for collection in self.__class__.collections:
            for i in range(n_cores):
                if n_cores==1:
                    file_path = rf'{self.table_path}/{self.region}/run_{collection}_all.csv'
                else:
                    file_path = rf'{self.table_path}/{self.region}/run_{collection}_{i}.csv'
                with open(file_path, 'w') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow(self.row_head)

    def findFiles(self, collection):
        """
        This function logs in to earthaccess and searches the available ecostress files available for the given
        time range and locations
        :param collection: str of ECOSTRESS data that should be searched for: tested vars are 'ECO2CLD' and 'ECO2LSTE'
        :return: two lists containing the available files of the specified collection and the file containing the lat-
        lon information of the ECOSTRESS swaths
        """
        # Setup connections to Earthdata server and search for available ECOSTRESS images
        earthaccess.login()
        search_lst = earthaccess.search_data(
            short_name=collection,
            bounding_box=self.bbox,
            temporal=(f'{self.year_min}-01-01', f'{self.year_max}-12-31'),
            cloud_hosted=True,
        )

        search_geo = earthaccess.search_data(
            short_name='ECO1BGEO',
            bounding_box=self.bbox,
            temporal=(f'{self.year_min}-01-01', f'{self.year_max}-12-31'),
            cloud_hosted=True,
        )

        # Check if geo files with lat lon information are from the same date and time as LST data
        # and remove if they do not correspond
        for i in range(len(search_lst)):
            date_time_a = search_lst[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            date_time_b = search_geo[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            if not date_time_a == date_time_b:
                search_geo.pop(i)
        # Check if it worked and all correspond. Remove files where this is not the case
        a = []
        for i in range(len(search_lst)):
            date_time_a = search_lst[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            date_time_b = search_geo[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            a.append(date_time_a == date_time_b)
        file_false = [i for i, x in enumerate(a) if not x]
        file_false.reverse()
        for ff in file_false:
            search_geo.pop(ff)
            search_lst.pop(ff)
        print(len(search_lst), len(search_geo))

        return search_lst, search_geo

    def file2tab(self, search_lst, search_geo, collection, core):
        """
        This function downloads the tiles selected in self.findFiles and extracts the data for all specified fields
        :param search_lst: list of ECOSTRESS tiles from where to extract the data
        :param search_geo: list of the corresponding lat/lon files from ECOSTRESS
        :param collection: str of ECOSTRESS data that should be searched for: tested vars are 'ECO2CLD' and 'ECO2LSTE'
        :param core: int for on which CPU core the processing is done
        :return: writes to ECOSTRESS data per field to csv-file established in self.createcsv
        """
        for i_files in range(len(search_lst)):
        # for i_files in range(5):
            date_time = search_lst[i_files:i_files + 1][0]['umm']['TemporalExtent']['RangeDateTime'][
                'BeginningDateTime']
            print('\n', i_files, date_time)
            date, time = date_time.split('T')[0], date_time.split('T')[1][:5]

            file_lst = earthaccess.open(search_lst[i_files:i_files + 1], verbose=0)
            file_geo = earthaccess.open(search_geo[i_files:i_files + 1])

            sf, add_off, file_name = eco2tif(path_lst=file_lst[0], path_geo=file_geo[0], path_out=self.path_temp, core=core)

            src = rasterio.open(file_name)
            self.fields = self.fields.to_crs(src.crs)

            # Per scene and band extract and summarize the information per field
            fields_data = pd.DataFrame(data=None, index=range(2), columns=self.row_head)

            fields_data.iloc[:, 0] = [f'{date}_median', f'{date}_std']
            fields_data.iloc[:, 1] = [time, time]

            file_path = rf'{self.table_path}/{self.region}/run_{collection}_{core}.csv'

            for ipol in range(len(self.fields)):
                polygon = self.fields[ipol:ipol + 1]
                try:
                    out_image, out_transform = mask(src, polygon.geometry, crop=True)
                    out_image = np.where(out_image == 0, np.nan, out_image)
                    if np.isnan(out_image).all():
                        fields_data.loc[:, polygon.field_id.values[0]] = [np.nan, np.nan]
                    else:
                        if collection == 'ECO2CLD':
                            arr_flat = out_image.flatten()
                            arr_flat = arr_flat[~np.isnan(arr_flat)]
                            fields_data.loc[:, polygon.field_id.values[0]] = [int(st.mode(arr_flat)[0][0]),
                                                                              np.nanstd(out_image)]
                        else:
                            fields_data.loc[:, polygon.field_id.values[0]] = [np.nanmedian(out_image) * sf + add_off,
                                                                              np.nanstd(out_image) * sf + add_off]
                except:
                    fields_data.loc[:, polygon.field_id.values[0]] = [np.nan, np.nan]
                    # warnings.warn('Input shapes do not overlap raster')

            with open(file_path, 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(list(fields_data.iloc[0, :]))
                writer.writerow(list(fields_data.iloc[1, :]))
            src.close()
            os.remove(file_name)
            gc.collect()

    def find_existing(self, search_lst, search_geo, collection, core):
        """
        Function needed in case the file2tab crashed during process. It searches for the ECOSTRESS files
        that have not yet been processed. Same parameters required as for self.file2tab
        :param search_lst: list of ECOSTRESS tiles from where to extract the data
        :param search_geo: list of the corresponding lat/lon files from ECOSTRESS
        :param collection: str of ECOSTRESS data that should be searched for: tested vars are 'ECO2CLD' and 'ECO2LSTE'
        :param core: int for on which CPU core the processing is done
        :return: two updated lists of search_lst and search_geo including all files that do not have results yet.
        """
        done_files_path = rf'{self.table_path}/{self.region}/run_{collection}_{core}.csv'
        if core=='all':
            done_files_path = rf'{self.table_path}/{self.region}/{collection}_all_run.csv'
        done_files = pd.read_csv(done_files_path, index_col=0)
        done_dates = [a.split('_')[0] for a in done_files.index]
        search_lst_new = []
        search_geo_new = []
        for i in range(len(search_lst)):
            date_time_a = search_lst[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            date_time_a = str(date_time_a[:10])
            if not date_time_a in done_dates:
                search_geo_new.append(search_geo[i:i + 1])
                search_lst_new.append(search_lst[i:i + 1])
        search_geo_new = list(itertools.chain(*search_geo_new))
        search_lst_new = list(itertools.chain(*search_lst_new))
        return search_lst_new, search_geo_new

    def clean_temp(self):
        """
        Needs to be run if file2tab crushed during processing and not all files in the temporary folders were deleted
        :return: deletes all files in temp folder
        """
        tempFiles = os.listdir(self.path_temp)
        if len(tempFiles)>0:
            for tempFile in tempFiles:
                os.remove(os.path.join(self.path_temp, tempFile))

    def run(self, n_cores=1, new=False):
        """
        This functions runs the code required to download ECOSTRESS tiles and extract it to field-level
        :param n_cores: int number of CPU cores to which to parallelize the processing
        :param new: Boolean leave false if there are already some results stored. True if its the first run
        :return: ECOSTRESS data per field stored in csv files
        """
        start_time = datetime.datetime.now()
        self.clean_temp()
        if new:
            self.createcsv(n_cores=n_cores)

        for collection in self.__class__.collections:
            search_lst, search_geo = self.findFiles(collection)

            if n_cores > 1:
                with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
                    for i in range(n_cores):
                        chunks = int(len(search_lst) / n_cores)
                        start, end = i * chunks, (i + 1) * chunks
                        if end > len(search_lst):
                            end = len(search_lst)
                        if not new:
                            search_lst_new, search_geo_new = self.find_existing(search_lst=search_lst[start:end], search_geo=search_geo[start:end], collection=collection, core=i)
                        else:
                            search_lst_new, search_geo_new = search_lst[start:end], search_geo[start:end]

                        pool.submit(self.file2tab, search_lst=search_lst_new, search_geo=search_geo_new, collection=collection, core=i)
                        # self.file2tab(search_lst=search_lst_new, search_geo=search_geo_new, collection=collection, core=i)
            else:
                self.file2tab(search_lst, search_geo, collection)
        print(datetime.datetime.now() - start_time)

    def run_leftovers(self, n_cores=1):
        """
        :param n_cores: int number of CPU cores on which to parallelize the computation
        :return: does the same as self.run but first checks if there are any tiles that have not yet been processed.
        Self.merge_files must be run before that
        """
        start_time = datetime.datetime.now()
        self.clean_temp()
        for collection in self.__class__.collections:
            search_lst, search_geo = self.findFiles(collection)
            search_lst_new, search_geo_new = self.find_existing(search_lst=search_lst, search_geo=search_geo,
                                                                collection=collection, core='all')
            print(len(search_lst_new), len(search_geo_new))
            print(search_lst_new[:1], search_geo_new[:1])
            print(search_lst_new[-1:], search_geo_new[-1:])
            if n_cores > 1:
                with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
                    for i in range(n_cores):
                        chunks = int(len(search_lst) / n_cores)
                        start, end = i * chunks, (i + 1) * chunks
                        if end > len(search_lst):
                            end = len(search_lst)

                        pool.submit(self.file2tab, search_lst=search_lst_new[start:end], search_geo=search_geo_new[start:end], collection=collection, core=i)
                        # self.file2tab(search_lst=search_lst_new, search_geo=search_geo_new, collection=collection, core=i)
            else:
                self.file2tab(search_lst, search_geo, collection)
        print(datetime.datetime.now() - start_time)

    def merge_files(self):
        """
        This function merges all tables that were established with self.run to one single table
        :return: writes csv file
        """
        path = os.path.join(self.table_path, self.region)
        for collection in self.__class__.collections:
            files = os.path.join(path, f"run_{collection}*.csv")
            files = glob.glob(files)
            df = pd.concat(map(pd.read_csv, files), ignore_index=True)
            df.to_csv(os.path.join(path, f"{collection}_all_run.csv"), index=0)

    def table2nc(self):
        """
        :return:combines all csv files with the data of the ECOSTRESS data collection to one netcdf file.
        Needs the results of self.merge_files
        """

        # define all sentinal-2 L2A bands that should be considered
        bands = self.__class__.collections
        band = bands[0]

        # Load first file to get all fields names
        file_path = os.path.join(self.table_path, self.region, f"{band}_all_run.csv")
        file = pd.read_csv(file_path, index_col=0)
        fields = file.columns[1:]

        # Filter fields with only nan values
        nan_fields = file.iloc[:, 1:].isna().all(axis=0)
        fields = fields[~nan_fields]

        if not os.path.exists(os.path.join(self.table_path, self.region,'nc')):
            os.makedirs(os.path.join(self.table_path, self.region,'nc'))

        if np.sum(nan_fields) > 0:
            with open(rf'{self.table_path}\{self.region}\nc\read_me.txt', 'w') as file:
                file.write(
                    f'There are fields with no pixels of S-2 L2A data inside. These are the fields:{fields[nan_fields].values}')

        # Loop through all fields to establish one nc field per field with all bands
        for f, field in enumerate(fields[:1]):
            # loop through all bands to collect information of all bands per field
            for b, band in enumerate(bands):
                file_path = os.path.join(self.table_path, self.region, f"{band}_all_run.csv")
                file = pd.read_csv(file_path, index_col=0)
                file.iloc[:, 1:] = file.iloc[:, 1:].round()

                # Split file into median and std
                med_f = file.iloc[::2, :]
                std_f = file.iloc[1::2, :]
                # Remove rows with only nan
                nan_fields = med_f.iloc[:, 1:].isna().all(axis=1)
                med_f = med_f[~nan_fields.values]
                std_f = std_f[~nan_fields.values]

                # check if there are only std values in the std_f file
                char = [a.split('_')[1] for a in std_f.index]
                if not char[1:] == char[:-1]:
                    raise ValueError('There are not only std values in the std file')

                # Establish a pandas Series as target variable with daily timestep from 2019 to 2022
                dates_all = pd.date_range(start='1/1/2019', end='31/12/2022')
                target_var = pd.Series(data=None, index=dates_all)

                # Establish pandas series with sentinel-2 data as loaded in the file and merge it with the target series
                dates = [datetime.datetime.strptime(a.split('_')[0], '%Y-%m-%d') for a in med_f.index]
                med_field = pd.Series(data=med_f.iloc[:, f + 2].values, index=dates)
                std_field = pd.Series(data=std_f.iloc[:, f + 2].values, index=dates)

                # resample to daily values for cases where there are several observations per day
                med_field = med_field.resample('D').mean()
                std_field = std_field.resample('D').mean()

                _, med_daily = target_var.align(med_field, axis=0)
                _, std_daily = target_var.align(std_field, axis=0)
                med_daily = med_daily.replace(np.nan, -9999)
                std_daily = std_daily.replace(np.nan, -9999)

                # Establish new xr Dataset in first loop. Afterwards add bands to this file.
                if band=='ECO2CLD':
                    xr_file = xr.Dataset.from_dataframe(pd.DataFrame(med_daily, columns=[f'{band}_mod']))
                    xr_file[f'{band}_mod'].attrs = dict(
                        # FillValue=np.nan,
                        units='cloud flags',
                        long_name=f'ECOSTRESS mode of pixel flags within field',
                        description='binary flags for 2**n: 0: cloud mask flag, 1: Final cloud plus region-growing, 2: Final Cloud, either one of bits 2, 3 ,or 4 set, 3:band 4 Brightness Threshold Test, 4: Band 4-5 Thermal Difference test, 5: Land/Water mask'
                    )
                else:
                    xr_file[f'{band}_median'] = xr.DataArray.from_series(med_daily)
                    xr_file[f'{band}_std'] = xr.DataArray.from_series(std_daily)
                    xr_file[f'{band}_median'].attrs = dict(
                        FillValue=-9999,
                        units='K',
                        long_name=f'median land surface temperature of all ECOSTRESS pixels laying within field',
                    )
                    xr_file[f'{band}_std'].attrs = dict(
                        FillValue=-9999,
                        units='K',
                        long_name=f'std land surface temperature of all ECOSTRESS pixels laying within field',
                    )

            xr_file = xr_file.rename({'index': 'time'})
            # print(xr_file.time)
            # comp = dict(zlib=True, complevel=9)
            # encoding = {var: comp for var in xr_file.data_vars}
            xr_file.to_netcdf(os.path.join(self.table_path, self.region,'nc', f'{field}.nc'))


def eco2tif(path_lst, path_geo, path_out, uncertainty=False, core=1):
    """
    :param path_lst: str of location where to load ecostress lst data from
    :param path_geo: str of location where to load ecostress grid data from
    :param uncertainty: boolean if lst uncertainties should be written to tif file too
    :return: converts ecostress data to WGS84 and writes it to tiff file
    """
    f = h5py.File(path_lst)['SDS']
    eco_objs = []
    f.visit(eco_objs.append)
    ecoSDS = [str(obj) for obj in eco_objs if isinstance(f[obj], h5py.Dataset)]

    sds = ['LST','LST_err','CloudMask']
    ecoSDS = [dataset for dataset in ecoSDS if dataset.endswith(tuple(sds))]

    # Open Geo File
    g = h5py.File(path_geo)
    geo_objs = []
    g.visit(geo_objs.append)

    # Search for lat/lon SDS inside data file
    latSD = [str(obj) for obj in geo_objs if isinstance(g[obj], h5py.Dataset) and '/latitude' in obj]
    lonSD = [str(obj) for obj in geo_objs if isinstance(g[obj], h5py.Dataset) and '/longitude' in obj]

    # Open SDS as arrays
    lat = g[latSD[0]][()].astype(float)
    lon = g[lonSD[0]][()].astype(float)

    # Read the array dimensions
    dims = lat.shape

    swathDef = geom.SwathDefinition(lons=lon, lats=lat)

    # Define the lat/lon for the middle of the swath
    mid = [int(lat.shape[1] / 2) - 1, int(lat.shape[0] / 2) - 1]
    midLat, midLon = lat[mid[0]][mid[1]], lon[mid[0]][mid[1]]

    # Define AEQD projection centered at swath center
    epsgConvert = pyproj.Proj("+proj=aeqd +lat_0={} +lon_0={}".format(midLat, midLon))

    # Use info from AEQD projection bbox to calculate output cols/rows/pixel size
    llLon, llLat = epsgConvert(np.min(lon), np.min(lat), inverse=False)
    urLon, urLat = epsgConvert(np.max(lon), np.max(lat), inverse=False)
    areaExtent = (llLon, llLat, urLon, urLat)
    cols = int(round((areaExtent[2] - areaExtent[0]) / 70))  # 70 m pixel size
    rows = int(round((areaExtent[3] - areaExtent[1]) / 70))

    # Define Geographic projection
    epsg, proj, pName = '4326', 'longlat', 'Geographic'

    # Define bounding box of swath
    llLon, llLat, urLon, urLat = np.min(lon), np.min(lat), np.max(lon), np.max(lat)
    areaExtent = (llLon, llLat, urLon, urLat)

    # Create area definition with estimated number of columns and rows
    projDict = pyproj.CRS("epsg:4326")
    areaDef = geom.AreaDefinition(epsg, pName, proj, projDict, cols, rows, areaExtent)

    # Square pixels and calculate output cols/rows
    ps = np.min([areaDef.pixel_size_x, areaDef.pixel_size_y])
    cols = int(round((areaExtent[2] - areaExtent[0]) / ps))
    rows = int(round((areaExtent[3] - areaExtent[1]) / ps))

    # Set up a new Geographic area definition with the refined cols/rows
    areaDef = geom.AreaDefinition(epsg, pName, proj, projDict, cols, rows, areaExtent)

    # Get arrays with information about the nearest neighbor to each grid point
    index, outdex, indexArr, distArr = kdt.get_neighbour_info(swathDef, areaDef, 210, neighbours=1)

    # Read in ETinst and print out SDS attributes
    s = ecoSDS[0]
    ecoSD = f[s][()]

    # Read SDS attributes and define fill value, add offset, and scale factor if available
    try:
        fv = int(f[s].attrs['_FillValue'])
    except KeyError:
        fv = None
    except ValueError:
        fv = f[s].attrs['_FillValue'][0]
    try:
        sf = f[s].attrs['scale_factor'][0]
    except:
        sf = 1
    try:
        add_off = f[s].attrs['add_offset'][0]
    except:
        add_off = 0
    try:
        units = f[s].attrs['units'].decode("utf-8")
    except:
        units = 'none'

    # Perform K-D Tree nearest neighbor resampling (swath 2 grid conversion)
    ETgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, ecoSD, index, outdex, indexArr, fill_value=None)

    # Define the geotransform
    gt = [areaDef.area_extent[0], ps, 0, areaDef.area_extent[3], 0, -ps]
    #Apply scale factor and add offset and fill value
    # ETgeo = ETgeo * sf + add_off            # Apply Scale Factor and Add Offset
    # ETgeo[ETgeo == fv * sf + add_off] = fv  # Set Fill Value

    #Rerun steps above for uncertainty file if needed
    if uncertainty:
        s = ecoSDS[1]
        ecoSD = f[s][()]
        try:
            fv = int(f[s].attrs['_FillValue'])
        except KeyError:
            fv = None
        except ValueError:
            fv = f[s].attrs['_FillValue'][0]
        try:
            sf = f[s].attrs['_Scale'][0]
        except:
            sf = 1
        try:
            add_off = f[s].attrs['_Offset'][0]
        except:
            add_off = 0
        UNgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, ecoSD, index, outdex, indexArr, fill_value=None)
        # UNgeo = UNgeo * sf + add_off
        # UNgeo[UNgeo == fv * sf + add_off] = fv

        # Set up dictionary of arrays to export
        outFiles = {'lst': ETgeo, 'lstUncertainty': UNgeo}
    else:
        outFiles = {'lst': ETgeo}

    for file in outFiles:
        # Set up output name
        outName = rf'{path_out}/{file}_{core}.tif'

        # non gdal raster writing
        ndvi = (outFiles[file].astype(rasterio.int16))
        xres, yres = areaDef.pixel_size_x, areaDef.pixel_size_y
        transform = Affine.translation(llLon - xres / 2, llLat - yres / 2) * Affine.scale(xres, yres)
        #
        with rasterio.open(outName, 'w', driver='GTiff', height=ndvi.shape[0], width=ndvi.shape[1], count=1,
                           dtype=ndvi.dtype, crs='EPSG:4326', transform=transform) as dst:
            dst.write_band(1, np.flip(ndvi.astype(rasterio.int16), axis=0))

    return sf, add_off, outName


if __name__=='__main__':
    warnings.filterwarnings('ignore')
    print(datetime.datetime.now())
    a = ecostress(region='czr')
    a.merge_files()
    # a.run_leftovers(n_cores=2)
    # a.table2nc()
    # a.run(n_cores=20, new=False)
    print('finished at: ',datetime.datetime.now())