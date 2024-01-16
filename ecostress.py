import csv
import datetime
import gc
import warnings
import os
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
        self.path_temp = '/home/pbueechi/thinclient_drives/D:/DATA/yipeeo/Predictors/ECOSTRESS/temp_files2'
        self.table_path = '/home/pbueechi/thinclient_drives/D:/data-write/YIPEEO/predictors/ECOSTRESS/tstest'
        self.fields = gpd.read_file(f'/home/pbueechi/thinclient_drives/D:/DATA/yipeeo/Crop_data/Crop_yield/all/field_scale_{region}.shp')
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
        if not os.path.exists(rf'{self.table_path}\{self.region}'):
            os.makedirs(rf'{self.table_path}\{self.region}')

        for collection in self.__class__.collections:
            for i in range(n_cores):
                if n_cores==1:
                    file_path = rf'{self.table_path}\{self.region}\run_{collection}_all.csv'
                else:
                    file_path = rf'{self.table_path}\{self.region}\run_{collection}_{i}.csv'
                with open(file_path, 'w') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow(self.row_head)

    def findFiles(self, collection):
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
        print(len(search_lst), len(search_geo))
        return search_lst, search_geo

    def file2tab(self, search_lst, search_geo, collection, core):
        for i_files in range(len(search_lst)):
        # for i_files in range(5):
            date_time = search_lst[i_files:i_files + 1][0]['umm']['TemporalExtent']['RangeDateTime'][
                'BeginningDateTime']
            print('\n', i_files, date_time)
            date, time = date_time.split('T')[0], date_time.split('T')[1][:5]

            file_lst = earthaccess.open(search_lst[i_files:i_files + 1])
            file_geo = earthaccess.open(search_geo[i_files:i_files + 1])

            sf, add_off, file_name = eco2tif(path_lst=file_lst[0], path_geo=file_geo[0], path_out=self.path_temp, core=core)

            src = rasterio.open(file_name)
            self.fields = self.fields.to_crs(src.crs)

            # Per scene and band extract and summarize the information per field
            fields_data = pd.DataFrame(data=None, index=range(2), columns=self.row_head)

            fields_data.iloc[:, 0] = [f'{date}_median', f'{date}_std']
            fields_data.iloc[:, 1] = [time, time]

            file_path = rf'{self.table_path}\{self.region}\run_{collection}_{core}.csv'

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

    def run(self, n_cores=1, new=False):
        start_time = datetime.datetime.now()
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

                        pool.submit(self.file2tab, search_lst=search_lst[start:end], search_geo=search_geo[start:end], collection=collection, core=i)
            else:
                self.file2tab(search_lst, search_geo, collection)
        print(datetime.datetime.now() - start_time)


def extract_eco(region, new=True):
    print('started calculating...')
    #Define datasets that should be loaded: ECOSTRESS L1 Radiation and L2 LSTE and clouds
    collections = ['ECO2CLD', 'ECO2LSTE']
    path_temp = '/home/pbueechi/thinclient_drives/D:/DATA/yipeeo/Predictors/ECOSTRESS/temp_files2'
    table_path = '/home/pbueechi/thinclient_drives/D:/data-write/YIPEEO/predictors/ECOSTRESS/ts'

    #Load fields and remove identical fields
    fields = gpd.read_file(f'/home/pbueechi/thinclient_drives/D:/DATA/yipeeo/Crop_data/Crop_yield/all/field_scale_{region}.shp')
    year_min, year_max = np.nanmin(fields.c_year.values),np.nanmax(fields.c_year.values)

    if year_max<2018:
        raise ValueError('no ECOSTRESS data available for the given fields at the time of harvesting')
    if year_min<2019:
        year_min=2019
        warnings.warn('There are harvest dates before 2018. No ECOSTRESS data is available by then. Hence, only ECOSTRESS data from 2018 onwards will be extracted')

    fields = fields.drop_duplicates(subset='field_id')

    #Define bounding box around fields where ECOSTRESS data will be loaded from
    xmin, xmax = fields.bounds.min(0).minx, fields.bounds.max(0).maxx
    ymin, ymax = fields.bounds.min(0).miny, fields.bounds.max(0).maxy
    bbox=(xmin, ymin,xmax,ymax)
    #Establish location and files where information will be stored
    #if new==False the observations will be added to the already existing files
    row_head = ['observation','day_time'] + list(fields.field_id.values)
    if not os.path.exists(rf'{table_path}\{region}'):
        os.makedirs(rf'{table_path}\{region}')
    if new:
        for collection in collections:
            file_path = rf'{table_path}\{region}\run_{collection}_all.csv'
            with open(file_path, 'w') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(row_head)

    # Setup connections to Earthdata server and search for available ECOSTRESS images
    earthaccess.login()
    for collection in collections:
        search_lst = earthaccess.search_data(
            short_name=collection,
            bounding_box=bbox,
            temporal=(f'{year_min}-01-01',f'{year_max}-12-31'),
            cloud_hosted=True,
        )

        search_geo = earthaccess.search_data(
            short_name='ECO1BGEO',
            bounding_box=bbox,
            temporal=(f'{year_min}-01-01', f'{year_max}-12-31'),
            cloud_hosted=True,
        )

        # Check if geo files with lat lon information are from the same date and time as LST data
        # and remove if they do not correspond
        for i in range(len(search_lst)):
            date_time_a = search_lst[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            date_time_b = search_geo[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            if not date_time_a==date_time_b:
                search_geo.pop(i)
        print(len(search_lst),len(search_geo))

        # for i_files in range(len(search_lst)):
        for i_files in range(5, 10):
            date_time = search_lst[i_files:i_files + 1][0]['umm']['TemporalExtent']['RangeDateTime'][
                'BeginningDateTime']
            print('\n', i_files, date_time)
            date, time = date_time.split('T')[0], date_time.split('T')[1][:5]

            file_lst = earthaccess.open(search_lst[i_files:i_files + 1])
            file_geo = earthaccess.open(search_geo[i_files:i_files + 1])

            sf, add_off, file_name = eco2tif(path_lst=file_lst[0], path_geo=file_geo[0], path_out=path_temp)

            src = rasterio.open(file_name)
            fields = fields.to_crs(src.crs)

            # Per scene and band extract and summarize the information per field
            fields_data = pd.DataFrame(data=None, index=range(2), columns=row_head)

            fields_data.iloc[:, 0] = [f'{date}_median', f'{date}_std']
            fields_data.iloc[:, 1] = [time, time]

            file_path = rf'{table_path}\{region}\run_{collection}_all.csv'

            for ipol in range(len(fields)):
                polygon = fields[ipol:ipol + 1]
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
        outName = rf'{path_out}\{file}_{core}.tif'

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
    a.run(n_cores=40, new=True)
    print('finished at: ',datetime.datetime.now())