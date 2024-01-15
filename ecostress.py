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
from pystac.extensions.eo import EOExtension as eo
from scipy import stats as st
from pystac_client import Client
from pyresample import geometry as geom
from pyresample import kd_tree as kdt
from rasterio.mask import mask
from osgeo import gdal, gdal_array, gdalconst, osr



def extract_s2(region, year=None, new=True):
    """
    :param region: str of region where fields are located which will be extract from S-2 L2A data. So far available: czr, nl, rom, ukr_chmel, ukr_horod, ukr_lviv
    :param year: int if only individual years should be considered
    :param new: binary True/False. Set True if new files should be generated. Attention this will delete potential older files there
    :return: dataframe of S2 observations per field
    """
    #ToDo adjust that all fields can be loaded at the same time and loop through the farms by selecting by attribute farm_code
    print('started calculating...')

    #Load fields and remove identical fields
    fields = gpd.read_file(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale_{region}.shp')
    year_min, year_max = np.nanmin(fields.c_year.values),np.nanmax(fields.c_year.values)

    if year_max<2016:
        raise ValueError('no S-2 data available for the given fields at the time of harvesting')
    if year_min<2016:
        year_min=2016
        warnings.warn('There are harvest dates before 2016. No S-2 data is available by then. Hence, only S-2 data from 2016 onwards will be extracted')

    fields = fields.drop_duplicates(subset='field_id')

    #Define bounding box around fields where S2 data will be loaded from
    xmin, xmax = fields.bounds.min(0).minx, fields.bounds.max(0).maxx
    ymin, ymax = fields.bounds.min(0).miny, fields.bounds.max(0).maxy
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

    # Set considered time and bands that should be extracted. So far set to visual, vegetation red edge, NIR and SWIR
    if year:
        time_of_interest = f'{year}-01-01/{year}-12-31'
    else:
        time_of_interest = f'{year_min}-01-01/{year_max}-12-31'
    bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']

    # Setup connections to planetary computer and search for available S-2 images
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 80}},               #Only scenes with cloud_cover<80% are considered
    )

    items = search.item_collection()

    #Establish files where information will be stored
    row_head = ['observation', 'cloud_cover[%]'] + list(fields.field_id.values)

    if not os.path.exists(rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}'):
        os.makedirs(rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}')

    if new:
        for band in bands:
            if year:
                file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_{year}.csv'
            else:
                file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_all.csv'
            with open(file_path, 'w') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(row_head)

    # Start looping through all S-2 scenes and defined S-2 bands
    #ToDo Check 58 and 113, ?156 >2018-07-15
    #2nd round fails at 58 (2022-03-26); 37 2021-09-14; 48 2021-04-05; 50 2020-07-19; 48 2020-02-02; 48 2019-07-07;
    #48 2018-12-04, 46 2018-07-05, 51 2017-10-13 (434:)
    #Error but continues: 76 2022-01-02
    #horod 41 2021-09-02; 477 2018-11-07
    ###
    for it_num,item in enumerate(items):
        print(it_num, item.datetime.date())
        # if (it_num>0) and (item.datetime.date()==prev_date):
        #     print('date already covered')
        #     continue
        # prev_date = item.datetime.date()
        signed_item = planetary_computer.sign(item)
        for band in bands:
            #Load S2 scenes
            src = rasterio.open(signed_item.assets[band].href)
            fields = fields.to_crs(src.crs)

            #Per scene and band extract and summarize the information per field
            fields_data = pd.DataFrame(data=None, index=range(2), columns=row_head)
            fields_data.iloc[:,0] = [f'{item.datetime.date()}_median', f'{item.datetime.date()}_std']
            fields_data.iloc[:, 1] = [eo.ext(item).cloud_cover, eo.ext(item).cloud_cover]

            if year:
                file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_{year}.csv'
            else:
                file_path = rf'D:\data-write\YIPEEO\predictors\S2_L2A\ts\{region}\run_{band}_all.csv'

            for ipol in range(len(fields)):
                polygon = fields[ipol:ipol+1]
                try:
                    out_image, out_transform = rasterio.mask.mask(src, polygon.geometry, crop=True)
                    out_image = np.where(out_image == 0, np.nan, out_image)
                    if np.isnan(out_image).all():
                        fields_data.loc[:,polygon.field_id.values[0]] = [np.nan, np.nan]
                    else:
                        if band=='SCL':
                            arr_flat = out_image.flatten()
                            arr_flat = arr_flat[~np.isnan(arr_flat)]
                            fields_data.loc[:,polygon.field_id.values[0]] = [int(st.mode(arr_flat)[0][0]), np.nanstd(out_image)]
                        else:
                            fields_data.loc[:,polygon.field_id.values[0]] = [np.nanmedian(out_image), np.nanstd(out_image)]
                except:
                    fields_data.loc[:, polygon.field_id.values[0]] = [np.nan, np.nan]
                    # warnings.warn('Input shapes do not overlap raster')

            with open(file_path, 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(list(fields_data.iloc[0,:]))
                writer.writerow(list(fields_data.iloc[1,:]))

            gc.collect()

def test_eco(region='nl', new=True):
    print('started calculating...')
    #Define datasets that should be loaded: ECOSTRESS L1 Radiation and L2 LSTE and clouds
    collections = ['ECO2CLD', 'ECO2LSTE']
    path_temp = r'D:\DATA\yipeeo\Predictors\ECOSTRESS\temp_files'

    #Load fields and remove identical fields
    fields = gpd.read_file(rf'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale_{region}.shp')
    year_min, year_max = np.nanmin(fields.c_year.values),np.nanmax(fields.c_year.values)

    if year_max<2018:
        raise ValueError('no ECOSTRESS data available for the given fields at the time of harvesting')
    if year_min<2019:
        year_min=2019
        warnings.warn('There are harvest dates before 2018. No ECOSTRESS data is available by then. Hence, only ECOSTRESS data from 2018 onwards will be extracted')

    fields = fields.drop_duplicates(subset='field_id')

    #Define bounding box around fields where S2 data will be loaded from
    xmin, xmax = fields.bounds.min(0).minx, fields.bounds.max(0).maxx
    ymin, ymax = fields.bounds.min(0).miny, fields.bounds.max(0).maxy
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
    bbox=(xmin, ymin,xmax,ymax)
    #Establish location and files where information will be stored
    #if new==False the observations will be added to the already existing files
    row_head = ['observation','day_time'] + list(fields.field_id.values)
    table_path = r'D:\data-write\YIPEEO\predictors\ECOSTRESS\ts'
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
        if not len(search_lst)==len(search_geo):
            print('file searches do not have same length')
            for i in range(len(search_lst)):
                date_time_a = search_lst[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
                date_time_b = search_geo[i:i + 1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
                if not date_time_a==date_time_b:
                    search_geo.pop(i)
        for i_files in range(12,len(search_lst)):
        # for i_files in range(1,2):
            date_time = search_lst[i_files:i_files+1][0]['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            print(i_files, date_time)
            date, time = date_time.split('T')[0], date_time.split('T')[1][:5]

            file_lst = earthaccess.open(search_lst[i_files:i_files+1])
            file_geo = earthaccess.open(search_geo[i_files:i_files+1])

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
                        fields_data.loc[:, polygon.field_id.values[0]] = [np.nanmedian(out_image)*sf+add_off, np.nanstd(out_image)*sf+add_off]
                except:
                    fields_data.loc[:, polygon.field_id.values[0]] = [np.nan, np.nan]
                    # warnings.warn('Input shapes do not overlap raster')

            with open(file_path, 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(list(fields_data.iloc[0, :]))
                writer.writerow(list(fields_data.iloc[1, :]))

            gc.collect()



def eco2tif(path_lst, path_geo, path_out, uncertainty=False):
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
        outName = rf'{path_out}\{file}.tif'

        # Get driver, specify dimensions, define and set output geotransform
        height, width = outFiles[file].shape
        driv = gdal.GetDriverByName('GTiff')
        dataType = gdal_array.NumericTypeCodeToGDALTypeCode(outFiles[file].dtype)
        d = driv.Create(outName, width, height, 1, dataType)
        d.SetGeoTransform(gt)

        # Create and set output projection, write output array data
        # Define target SRS
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        d.SetProjection(srs.ExportToWkt())
        srs.ExportToWkt()

        # Write array to band
        band = d.GetRasterBand(1)
        band.WriteArray(outFiles[file])

        # Define fill value if it exists, if not, set to mask fill value
        if fv is not None and fv != 'NaN':
            band.SetNoDataValue(fv)
        else:
            try:
                band.SetNoDataValue(outFiles[file].fill_value)
            except:
                pass
        band.FlushCache()
        d, band = None, None

    return sf, add_off, outName


if __name__=='__main__':
    print(datetime.datetime.now())
    test_eco(new=False)
    print('finished at: ',datetime.datetime.now())