import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from pandas.tseries.offsets import DateOffset
import xarray as xr
from pathlib import Path
import os
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time

warnings.filterwarnings('ignore')


def load_land_cover_data(lc_path, target_transform, target_shape, target_crs):
    """
    Load and reproject CCI Land Cover data to match MODIS grid

    Parameters:
    -----------
    lc_path : str
        Path to CCI Land Cover netCDF file
    target_transform : affine.Affine
        Target transform from MODIS data
    target_shape : tuple
        Target shape (height, width) from MODIS data
    target_crs : str/CRS
        Target CRS from MODIS data

    Returns:
    --------
    numpy.ndarray : Reprojected land cover data
    """
    print("Loading CCI Land Cover data...")

    # Load netCDF file
    with xr.open_dataset(lc_path) as ds:
        # Get the land cover variable (adjust variable name as needed)
        # Common names: 'lccs_class', 'land_cover_lccs', 'Band1'
        lc_var_names = ['lccs_class', 'land_cover_lccs', 'Band1', 'lc']
        lc_var = None

        for var_name in lc_var_names:
            if var_name in ds.variables:
                lc_var = ds[var_name]
                break

        if lc_var is None:
            print("Available variables:", list(ds.variables.keys()))
            raise ValueError("Could not find land cover variable. Please check variable names.")

        # Get coordinate information
        if 'lat' in ds.coords:
            lats = ds.lat.values
            lons = ds.lon.values
        elif 'latitude' in ds.coords:
            lats = ds.latitude.values
            lons = ds.longitude.values
        else:
            raise ValueError("Could not find latitude/longitude coordinates")

        # Extract land cover data
        lc_data = lc_var.values
        if len(lc_data.shape) == 3:  # If there's a time dimension
            lc_data = lc_data[0]  # Take first time step

    # Create source transform
    lon_res = (lons.max() - lons.min()) / (len(lons) - 1)
    lat_res = (lats.max() - lats.min()) / (len(lats) - 1)

    from rasterio.transform import from_bounds
    src_transform = from_bounds(
        lons.min() - lon_res / 2, lats.min() - lat_res / 2,
        lons.max() + lon_res / 2, lats.max() + lat_res / 2,
        len(lons), len(lats)
    )

    # Reproject to match MODIS grid
    print("Reprojecting land cover data to MODIS grid...")
    reprojected_lc = np.empty(target_shape, dtype=lc_data.dtype)

    reproject(
        source=lc_data,
        destination=reprojected_lc,
        src_transform=src_transform,
        src_crs='EPSG:4326',  # Assuming CCI is in WGS84
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest  # Use nearest neighbor for categorical data
    )

    return reprojected_lc

def create_cropland_mask(land_cover_data, cropland_classes=None):
    """
    Create a binary mask for cropland pixels based on CCI Land Cover classes

    Parameters:
    -----------
    land_cover_data : numpy.ndarray
        Land cover classification data
    cropland_classes : list
        List of class values representing cropland (CCI LC classes 10-40 are typically crops)

    Returns:
    --------
    numpy.ndarray : Binary mask (True for cropland)
    """
    if cropland_classes is None:
        # Default CCI Land Cover cropland classes
        # Classes 10-40 represent different crop types in CCI LC
        cropland_classes = list(range(10, 41))

    print(f"Creating cropland mask for classes: {cropland_classes}")
    cropland_mask = np.isin(land_cover_data, cropland_classes)

    print(f"Cropland pixels: {np.sum(cropland_mask)} ({np.sum(cropland_mask) / cropland_mask.size * 100:.2f}%)")

    return cropland_mask

def process_single_evi_file(evi_file, regions_gdf_pickle, cropland_mask, target_transform, target_crs):
    """
    Process a single EVI file and calculate regional means (for parallel processing)

    Parameters:
    -----------
    evi_file : str
        Path to EVI TIFF file
    regions_gdf_pickle : bytes
        Pickled GeoDataFrame of regions (for multiprocessing compatibility)
    cropland_mask : numpy.ndarray
        Binary mask for cropland pixels
    target_transform : affine.Affine
        Target transform from first EVI file
    target_crs : CRS
        Target CRS from first EVI file

    Returns:
    --------
    list : List of result dictionaries for this file
    """
    import pickle

    # Unpickle the regions GeoDataFrame
    regions_gdf = pickle.loads(regions_gdf_pickle)

    file_results = []

    # Extract date from filename
    try:
        filename = os.path.basename(evi_file)
        if '.A' in filename:
            date_str = filename.split('.A')[1][:7]  # YYYYDDD format
            date = datetime.strptime(date_str, '%Y%j').strftime('%Y-%m-%d')
        else:
            date = filename.split('.')[0]  # Use first part as date
    except:
        date = os.path.basename(evi_file)

    try:
        # Read EVI data
        with rasterio.open(evi_file) as src:
            evi_data = src.read(1)
            transform = src.transform
            crs = src.crs

            # Handle MODIS scaling
            evi_data = evi_data.astype(np.float32)
            evi_data = evi_data / 10000.0

            # Mask invalid values
            valid_mask = (evi_data >= -1) & (evi_data <= 1)
            evi_data = np.where(valid_mask, evi_data, np.nan)

        # Reproject regions to match EVI CRS if needed
        if regions_gdf.crs != crs:
            regions_proj = regions_gdf.to_crs(crs)
        else:
            regions_proj = regions_gdf

        # Process each region
        for idx, region in regions_proj.iterrows():
            # Create mask for current region

            region_mask = geometry_mask(
                [region.geometry],
                transform=transform,
                invert=True,
                out_shape=evi_data.shape
            )

            # Combine region mask with cropland mask
            combined_mask = region_mask & cropland_mask & ~np.isnan(evi_data)

            if np.sum(combined_mask) > 0:
                mean_evi = np.mean(evi_data[combined_mask])
                pixel_count = np.sum(combined_mask)
            else:
                mean_evi = np.nan
                pixel_count = 0
            file_results.append({
                'region_id': idx,
                'region_name': region.NUTS2_ID,
                'date': date,
                'mean_evi': mean_evi,
                'cropland_pixel_count': pixel_count,
                'file': os.path.basename(evi_file)
            })

    except Exception as e:
        print(f"Error processing {os.path.basename(evi_file)}: {str(e)}")
        return []

    return file_results

def process_evi_files_parallel(evi_files, regions_gdf, cropland_mask, target_transform, target_crs, n_cores=None):
    """
    Process multiple EVI files in parallel and calculate regional means

    Parameters:
    -----------
    evi_files : list
        List of paths to EVI TIFF files
    regions_gdf : GeoDataFrame
        Regions shapefile
    cropland_mask : numpy.ndarray
        Binary mask for cropland pixels
    target_transform : affine.Affine
        Target transform from first EVI file
    target_crs : CRS
        Target CRS from first EVI file
    n_cores : int, optional
        Number of CPU cores to use (default: all available cores)

    Returns:
    --------
    pandas.DataFrame : Results with columns for region, date, and mean EVI
    """
    import pickle

    if n_cores is None:
        n_cores = cpu_count()

    print(f"Processing {len(evi_files)} EVI files using {n_cores} CPU cores...")

    # Pickle the regions GeoDataFrame for multiprocessing compatibility
    regions_gdf_pickle = pickle.dumps(regions_gdf)

    # Create partial function with fixed parameters
    process_func = partial(
        process_single_evi_file,
        regions_gdf_pickle=regions_gdf_pickle,
        cropland_mask=cropland_mask,
        target_transform=target_transform,
        target_crs=target_crs
    )

    # Process files in parallel
    start_time = time.time()

    with Pool(n_cores) as pool:
        # Use imap to get progress updates
        results_list = []
        for i, file_results in enumerate(pool.imap(process_func, evi_files)):
            results_list.extend(file_results)
            if (i + 1) % 10 == 0 or (i + 1) == len(evi_files):
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(evi_files)} files ({elapsed / 60:.1f} minutes elapsed)")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    elapsed_total = time.time() - start_time
    print(f"Parallel processing completed in {elapsed_total / 60:.1f} minutes")
    print(f"Average processing time per file: {elapsed_total / len(evi_files):.2f} seconds")

    return results_df

def main(shapefile_path, evi_folder, land_cover_path, output_path, cropland_classes=None, n_cores=None):
    """
    Main function to process MODIS EVI data with land cover filtering (parallelized)

    Parameters:
    -----------
    shapefile_path : str
        Path to regions shapefile
    evi_folder : str
        Folder containing EVI TIFF files
    land_cover_path : str
        Path to CCI Land Cover netCDF file
    output_path : str
        Path for output CSV file
    cropland_classes : list, optional
        Custom cropland class values
    n_cores : int, optional
        Number of CPU cores to use (default: all available cores)
    """

    print("Starting MODIS EVI analysis (parallelized)...")
    print(f"Available CPU cores: {cpu_count()}")

    if n_cores is None:
        n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores")

    # Load regions shapefile
    print("Loading regions shapefile...")
    regions_gdf = gpd.read_file(shapefile_path)
    regions_gdf = regions_gdf.drop_duplicates(subset='NUTS2_ID')
    print(f"Loaded {len(regions_gdf)} regions")

    # Get list of EVI files
    evi_files = list(Path(evi_folder).glob('*.tif')) + list(Path(evi_folder).glob('*.tiff'))
    if not evi_files:
        raise ValueError(f"No TIFF files found in {evi_folder}")

    print(f"Found {len(evi_files)} EVI files")

    # Use first EVI file to get target grid information
    with rasterio.open(evi_files[0]) as src:
        target_transform = src.transform
        target_shape = src.shape
        target_crs = src.crs

    # Load and reproject land cover data
    land_cover_data = load_land_cover_data(
        land_cover_path, target_transform, target_shape, target_crs
    )

    # Create cropland mask
    cropland_mask = create_cropland_mask(land_cover_data, cropland_classes)

    # Process all EVI files in parallel
    results_df = process_evi_files_parallel(
        evi_files, regions_gdf, cropland_mask, target_transform, target_crs, n_cores
    )

    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Print summary statistics
    print("\nSummary:")
    print(f"Total records: {len(results_df)}")
    print(f"Regions processed: {results_df['region_name'].nunique()}")
    print(f"Date range: {results_df['date'].min()} to {results_df['date'].max()}")
    print(f"Valid EVI measurements: {results_df['mean_evi'].notna().sum()}")
    print(f"Mean EVI across all regions and dates: {results_df['mean_evi'].mean():.4f}")

    return results_df

def convert2LT_tab(csv_path, crop):
    """
    Convert csv with columns [region_name, date, mean_evi] to a dataframe indexed by region_name,
    with one row per region and 9 columns:
    'year', 'evi_LT1', ..., 'evi_LT8', where LT1 is mean mean_evi 0-14 days before that region's ref_date,
    LT2 is mean mean_evi 15-28 days before ref_date, etc.

    Parameters:
        csv_path (str): Path to the csv file.
        region_ref_dates (dict or pd.Series): Mapping of region_name to reference date (str or pd.Timestamp).

    Returns:
        pd.DataFrame: Dataframe indexed by region, showing year and 8 lag columns (means).
    """
    harvest_date_wheat = {'AT': '06-30', 'CZ': '07-16', 'DE': '07-16', 'FR': '07-11', 'HR': '06-30',
                          'HU': '06-30', 'PL': '07-16', 'SI': '07-16', 'SK': '07-16', 'UA': '07-16'}
    harvest_date_maize = {'AT': '09-30', 'CZ': '09-30', 'DE': '09-30', 'FR': '09-30', 'HR': '09-30',
                          'HU': '09-30', 'PL': '09-30', 'SI': '09-30', 'SK': '09-30', 'UA': '09-30'}
    harvest_date_barley = {'AT': '06-30', 'CZ': '07-16', 'DE': '07-16', 'FR': '07-11', 'HR': '06-30',
                           'HU': '06-30', 'PL': '07-16', 'SI': '07-16', 'SK': '07-16', 'UA': '07-16'}

    harvest_date_pc = {'winter_wheat': harvest_date_wheat,
                       'maize': harvest_date_maize,
                       'spring_barley': harvest_date_barley}

    region_ref_md = harvest_date_pc[crop]

    # Read CSV and parse date column
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['country'] = [a[:2] for a in df.region_name]
    print(df.head())

    if isinstance(region_ref_md, dict):
        region_ref_md = pd.Series(region_ref_md)

    results = []

    grouped = df.groupby('country')

    for region, group in grouped:
        # Get ref month-day for this region; skip if missing
        ref_md = region_ref_md.get(region, None)
        if ref_md is None:
            continue
        # Parse ref_md into month and day if given as string 'MM-DD'
        if isinstance(ref_md, str):
            month, day = map(int, ref_md.split('-'))
        else:
            month, day = ref_md  # assume tuple (month, day)

        # For each year in this region's data, compute lag means relative to ref month-day that year
        for year in group['date'].dt.year.unique():
            # Construct ref date for this year
            try:
                ref_date = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                # Invalid date (e.g., Feb 29 on non-leap year) - skip this year
                continue

            region_year_group = group[group['date'].dt.year == year]

            region_dict = {'region_name': region, 'year': year}

            for i in range(1, 9):
                start_date = ref_date - pd.Timedelta(days=14 * i)
                end_date = ref_date - pd.Timedelta(days=14 * (i - 1) + 1)

                mask = (region_year_group['date'] >= start_date) & (region_year_group['date'] <= end_date)
                values = region_year_group.loc[mask, 'mean_evi']

                lag_mean = values.mean() if not values.empty else np.nan
                region_dict[f'evi_LT{i}'] = lag_mean

            results.append(region_dict)
    result_df = pd.DataFrame(results).set_index(['region_name', 'year'])
    return result_df

def own(csv_path, crop):
    harvest_date_wheat = {'AT': [6, 30], 'CZ': [7, 16], 'DE': [7, 16], 'FR': [7, 11], 'HR': [6, 30],
                          'HU': [6, 30], 'PL': [7, 16], 'SI': [7, 16], 'SK': [7, 16], 'UA': [7, 16]}
    harvest_date_maize = {'AT': [9, 30], 'CZ': [9, 30], 'DE': [9, 30], 'FR': [9, 30], 'HR': [9, 30],
                          'HU': [9, 30], 'PL': [9, 30], 'SI': [9, 30], 'SK': [9, 30], 'UA': [9, 30]}
    harvest_date_barley = {'AT': [6, 30], 'CZ': [7, 16], 'DE': [7, 16], 'FR': [7, 11], 'HR': [6, 30],
                           'HU': [6, 30], 'PL': [7, 16], 'SI': [7, 16], 'SK': [7, 16], 'UA': [7, 16]}
    harvest_date_pc = {'winter_wheat': harvest_date_wheat, 'maize': harvest_date_maize,
                            'spring_barley': harvest_date_barley}
    harvest_date = harvest_date_pc[crop]

    lead_times = ['_LT8', '_LT7', '_LT6', '_LT5', '_LT4', '_LT3', '_LT2', '_LT1']
    params = ['evi']

    df = pd.read_csv(csv_path)
    df = df.dropna(axis=0)
    df['date'] = pd.to_datetime(df['date'])
    df['country'] = [a[:2] for a in df.region_name]
    print(df.columns)

    nuts = np.unique(df.region_name)

    years = range(2000, 2023)
    col_names = [f'evi_{i}' for i in range(1,9)]

    pipeline_df = pd.DataFrame(data=None, columns=['field_id', 'c_year'] + col_names, index=None)

    for nut in nuts:
        print(nut)
        nut_file = df.iloc[np.where(df.region_name==nut)[0], 1:4]
        nut_file['date'] = pd.to_datetime(nut_file['date'])
        # Sort by date
        nut_file_sorted = nut_file.sort_values('date')

        # Create a Series with 'date' as index and 'mean_evi' as values
        evi_series = pd.Series(data=nut_file_sorted['mean_evi'].values, index=nut_file_sorted['date'])
        years = np.unique(evi_series.index.year)
        years = [a for a in years if a<2023]
        for year in years:
            ndvi_m = evi_series.resample('2W').mean().interpolate()
            this_harvest_date = pd.to_datetime(f'{year}-{harvest_date[nut[:2]][0]}-{harvest_date[nut[:2]][1] - 2}')  # harvest date -2 days to make sure harvested field is not included
            start_date = this_harvest_date - DateOffset(months=4)
            this_df = ndvi_m[start_date:this_harvest_date]

            if len(this_df) >= len(lead_times):
                this_df = this_df[:len(lead_times)]

            # print(pipeline_df.loc[len(pipeline_df), :])
            new_vals = [nut, year] + list(this_df.values)
            pipeline_df.loc[len(pipeline_df), :] = new_vals
    # print(pipeline_df)
    pipeline_df.to_csv(f'Data/{crop}_evi_new.csv')

def add_yield(crop):
    file = pd.read_csv(f'Data/{crop}_evi_new.csv', index_col=0)
    ref = pd.read_csv(f'Data/{crop}_all_abs_fin_year_det.csv', index_col=0)
    file.loc[:, 'yield_anom'] = [np.nan] * len(file)
    for i in range(len(file)):
    # for i in range(100):
        field, year = file.iloc[i, :2]
        ind_old = np.where((ref.field_id==field) & (ref.c_year==year))[0]
        if len(ind_old)==1:
            file.iloc[i, -1] = ref.iloc[ind_old, 2]
    file = file.dropna(axis=0)

    file.to_csv(f'Data/{crop}_evi_new_fin.csv')

def reshape_evi():
    path = r"M:\Projects\YIPEEO\07_data\Predictors\SC2\evi.csv"
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    df = df.drop(columns=['region_id', 'cropland_pixel_count', 'file'])
    df.index = range(len(df))
    df_pivot = df.pivot_table(index='date', columns='region_name', values='mean_evi', aggfunc='sum')
    # df_pivot = df.pivot(index='date', columns='region_name', values='mean_evi')
    df_pivot.to_csv(path)

    # grouped = df.groupby('country')
    # print(grouped)


# Example usage
if __name__ == "__main__":
    # Define file paths (adjust these to your actual file paths)

    # pd.set_option('display.max_columns', None)
    reshape_evi()
    # shapefile_path = "C:/Users/pbueechi/Data/All_NUTS2_fin.shp"
    # evi_folder = "C:/Users/pbueechi/Data/tif"
    # land_cover_path = "C:/Users/pbueechi/Data/LCCEU.nc"
    # output_path = "C:/Users/pbueechi/Data/evi.csv"
    #
    # # Optional: Define specific cropland classes
    # # CCI Land Cover classes for cropland (typically 10-40)
    # cropland_classes = [10, 11, 12, 20, 30, 40]  # Adjust based on your CCI version
    #
    # # Optional: Specify number of CPU cores (default: use all available cores)
    # n_cores = 10  # Use all cores, or specify a number like n_cores = 4
    #
    # # Run analysis
    # results = main(
    #     shapefile_path=shapefile_path,
    #     evi_folder=evi_folder,
    #     land_cover_path=land_cover_path,
    #     output_path=output_path,
    #     cropland_classes=cropland_classes,
    #     n_cores=n_cores
    # )
    # add_yield('spring_barley')
    # add_yield('winter_wheat')
    # for crop in ['spring_barley', 'winter_wheat', 'maize']: own(output_path, crop)
    print("Analysis complete!")