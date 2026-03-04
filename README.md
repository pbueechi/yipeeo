This repository contains the code developed in the project Yield Prediction and Estimation from Earth Observation (YIPEEO) funded by ESA. It is used to forecast crop yields at regional and field scales.

Used predictor datasets are from Earth observation (Sentinel- 1 and Sentinel-2, ESA CCI Soil Moisture, Vegetation Optical Depth Climate Archive, ECOSTRESS, and MODIS) and reanalysis from ERA5-Land.
Unfortunately most crop yield data cannot be shared, hence the code cannot be run entirely.

The data extraction per region or field is done in the files ecostress.py, extract_evi.py, s2_extract.py.
The actual crop yield forecasting is done in the files ml_forecast.py. There are also the files ANN_pred.py and XGB_pred.py for more specific model runs. 
This repository also includes the code for two papers published (and under review) in the project: the file tl_paper.py contains all modelling done in the paper Making optimal use of limited field-scale data for crop yield forecasting using transfer learning and Sentinel-1 and 2 data (https://doi.org/10.1016/j.atech.2025.101567) and Estimating war-related crop yield losses in Ukraine - a novel approach using transfer learning (https://dx.doi.org/10.2139/ssrn.5934805)
