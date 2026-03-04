import os
import matplotlib.pyplot as plt
import pandas as pd
from data_analysis import plot_crop_year
from s_2 import plot_s2_class
import numpy as np
from ml_forecast import plotting, run_dl, plot_tlvsfield, plot_loocv, plot_sample_random_vs_cluster, \
    plot_dist_crop_preds, tab_diff_sample, plotting_cum_compare
from datetime import datetime

start_pro = datetime.now()
# All functions that are required to plot the figures of the transferlearning paper are summarized here.
# Except for Fig. 1 that is produced by Lucie, Fig. 4 done in libreoffice

#Fig. 2 Crop_hist
# plot_crop_year()

#Fig. 3 cloud_mask
# plot_s2_class(region='czr')

#Fig. 5-8 are based on:     #ToDo: adjust link in the upcoming functions
path_out_t = 'Results/Validation/dl/202510_revision'
# plotting_cum_compare(path_out_t)
# run_dl(cv_method='random', lts=[1, 2, 3, 4], path_out_t=path_out_t, standardize=True)
# run_dl(cv_method='random', lts=[1, 2, 3, 4], path_out_t=path_out_t, standardize=True)
# run_dl(cv_method='random', lts=[1], path_out_t=path_out_t, standardize=False, feature_importance=True)
# run_dl(cv_method='loocv', lts=[1], path_out_t=path_out_t, standardize=True)
# run_dl(cv_method='random', lts=[1], p_tls=np.linspace(0.1, 0.9, 9), path_out_t=path_out_t)
# run_dl(cv_method='random', lts=[1], p_tls=[0.2], not_random=True, path_out_t=path_out_t)

#Fig. 5 tl_comparison_2
# plotting(basepath=os.path.join(path_out_t,'run5_random'))

#Fig. 6 tlvsfield
# plot_tlvsfield(basepath=os.path.join(path_out_t,'run7_random'))

#Fig. 7 impact of sampling
# plot_sample_random_vs_cluster(basepath=path_out_t)

#Fig. 8 loocv
# plot_loocv(basepath=os.path.join(path_out_t,'run6_loocv'))

#Fig. 9
# plot_dist_crop_preds(basepath=path_out_t, standardize=False)
# for crop in ['winter_wheat', 'spring_barley', 'maize']: tab_diff_sample(crop)

print(f'calculation stopped and took {datetime.now() - start_pro}')

