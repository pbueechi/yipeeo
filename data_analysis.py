import seaborn
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime



def crop_summary_stats():
    path = r'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale_rest.shp'
    file = gpd.read_file(path)
    file = file.iloc[np.where(file.c_year>2015)[0],:]
    crops_obs = file.crop_type.value_counts()
    crops_test = crops_obs[crops_obs > 50].index
    obs = pd.DataFrame(index=crops_test, columns=np.unique(file.country_co))
    obs.replace(np.nan, 0)
    for crop in crops_test:
        ind = np.where(file.crop_type==crop)[0]
        sub_file = file.iloc[ind,:]
        crops_obs = sub_file.country_co.value_counts()
        for country in crops_obs.index:
            obs.loc[crop, country] = crops_obs[country]
    obs.to_csv('Results/explore_data/obs_country_50.csv')

def plot_crop_summary():
    file = pd.read_csv('Results/explore_data/obs_country_50.csv', index_col=0)
    indes_repl = {'grain maize and corn-cob-mix': 'maize', 'common winter wheat':'winter wheat','winter rape and turnip rape seeds':'winter rape'}
    file.index = [indes_repl[a] if a in indes_repl.keys() else a for a in file.index]

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)

    cols = cm.get_cmap('tab20c', len(file.columns)).colors

    left=[0]*len(file.index)
    for i in range(len(file.columns)):
        ax1.bar(file.index, file.iloc[:, i], bottom=left, color=cols[i], label=file.columns[i])
        left = left+file.iloc[:,i]
    for label in (ax1.get_yticklabels()+ax1.get_xticklabels()):
        label.set_fontsize(12)

    ax1.set_ylabel("observations", fontsize=14)
    ax1.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=45)
    ha, le = ax1.get_legend_handles_labels()
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.95)
    ax1.legend(ha,le,ncol=4, fontsize=12, loc=1)

    plt.savefig('Results/explore_data/barplot_crops_50.png', dpi=300)

def plot_crop_year(crop):
    file = pd.read_csv(f'Data/M/cz/rost_{crop}_all.csv', index_col=0)
    seaborn.boxplot(data=file, y="yield", x="c_year")
    plt.show()


if __name__ == '__main__':
    pd.set_option('display.max_columns', 15)
    warnings.filterwarnings('ignore')

    # Predictors resampling
    crops = ['common winter wheat','grain maize and corn-cob-mix','spring barley']
    for crop in crops[2:]:
        plot_crop_year(crop)