import seaborn
import warnings
import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
# from pysal.lib import weights
from PIL import Image
from cmcrameri import cm as cmr
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Patch

class Moron:
    def __init__(self, y,w, permutations=99):
        self.y = y
        self.w = w
        sy = y.std()
        # self.z /= sy
        # print(self.z)
        self.n = len(self.y)
        self.z = y - y.mean()
        self.z /= sy
        self.ind = self.z.index
        self.z2ss = (self.z*self.z).sum()
        self.permutations = permutations
        self.I = self.calc(self.z)

    def calc(self, z=None):
        if z is not None:
            zl = [z.loc[self.w.neighbors[ind]].mean() for ind in self.ind]
            inum = (z * zl).sum()
        else:
            zl = [self.z.loc[self.w.neighbors[ind]].mean() for ind in self.ind]
            inum = (self.z * zl).sum()
        return self.n / self.w.s0 * inum / self.z2ss

    def calc_p(self):
        sim = [self.calc(np.random.permutation(self.z)) for i in range(self.permutations)]
        self.sim = sim = np.array(sim)
        above = sim >= self.I
        larger = above.sum()
        if (self.permutations - larger) < larger:
            larger = self.permutations - larger
        p_sim = (larger + 1.0) / (self.permutations + 1.0)
        return p_sim

def crop_summary_stats():
    path = r'M:\Projects\YIPEEO\07_data\Crop yield\Database\field_scale_050524.shp'
    file = gpd.read_file(path)
    file = file.iloc[np.where(file.c_year>2015)[0],:]
    # file = file.iloc[np.where(file.country_co!='es')[0],:]
    crops_obs = file.crop_type.value_counts()
    crops_test = crops_obs[crops_obs > 100].index
    obs = pd.DataFrame(index=crops_test, columns=np.unique(file.country_co))
    obs.replace(np.nan, 0)
    for crop in crops_test:
        ind = np.where(file.crop_type==crop)[0]
        sub_file = file.iloc[ind,:]
        crops_obs = sub_file.country_co.value_counts()
        for country in crops_obs.index:
            obs.loc[crop, country] = crops_obs[country]
    obs.to_csv('Results/explore_data/obs_country_24_spa.csv')

def plot_crop_summary():
    file = pd.read_csv('Results/explore_data/obs_country_24.csv', index_col=0)
    indes_repl = {'grain maize and corn-cob-mix': 'maize', 'common winter wheat':'winter wheat','winter rape and turnip rape seeds':'winter rape'}
    file.index = [indes_repl[a] if a in indes_repl.keys() else a for a in file.index]

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)

    cols = cm.get_cmap('plasma', len(file.columns)).colors

    left=[0]*len(file.index)
    for i in range(len(file.columns)):
        ax1.bar(file.index, file.iloc[:, i], bottom=left, color=cols[i], label=file.columns[i])
        left = left+file.iloc[:,i]
    for label in (ax1.get_yticklabels()+ax1.get_xticklabels()):
        label.set_fontsize(18)

    ax1.set_ylabel("observations", fontsize=18)
    ax1.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=20)
    ha, le = ax1.get_legend_handles_labels()
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.95)
    ax1.legend(ha,le,ncol=2, fontsize=18, loc=1)

    plt.savefig('Results/explore_data/barplot_crops_24.png', dpi=300)

def plot_crop_year(reg='all'):
    seaborn.set_style('whitegrid')
    if reg=='ukraine':
        file = pd.read_csv('Results/SC2/loocv/UA/all_crops_1_anom.csv')
        s = seaborn.boxplot(data=file.explode('observed'), x='year', y='observed', hue='crop')
    elif reg=='cz':
        file = pd.read_csv('Data/M/cz/rost_all_crops.csv', decimal=',')
        s = seaborn.boxplot(data=file.explode('yield'), x='c_year', y='yield', hue='crop_short')
    elif reg=='all':
        file = pd.read_csv('Data/SC2/M/final/crop_yield_sum.csv')
        file.crop = [a.replace('_', ' ').capitalize() for a in file.crop]
        s = seaborn.boxplot(data=file.explode('yield_anom'), x='crop', y='yield_anom', hue='country', gap=0.2)
        nobs = file['crop'].value_counts()
        print(nobs)
        print(nobs["Spring barley"])
        label_mapping = {'Maize': f'Maize\n obs={nobs.Maize}',
                         'Winter wheat': f'Winter wheat\n obs={nobs["Winter wheat"]}',
                         'Spring barley': f'Spring barley\n obs={nobs["Spring barley"]}'}
        current_labels = [tick.get_text() for tick in s.get_xticklabels()]
        new_labels = [label_mapping.get(label, label) for label in current_labels]
        s.set_xticklabels(new_labels)

        s.set_xlabel('')
    else:
        raise ValueError(f'reg not available. Please choose from ukraine, cz, all')
    # s.set_xlabel('Year')
    s.set_ylabel('Yield [t/ha]')
    [s.axvline(x + .5, color='k') for x in s.get_xticks()]
    s.legend(ncol=5, title=None)
    plt.show()
    # if reg=='ukraine':
    #     plt.savefig(r'Results/SC2/Figs/UA_crop_hist.png', dpi=300)
    # elif reg=='cz':
    #     plt.savefig(r'Figures/Data_prep/crop_hist.png', dpi=300)
    # elif reg=='all':
    #     plt.savefig(r'Results/SC2/Figs/202506/crop_hist.png', dpi=300)


def plot_crop_year_reg():
    crops = ['winter_wheat', 'spring_barley', 'maize']
    for i, crop in enumerate(crops):
        for s, scale in enumerate(['field', 'regional']):
            path = f'Data/M/TL/{crop}_s1s2_{scale}.csv'
            if (i == 0) & (s == 0):
                csv_fin = pd.read_csv(path).loc[:,['field_id', 'c_year', 'yield']]
                csv_fin.loc[:,'crop_short'] = [f"{crop.replace('_', ' ')} {scale}"] * csv_fin.shape[0]
            else:
                csv_next = pd.read_csv(path).loc[:,['field_id', 'c_year', 'yield']]
                csv_next.loc[:, 'crop_short'] = [f"{crop.replace('_', ' ')} {scale}"] * csv_next.shape[0]
                csv_fin = pd.concat([csv_fin, csv_next])

    csv_fin = csv_fin.iloc[np.where(csv_fin.c_year>2016)[0], :]

    palette = {'winter wheat field':"#9ecae1", 'winter wheat regional':"#3182bd",
               'spring barley field': "#a1d99b", 'spring barley regional': "#31a354",
               'maize field': "#fa9fb5", 'maize regional': "#c51b8a",
               }

    fig = plt.figure(figsize=(15, 9))
    seaborn.set(font_scale=1.9, style='whitegrid')
    years = np.unique(csv_fin.c_year)
    positions = np.arange(len(years)) + 0.5 * (np.arange(len(years)) // 2)
    s = seaborn.boxplot(data=csv_fin.explode('yield'), x='c_year', y='yield', hue='crop_short', palette=palette, gap=0.1, width=0.9)

    s.set_xlabel('Year')
    s.set_ylabel('Yield anomalies [t/ha]')
    [s.axvline(x + .5, color='k') for x in s.get_xticks()]
    s.legend_.set_title(None)
    h, l = s.get_legend_handles_labels()
    s.legend_.remove()
    s.legend(h, l, ncol=3)
    plt.subplots_adjust(bottom=0.11, left=0.09, right=0.95, top=0.95)
    # plt.show()
    plt.savefig(r'H:\Emanuel\Code_new\yipeeo\Results\Figs/crop_hist.png', dpi=300)
    plt.close()



def merge_ua():
    crops = ['maize', 'winter_wheat', 'spring_barley']
    for i, crop in enumerate(crops):
        file = pd.read_csv(f'Results/SC2/loocv/UA/{crop}_1.csv', index_col=0)
        file.loc[:, 'crop'] = [crop] * file.shape[0]
        file.loc[:, 'observed'] = file.loc[:, 'observed'] - np.mean(file.loc[:, 'observed'])
        if i==0:
            res = file
        else:
            res = pd.concat([res, file])

    res.to_csv('Results/SC2/loocv/UA/all_crops_1_anom.csv')


def plot_crop_year_pred():
    file = pd.read_csv('Results/forecasts/cz_common winter wheat_RF_loocv_opt=True.csv', decimal=',', index_col=0)
    file = file.melt(id_vars='year', var_name='name')
    file = file.iloc[np]
    print(file.explode('value'))

    file = pd.read_csv('Data/M/cz/rost_all_crops.csv', decimal=',')
    print(file.explode('yield'))
    seaborn.set_style('whitegrid')
    s = seaborn.boxplot(data=file.explode('value'), x='year', y='value', hue='name')
    s.set_xlabel('Year')
    s.set_ylabel('Yield [t/ha]')
    [s.axvline(x + .5, color='k') for x in s.get_xticks()]
    s.legend_.set_title(None)
    plt.show()
    # plt.savefig(r'M:\Projects\YIPEEO\04_deliverables_documents\03_PVR\Figures\crop_hist.png', dpi=300)

def prepare_crop_hist():
    crops = ['maize', 'winter_wheat', 'spring_barley']
    for i, crop in enumerate(crops):
        path = f'Data/SC2/M/final/{crop}_all_abs_fin.csv'
        file = pd.read_csv(path, index_col=0).iloc[:,:3]
        file.loc[:, 'crop'] = [crop] * file.shape[0]
        file.loc[:, 'country'] = [a[:2] for a in file.loc[:, 'field_id']]
        if i==0:
            res = file
        else:
            res = pd.concat([res, file])

    res.to_csv('Data/SC2/M/final/crop_yield_sum.csv')



def ml_char_plot():
    file = pd.read_csv('Data/M/common winter wheat_all.csv', decimal=',')

    # seaborn.set_style('whitegrid')
    seaborn.displot(file, x="yield", hue="region", kind="kde", fill=True)
    plt.title('Winter wheat')
    plt.subplots_adjust(top=0.95)
    plt.xlabel('Yield [t/ha]')
    # seaborn.set_style('whitegrid')
    # seaborn.jointplot(data=file, x="ndvi_LT2", y='sig40_cr_mean_daily_LT2', hue="region", kind="kde")

    # plt.show()
    plt.savefig(r'M:\Projects\YIPEEO\04_deliverables_documents\03_PVR\Figures\yield_density.png', dpi=300)

def merge_plots():
    # Read the two images
    image1 = Image.open(r'M:\Projects\YIPEEO\04_deliverables_documents\03_PVR\Figures\yield_density.png')
    image2 = Image.open(r'M:\Projects\YIPEEO\04_deliverables_documents\03_PVR\Figures\pred_density.png')
    # resize, first image
    # image1 = image1.resize((426, 240))
    image1_size = image1.size
    image2 = image2.resize(image1_size)
    new_image = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (250, 250, 250))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    new_image.save(r'M:\Projects\YIPEEO\04_deliverables_documents\03_PVR\Figures\merged_image.png', "PNG")
    # new_image.show()

def morans(crop):
    path = rf'D:\DATA\yipeeo\Crop_data\Crop_yield\all\field_scale_rost.shp'
    fields = gpd.read_file(path)
    # ToDo: filter data per crop
    fields.index = fields.field_id
    mors = []
    years = np.unique(fields.c_year)
    for year in years[:1]:
        fields_year = fields.iloc[np.where(fields.c_year==year)[0],:]
        w = weights.Queen.from_dataframe(fields_year, idVariable='field_id')
        w.transform = 'R'
        y = fields_year['yield']
        a = Moron(y, w)
        mors.append(a.calc())
        print(f'Spatial autocorrelation of {crop} in {year}: {a.calc()} Morans I')
    return mors

def yearly_val():
    crops = ['common winter wheat', 'grain maize and corn-cob-mix', 'spring barley']

    fig = plt.figure(figsize=(8, 4))
    outer = gridspec.GridSpec(3, 4, height_ratios=[0.45,0.45,0.1])
    ax1 = plt.Subplot(fig, outer[0])

    path = f'Results/forecasts/cz_{crops[1]}_XGB_loocv_opt=True.csv'
    file = pd.read_csv(path, index_col=0)
    seaborn.set_style('whitegrid')
    s = seaborn.scatterplot(ax=ax1,data=file.explode('observed'), x='observed', y='forecast_LT_2', hue='year', palette=cmr.berlin)
    s.set_xlabel('Observed yield [t/ha]')
    s.set_ylabel('Forecasted yield LT1 [t/ha]')
    # [s.axvline(x + .5, color='k') for x in s.get_xticks()]
    corr = pearsonr(file.observed, file.forecast_LT_2)[0]
    s.legend_.set_title(f'pearson R: {np.round(corr,2)}')
    fig.add_subplot(ax1)
    plt.show()


def overview_map_ua():
    # Load the shapefiles
    nc_file = xr.open_dataset(r'M:\Datapool\koeppen_geiger_climate_classification\02_processed\Kottek2006\datasets\kottek2006_kgc.nc')
    shapefile2 = gpd.read_file(r'M:\Projects\YIPEEO\07_data\Crop yield\Regional_final/All_NUTS2_fin.shp')
    country_borders = gpd.read_file('Data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
    fontsize = 18

    # Define the column name for coloring shapefile1
    variable_name  = 'climate_class'

    color_mapping  = {
        7: ('#e28743', 'BSh'),
        6: ('#eab676', 'BSk'),
        11: ('#AFB42B', 'CSa'),
        12: ('#DCE775', 'CSb'),
        8: ('#005a32', 'Cfa'),
        9: ('#238443', 'Cfb'),
        10: ('#41ab5d', 'Cfc'),
        19: ('#CE93D8', 'Dfc'),
        18: ('#88419d', 'Dfb'),
        17: ('#6e016b', 'Dfa'),
        30: ('#80DEEA', 'ET'),
        29: ('#1E88E5', 'EF'),
    }
    koppen_geiger_map = {
        7: {"code": "BSh", "color": "#e28743", "name": "Hot semi-arid"},
        6: {"code": "BSk", "color": "#eab676", "name": "Cold semi-arid"},
        11: {"code": "CSa", "color": "#AFB42B", "name": "Hot-summer Mediterranean"},
        12: {"code": "CSb", "color": "#DCE775", "name": "Warm-summer Mediterranean"},
        8: {"code": "Cfa", "color": "#005a32", "name": "Humid subtropical"},
        9: {"code": "Cfb", "color": "#238443", "name": "Oceanic"},
        10: {"code": "Cfc", "color": "#41ab5d", "name": "Subpolar oceanic"},
        19: {"code": "Dfc", "color": "#CE93D8", "name": "Subarctic"},
        18: {"code": "Dfb", "color": "#88419d", "name": "Humid continental, warm summer"},
        17: {"code": "Dfa", "color": "#6e016b", "name": "Humid continental, hot summer"},
        30: {"code": "ET", "color": "#80DEEA", "name": "Tundra"},
        29: {"code": "EF", "color": "#1E88E5", "name": "Ice Cap"},
    }

    # for map in color_mapping.items():
    #     print(map)
    # Create a custom colormap

    colors = [color_mapping.get(i, ('gray', 'Other'))[0] for i in range(max(color_mapping.keys()) + 1)]
    # colors = [koppen_geiger_map[i]['color'] for i in range(1, 31)]
    custom_cmap = ListedColormap(colors)

    # Set up the plot with a specific projection (you can change this as needed)
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot the raster data
    nc_file[variable_name].plot(ax=ax, cmap=custom_cmap, add_colorbar=False)

    # Plot country borders
    country_borders.boundary.plot(ax=ax, color='black', linewidth=3)
    shapefile2.boundary.plot(ax=ax, color='#525252', linestyle='--', linewidth=2)


    # Set the extent to focus on Europe
    ax.set_xlim([-7, 43])
    ax.set_ylim([40, 60])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

        # Create legend
    # legend_elements = [Patch(facecolor=color, edgecolor='black', label=label)
    #                    for value, (color, label) in color_mapping.items()]
    legend_elements = [Patch(facecolor=map[1]['color'], edgecolor='black', label=map[1]['code'])
                       for map in koppen_geiger_map.items()]

    # Add a line for country borders
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=3, label='Countries'))
    legend_elements.append(plt.Line2D([0], [0], color='#525252', linestyle='-', lw=2, label='NUTS2'))

    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1, 0.1), fontsize=fontsize, ncol=1)

    # Add title and labels
    ax.set_xlabel('Longitude [°E]', fontsize=fontsize)
    ax.set_ylabel('Latitude [°N]', fontsize=fontsize)

    # Add gridlines
    # ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    # Adjust layout to make room for the legend
    # plt.subplots_adjust(left=0.05, right=0.75, bottom=0.05, top=0.98)
    plt.tight_layout()

    # Show the plot
    # plt.show()
    # plt.savefig('Results/SC2/Figs/Overview_map.png', dpi=300)
    fig_path = 'Figures/Paper_orig'
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(f'{fig_path}/Fig_1.png', dpi=300)


if __name__ == '__main__':
    pd.set_option('display.max_columns', 15)
    warnings.filterwarnings('ignore')
    # prepare_crop_hist()
    # plot_crop_year_reg()
    # overview_map_ua()
    # merge_ua()
    plot_crop_year()
    # crop_summary_stats()
    # plot_crop_summary()
    # merge_plots()
    # plot_crop_year_pred()
    print('all done')