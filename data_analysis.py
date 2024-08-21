import seaborn
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pysal.lib import weights
from PIL import Image
from cmcrameri import cm as cmr
from scipy.stats import pearsonr, spearmanr

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

def plot_crop_year():
    file = pd.read_csv('Data/M/cz/rost_all_crops.csv', decimal=',')

    seaborn.set_style('whitegrid')
    s = seaborn.boxplot(data=file.explode('yield'), x='c_year', y='yield', hue='crop_short')
    s.set_xlabel('Year')
    s.set_ylabel('Yield [t/ha]')
    [s.axvline(x + .5, color='k') for x in s.get_xticks()]
    s.legend_.set_title(None)
    plt.show()
    # plt.savefig(r'M:\Projects\YIPEEO\04_deliverables_documents\03_PVR\Figures\crop_hist.png', dpi=300)

def plot_crop_year_pred():
    file = pd.read_csv('Results/forecasts/cz_common winter wheat_RF_loocv_opt=True.csv', decimal=',', index_col=0)
    file = file.melt(id_vars='year', var_name='name')
    file = file.iloc[np]
    print(file.explode('value'))

    file = pd.read_csv('Data/M/cz/rost_all_crops.csv', decimal=',')
    print(file.explode('yield'))
    # seaborn.set_style('whitegrid')
    # s = seaborn.boxplot(data=file.explode('value'), x='year', y='value', hue='name')
    # s.set_xlabel('Year')
    # s.set_ylabel('Yield [t/ha]')
    # [s.axvline(x + .5, color='k') for x in s.get_xticks()]
    # s.legend_.set_title(None)
    # plt.show()
    # plt.savefig(r'M:\Projects\YIPEEO\04_deliverables_documents\03_PVR\Figures\crop_hist.png', dpi=300)


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


if __name__ == '__main__':
    pd.set_option('display.max_columns', 15)
    warnings.filterwarnings('ignore')
    # crop_summary_stats()
    # plot_crop_summary()
    # merge_plots()
    plot_crop_year_pred()