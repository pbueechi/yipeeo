import os
import warnings
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
# from data_analysis import overview_map_ua
import numpy as np
from rich import palette
# from XGB_pred import run_xgb_ua
# from ANN_pred import run_ann_ua, run_fi
from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error
from datetime import datetime
from scipy.stats import ttest_ind
import matplotlib.gridspec as gridspec
# from keras.src.losses import mean_absolute_percentage_error

start_pro = datetime.now()

def plot_crop():
    crops = ['maize', 'winter_wheat', 'spring_barley']
    for i, crop in enumerate(crops):
        path = f'Data/SC2/nuts2/{crop}_M.csv'
        file = pd.read_csv(path, index_col=0).iloc[:,:3]
        file.loc[:, 'crop'] = [crop] * file.shape[0]
        file.loc[:, 'country'] = [a[:2] for a in file.loc[:, 'field_id']]
        if i==0:
            res = file
        else:
            res = pd.concat([res, file])

    seaborn.set_style('whitegrid')

    file = res
    file.crop = [a.replace('_', ' ').capitalize() for a in file.crop]
    s = seaborn.boxplot(data=file.explode('yield'), x='crop', y='yield', hue='country', gap=0.2, palette='colorblind')
    nobs = file['crop'].value_counts()
    label_mapping = {'Maize': f'Maize\n obs={nobs.Maize}',
                     'Winter wheat': f'Winter wheat\n obs={nobs["Winter wheat"]}',
                     'Spring barley': f'Spring barley\n obs={nobs["Spring barley"]}'}
    current_labels = [tick.get_text() for tick in s.get_xticklabels()]
    new_labels = [label_mapping.get(label, label) for label in current_labels]
    s.set_xticklabels(new_labels)

    s.set_xlabel('')
    s.set_ylabel('Yield [t/ha]')
    [s.axvline(x + .5, color='k') for x in s.get_xticks()]
    s.legend(ncol=6, title=None, bbox_to_anchor=(0.94, -0.15))
    plt.tight_layout()
    # plt.show()
    fig_path = 'Figures/Paper'
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(f'{fig_path}/Fig_2.png', dpi=300)

def plot_loocv(path, xgb=False):
    crops = ['maize', 'winter_wheat', 'spring_barley']
    fontsize = 22

    fig = plt.figure(figsize=(15, 10))
    outer = gridspec.GridSpec(2, 1, wspace=0.03, height_ratios=[0.95, 0.05], hspace=0.02)
    # outer = gridspec.GridSpec(1, len(crops), wspace=0.03, width_ratios=[0.3, 0.3, 0.3])

    for i, crop in enumerate(crops):
        file = pd.read_csv(f'{path}/{crop}.csv')
        file_xgb = pd.read_csv(f'{path}/{crop}_xgb.csv').iloc[:, 3:]
        file_xgb.columns = ['xgb_all_ua', 'xgb_eo_ua', 'xgb_met_ua', 'xgb_all_all', 'xgb_eo_all', 'xgb_met_all']
        a = pd.concat([file, file_xgb], axis=1)

        years = np.unique(a.year)

        for j, error in enumerate([True, False]):
            if xgb:
                res = pd.DataFrame(index=years, columns=['XGB_ua_eomet', 'XGB_all_eomet', 'ANN_tl_eomet', 'ANN_tl_eo', 'ANN_tl_met', 'ANN_all_eomet', 'crop'])
            else:
                res = pd.DataFrame(index=years, columns=['ANN_tl_eomet', 'ANN_tl_eo', 'ANN_tl_met', 'ANN_all_eomet', 'crop'])

            for year in years:
                b = a.loc[a.year == year]

                if error:
                    if xgb:
                        res.loc[year, 'XGB_ua_eomet'] = mean_absolute_percentage_error(b.observed_yield, b.xgb_all_ua)*100
                        res.loc[year, 'XGB_all_eomet'] = mean_absolute_percentage_error(b.observed_yield, b.xgb_all_all)*100
                    res.loc[year, 'ANN_tl_eomet'] = mean_absolute_percentage_error(b.observed_yield, b.predicted_all_transfer_learning)*100
                    res.loc[year, 'ANN_all_eomet'] = mean_absolute_percentage_error(b.observed_yield, b.predicted_all_direct_training_all_countries)*100
                    res.loc[year, 'ANN_tl_eo'] = mean_absolute_percentage_error(b.observed_yield, b.predicted_eo_transfer_learning)*100
                    res.loc[year, 'ANN_tl_met'] = mean_absolute_percentage_error(b.observed_yield, b.predicted_era5_transfer_learning)*100

                else:
                    if xgb:
                        res.loc[year, 'XGB_ua_eomet'] = r2_score(b.observed_yield, b.xgb_all_ua)
                        res.loc[year, 'XGB_all_eomet'] = r2_score(b.observed_yield, b.xgb_all_all)
                    res.loc[year, 'ANN_tl_eomet'] = r2_score(b.observed_yield, b.predicted_all_transfer_learning)
                    res.loc[year, 'ANN_all_eomet'] = r2_score(b.observed_yield, b.predicted_all_direct_training_all_countries)
                    res.loc[year, 'ANN_tl_eo'] = r2_score(b.observed_yield, b.predicted_eo_transfer_learning)
                    res.loc[year, 'ANN_tl_met'] = r2_score(b.observed_yield, b.predicted_era5_transfer_learning)

            res.crop = [crop.replace('_', ' ').capitalize()] * res.shape[0]
            if error:
                if i == 0:
                    all_err = res
                else:
                    all_err = pd.concat([all_err, res])
            else:
                if i == 0:
                    all_cor = res
                else:
                    all_cor = pd.concat([all_cor, res])
        cor_2022 = all_cor[all_cor.index == 2022]
        err_2022 = all_err[all_err.index == 2022]

    col1 = '#8c510a'
    col2 = '#bf812d'
    col3 = '#c7eae5'
    col4 = '#80cdc1'
    col5 = '#35978f'
    col6 = '#01665e'

    all_cor = pd.melt(all_cor, id_vars=['crop'])
    all_err = pd.melt(all_err, id_vars=['crop'])

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.1, height_ratios=[0.5, 0.5])
    ax_box_cor = plt.Subplot(fig, inner[1])
    ax_box_err = plt.Subplot(fig, inner[0])
    if xgb:
        col_palette = [col1, col2, col3, col4, col5, col6]
    else:
        col_palette = [col3, col4, col5, col2]
    col_cross = ['#FF0000'] * 6

    seaborn.boxplot(data=all_cor, x='crop', y='value', hue='variable', zorder=2, ax=ax_box_cor, palette=col_palette)
    seaborn.boxplot(data=all_err, x='crop', y='value', hue='variable', zorder=2, ax=ax_box_err, palette=col_palette)

    highlight_cor = pd.melt(cor_2022, id_vars=["crop"], value_vars=np.unique(all_cor.variable),
                           var_name='variable', value_name='value')
    highlight_err = pd.melt(err_2022, id_vars=["crop"], value_vars=np.unique(all_cor.variable),
                            var_name='variable', value_name='value')

    seaborn.stripplot(x='crop', y='value', hue='variable', data=highlight_cor, palette=col_cross,
                  dodge=True, marker='x', size=10, linewidth=2, color="red", ax=ax_box_cor, jitter=False)
    seaborn.stripplot(x='crop', y='value', hue='variable', data=highlight_err, palette=col_cross,
                      dodge=True, marker='x', size=10, linewidth=2, color="red", ax=ax_box_err, jitter=False)

    ax_box_cor.tick_params(axis='both', labelsize=fontsize-4)
    ax_box_err.tick_params(axis='both', labelsize=fontsize-4)
    for ax in [ax_box_cor, ax_box_err]:
        # ax.set_ylim(0, 1)
        # ax.set_yticklabels([])
        if ax == ax_box_cor:
            ax.set_yticks(np.arange(-1, 1.1, 0.1), minor=True)
        else:
            ax.set_yticks(np.arange(0, 31, 1), minor=True)

        ax.grid(axis='y', linestyle='--', color='gray', alpha=1)
        ax.grid(axis='y', which='minor', linestyle=':', color='gray', alpha=1, linewidth=0.5)
        ax.set_xlabel('')
        ax.legend().set_title('')
        plt.setp(ax.get_legend().get_texts(), fontsize=fontsize - 4)
        if ax == ax_box_cor:
            ax.set_ylabel('$\mathregular{R^{2}}$', fontsize=fontsize)
            ax.set_ylim(-1, 1)
        else:
            ax.set_xticklabels([])
            ax.set_ylabel('MAPE [%]', fontsize=fontsize)
            ax.set_ylim(0, 30)
        ha, le = ax.get_legend_handles_labels()
        ax.legend([], [], frameon=False)

        fig.add_subplot(ax)

    ax2 = plt.Subplot(fig, outer[1,0])
    ax2.axis('off')
    if xgb:
        le[6] = "2022 performance"
        ax2.legend(ha[:7], le[:7], ncol=4, fontsize=fontsize - 4, bbox_to_anchor=(0.95, -0.1))
    else:
        le[4] = "2022 performance"
        ax2.legend(ha[:5], le[:5], ncol=5, fontsize=fontsize - 4, bbox_to_anchor=(1.02, -0.1))

    fig.add_subplot(ax2)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.15)
    # plt.show()

    fig_path = 'Figures/Paper'
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(f'{fig_path}/Fig_4.png', dpi=300)

def plot_fi(path):
    # Read your CSV files (adjust file names)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    bar_width = 0.35
    base_fontsize = 10  # or any preferred base size

    plt.rcParams['font.size'] = base_fontsize * 1.5


    for idx, crop in enumerate(['maize', 'winter_wheat', 'spring_barley']):
        for v, var in enumerate(['all', 'eo', 'era5']):
            df = pd.read_csv(f'{path}/{crop}_feature_importance_{var}.csv')

            # Aggregate stats per feature
            agg = df.groupby("feature").agg({
                "base_importance_mean": ["mean", "std"],
                "finetuned_importance_mean": ["mean", "std"]
            })
            agg.columns = ['base_mean', 'base_std', 'finetuned_mean', 'finetuned_std']
            agg = agg.reset_index()

            # Rank by mean finetuned importance and select top 10
            agg = agg.sort_values("finetuned_mean", ascending=False).head(7)
            selected_features = agg["feature"].tolist()
            selected_features = [f.replace("VODCA_CXKu", "vod") for f in selected_features]

            # Normalize base mean and finetuned mean to sum to 1
            agg['base_mean'] = agg['base_mean'] / agg['base_mean'].sum()
            agg['finetuned_mean'] = agg['finetuned_mean'] / agg['finetuned_mean'].sum()

            # Normalize std dev proportionally to the same scale
            agg['base_std'] = agg['base_std'] / agg['base_mean'].sum()
            agg['finetuned_std'] = agg['finetuned_std'] / agg['finetuned_mean'].sum()

            x = np.arange(len(selected_features))
            # Top subplot: Base vs. Finetuned means
            ax1 = axes[v, idx]
            ax1.bar(x - bar_width / 2, agg['base_mean'], width=bar_width, yerr=agg['base_std'],
                    label="Base", alpha=1, color='#0057B7')
            ax1.bar(x + bar_width / 2, agg['finetuned_mean'], width=bar_width, yerr=agg['finetuned_std'],
                    label="Finetuned", alpha=1, color='#FFD700')
            if v==0:
                ax1.set_title(f"{crop.replace('_', ' ').capitalize()}")
            if idx == 0:
                if v==0:
                    ax1.set_ylabel("ANN_tl_eomet\nFeature Importance", fontsize=base_fontsize * 1.5)
                elif v == 1:
                    ax1.set_ylabel("ANN_tl_eo\nFeature Importance", fontsize=base_fontsize * 1.5)
                else:
                    ax1.set_ylabel("ANN_tl_met\nFeature Importance", fontsize=base_fontsize * 1.5)
            ax1.legend()
            ax1.set_xticks(x)
            ax1.set_xticklabels(selected_features, rotation=45, ha='center')
            ax1.tick_params(axis='x', labelsize=base_fontsize * 1.5)
            ax1.tick_params(axis='y', labelsize=base_fontsize * 1.5)

        # Bottom subplot: Difference
        # ax2 = axes[1, idx]
        #
        # ax2.bar(x, agg['finetuned_mean']-agg['base_mean'], width=0.6, color='gray', alpha=0.7)
        # if idx == 0:
        #     ax2.set_ylabel("Finetuned - Base Importance", fontsize=base_fontsize * 1.5)
        # ax2.set_xticks(x)
        # ax2.set_xticklabels(selected_features, rotation=45, ha='right')
        # # ax2.set_title(f"Finetuned - Base")
        # ax2.tick_params(axis='x', labelsize=base_fontsize * 1.5)
        # ax2.tick_params(axis='y', labelsize=base_fontsize * 1.5)

    plt.tight_layout()
    # plt.show()

    fig_path = 'Figures/Paper'
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(f'{fig_path}/Fig_5_all.png', dpi=300)

def plot_ua(path):
    """
    uses data from https://www.crisisgroup.org/content/ukraine-war-map-tracking-frontlines
    :return: paper_plot
    """
    war_map = pd.read_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\war_map.csv').iloc[:,1:]
    war_map.War = [a.replace('No', 'Unoccupied') for a in war_map.War]
    war_map_d = {a: b for a,b in zip(war_map.Region_ID, war_map.War)}
    fig = plt.figure(figsize=(12, 9))
    seaborn.set(font_scale=1.7)
    seaborn.set_style('white')
    seaborn.color_palette('Spectral')
    for pt, crop in enumerate(['maize', 'winter_wheat', 'spring_barley']):
        file = pd.read_csv(f'{path}/{crop}.csv')
        file.loc[:, 'country'] = [i[:2] for i in file.region]
        file = file.iloc[np.where(file.country == 'UA')[0], :]
        file.loc[:, 'war'] = [war_map_d[a] for a in file.loc[:, 'region']]
        file = file.iloc[np.where(file.year == 2022)[0], :]

        crop_name = crop.replace('_', ' ').capitalize()

        outer = gridspec.GridSpec(2, 3, wspace=0.16, hspace=0.1)
        ax_eo = plt.Subplot(fig, outer[0, pt])
        ax = plt.Subplot(fig, outer[1, pt])
        preds = file.predicted_era5_transfer_learning
        preds_eo = file.predicted_eo_transfer_learning
        obs = file.observed_yield
        if pt == 1:
            s = seaborn.scatterplot(x=obs, y=preds, hue=file.war, ax=ax,
                                    palette=dict(Unoccupied="#4b4453", Regained="#ff8066", Occupied="#e41a1c"), s=80)
            seaborn.move_legend(s, loc='lower center', bbox_to_anchor=(.5, -.4), ncol=6, title=None)
        else:
            s = seaborn.scatterplot(x=obs, y=preds, hue=file.war, ax=ax, legend=False,
                                    palette=dict(Unoccupied="#4b4453", Regained="#ff8066", Occupied="#e41a1c"), s=80)
        s_eo = seaborn.scatterplot(x=obs, y=preds_eo, hue=file.war, ax=ax_eo, legend=False,
                                    palette=dict(Unoccupied="#4b4453", Regained="#ff8066", Occupied="#e41a1c"), s=80)

        s.set_xlabel('Observed yield [t/ha]')
        s_eo.set_xlabel('')
        s_eo.set_title(crop_name)
        text = '$\mathregular{R^{2}}$=' + str(np.round(r2_score(obs, preds), 2))
        s.text(0.01, 0.98, text, ha='left', va='top', transform=s.transAxes)

        text = '$\mathregular{R^{2}}$=' + str(np.round(r2_score(obs, preds_eo), 2))
        s_eo.text(0.01, 0.98, text, ha='left', va='top', transform=s_eo.transAxes)

        if pt == 0:
            s.set_ylabel("ANN_tl_met\nForecasted yield [t/ha]")
            s_eo.set_ylabel("ANN_tl_eo\nForecasted yield [t/ha]")
        else:
            # ax.set_yticks([])
            s.set_ylabel("")
            s_eo.set_ylabel("")
        s_eo.set_xticks([])
        s_eo.set_xticklabels([])
        # Draw a line of x=y
        min0, min1 = np.min(preds), np.min(obs)
        max0, max1 = np.max(preds), np.max(obs)
        # if crop=='winter_wheat': min0, min1, max0, max1 = 2,2,6,6
        lims = [min(min0, min1)-0.1, max(max0, max1)+0.1]
        s.plot(lims, lims, '--', color='#999999')
        s_eo.plot(lims, lims, '--', color='#999999')
        s.set_ylim(lims)
        s.set_xlim(lims)
        s_eo.set_ylim(lims)
        s_eo.set_xlim(lims)

        fig.add_subplot(ax)
        fig.add_subplot(ax_eo)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.15)
    # plt.show()
    fig_path = 'Figures/Paper'
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(f'{fig_path}/Fig_6_eo.png', dpi=300)

def ua_line(path, old=False):
    """
    :return: paper_plot
    """
    war_map = pd.read_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\war_map.csv').iloc[:, 1:]
    crops = ['maize', 'winter_wheat', 'spring_barley']
    fontsize=20
    war_map_d = {a: b for a, b in zip(war_map.Region_ID, war_map.War)}
    war_zones = np.unique(war_map.iloc[np.where(war_map.War=='Occupied')[0],0])
    fig = plt.figure(figsize=(12, 10))
    seaborn.set(font_scale=1.7)
    seaborn.set_style('white')
    seaborn.color_palette('Spectral')
    outer = gridspec.GridSpec(len(crops)+1, 1, height_ratios=[0.3, 0.3, 0.3, 0.05])
    for c, crop in enumerate(crops):
        inner = gridspec.GridSpecFromSubplotSpec(1, len(war_zones), subplot_spec=outer[c])
        file = pd.read_csv(f'{path}/{crop}.csv')
        file.loc[:, 'country'] = [i[:2] for i in file.region]
        file = file.iloc[np.where(file.country == 'UA')[0], :]
        file.loc[:, 'war'] = [war_map_d[a] for a in file.region]

        errors_eo = {'year': [],
                  'errors': []}
        errors_met = {'year': [],
                     'errors': []}

        for year in np.unique(file.year):
            errors_eo['year'].append(year)
            errors_met['year'].append(year)
            subfile = file.iloc[np.where(file.year==year)[0],:]
            errors_eo['errors'].append(root_mean_squared_error(subfile.observed_yield, subfile.predicted_eo_transfer_learning))
            errors_met['errors'].append(
                root_mean_squared_error(subfile.observed_yield, subfile.predicted_era5_transfer_learning))
        errors_df_eo = pd.DataFrame(errors_eo)
        errors_df_met = pd.DataFrame(errors_met)

        crop_name = crop.replace('_', ' ').capitalize()

        for pt, war_zone in enumerate(war_zones):
            this_file = file.iloc[np.where(file.region==war_zone)[0],:]
            preds_eo = this_file.predicted_eo_transfer_learning
            preds_met = this_file.predicted_era5_transfer_learning
            obs = this_file.observed_yield
            ax = plt.Subplot(fig, inner[pt])
            time = this_file.year

            r2_eo = np.round(r2_score(preds_eo, obs), 2)
            r2_met = np.round(r2_score(preds_met, obs), decimals=2)

            print(crop, war_zone, r2_eo, r2_met)

            ers_eo = [errors_df_eo.iloc[np.where(errors_df_eo.year==t)[0],1].values[0] for t in time]
            ers_met = [errors_df_met.iloc[np.where(errors_df_met.year == t)[0], 1].values[0] for t in time]
            ax.plot(time, preds_eo, label='ANN_tl_eo')
            ax.fill_between(time, preds_eo-ers_eo, preds_eo+ers_eo, alpha=0.3)
            ax.plot(time, preds_met, label='ANN_tl_met')
            ax.fill_between(time, preds_met - ers_met, preds_met + ers_met, alpha=0.3)
            ax.scatter(time, obs, marker='x', label='Observed yield', linewidths=2, color='black')

            if c<(len(crops)-1):
                ax.set_xticklabels([])
            else:
                ax.set_xticks(time[::5])
                ax.set_xticklabels(time[::5], rotation=45, fontsize=fontsize-2)

            if c==0:
                ax.set_title(war_zone, fontsize=fontsize+2)
                ax.set_ylim(1.5,11)
            elif c==1:
                ax.set_ylim(2.5, 6)
            else:
                ax.set_ylim(1, 5)
            if pt==0:
                ax.set_ylabel(f'{crop_name} \nyields [t/ha]', fontsize=fontsize)
            else:
                ax.set_yticklabels([])


            ha, le = ax.get_legend_handles_labels()

            fig.add_subplot(ax)

    ax2 = plt.Subplot(fig, outer[3])
    ax2.axis('off')
    ax2.legend(ha, le, ncol=3, fontsize=fontsize, bbox_to_anchor=(0.9, 0.05))
    fig.add_subplot(ax2)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.07)
    plt.show()
    fig_path = 'Figures/Paper'
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(f'{fig_path}/Fig_7_eo.png', dpi=300)

def plot_scatter_ua(path):
    war_map = pd.read_csv(r'M:\Projects\YIPEEO\07_data\Crop yield\Ukraine\regional\war_map.csv').iloc[:, 1:]
    war_map_d = {a: b for a, b in zip(war_map.Region_ID, war_map.War)}

    # fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=False)
    # fig.suptitle('Predictions and Observed Yield Per Year', fontsize=16)
    fs = 20
    crops = ['maize', 'winter_wheat', 'spring_barley']
    ttest_res = pd.DataFrame(columns=range(2007,2023), index=crops)
    print(ttest_res)
    for row_idx, war in enumerate([False, True]):
        for col_idx, crop in enumerate(crops):
            df = pd.read_csv(f'{path}/{crop}.csv')
            df.loc[:, 'war'] = [war_map_d[a] for a in df.region]
            if war:
                # df = df.iloc[np.where(df.war=='Occupied')[0], :]
                df = df.iloc[np.where(df.war != 'No')[0], :]

        #         for year in df.year.unique():
        #             df_year = df.iloc[np.where(df.year == year)[0], :]
        #             ttest_res.loc[crop, year] = np.round(ttest_ind(df_year['predicted_eo_transfer_learning'],
        #                                      df_year['predicted_era5_transfer_learning']).pvalue, 2)
        # ttest_res.to_csv('Results/SC2/NUTS2/ttest_eo_met.csv')

            # Compute yearly means and stds
            yearly_stats = df.groupby('year').agg({
                'predicted_eo_transfer_learning': ['mean', 'std'],
                'predicted_era5_transfer_learning': ['mean', 'std'],
                'observed_yield': ['mean', 'std']
            }).reset_index()

            yearly_stats.columns = ['year',
                                    'eo_mean', 'eo_std',
                                    'era5_mean', 'era5_std',
                                    'observed_mean', 'observed_std']

            ax = axes[row_idx, col_idx]

            # Plot predicted_eo_transfer_learning with error
            r2_eo = np.round(r2_score(yearly_stats['observed_mean'], yearly_stats['eo_mean']),2)
            r2_met = np.round(r2_score(yearly_stats['observed_mean'], yearly_stats['era5_mean']), 2)

            # label_eo = 'ANN_tl_eo $\mathregular{R^{2}}$=' + str(r2_eo)
            # label_met = 'ANN_tl_met $\mathregular{R^{2}}$=' + str(r2_met)
            label_eo = 'ANN_tl_eo'
            label_met = 'ANN_tl_met'
            ax.plot(yearly_stats['year'], yearly_stats['eo_mean'], label=label_eo)
            ax.fill_between(yearly_stats['year'],
                                 yearly_stats['eo_mean'] - yearly_stats['eo_std'],
                                 yearly_stats['eo_mean'] + yearly_stats['eo_std'],
                                 alpha=0.3)

            # Plot predicted_era5_transfer_learning with error
            ax.plot(yearly_stats['year'], yearly_stats['era5_mean'], label=label_met)
            ax.fill_between(yearly_stats['year'],
                                 yearly_stats['era5_mean'] - yearly_stats['era5_std'],
                                 yearly_stats['era5_mean'] + yearly_stats['era5_std'],
                                 alpha=0.3)

            if col_idx==0:
                if row_idx==0:
                    ax.set_ylabel('Entire Ukraine\nCrop yield [t/ha]', fontsize=fs)
                else:
                    ax.set_ylabel('Attacked areas\nCrop yield [t/ha]', fontsize=fs)

            # Barplot observed_yield
            # ax.bar(x='year', height='observed_mean', data=yearly_stats, color='gray', label='Observed mean', zorder=0)
            ax.errorbar(x='year', y='observed_mean', yerr='observed_std', data=yearly_stats, fmt="o", color="black",
                        zorder=1, label='Observed')

            if (col_idx == 0) and (row_idx==0):
                ax.legend(fontsize=fs)
            if row_idx==1:
                ax.set_xlabel('Year', fontsize=fs)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=fs)
            else:
                ax.set_title(crop.replace('_', ' ').capitalize(), fontsize=fs + 2)
                ax.set_xticklabels([])
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs)
    plt.tight_layout()
    # plt.show()

    fig_path = 'Figures/Paper'
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(f'{fig_path}/Fig_8.png', dpi=300)

def yearly_hist():
    # Lade die Daten
    maize = pd.read_csv("Data/SC2/nuts2/maize_M.csv", index_col=0)
    winter_wheat = pd.read_csv("Data/SC2/nuts2/winter_wheat_M.csv", index_col=0)
    spring_barley = pd.read_csv("Data/SC2/nuts2/spring_barley_M.csv", index_col=0)
    seaborn.set(font_scale=1.5)
    seaborn.set_style('whitegrid')
    # Filtere Felder mit 'UA'
    maize = maize[maize['field_id'].str.startswith('UA')]
    winter_wheat = winter_wheat[winter_wheat['field_id'].str.startswith('UA')]
    spring_barley = spring_barley[spring_barley['field_id'].str.startswith('UA')]

    # Crop-Kategorie hinzufügen
    maize['crop'] = 'Maize'
    winter_wheat['crop'] = 'Winter Wheat'
    spring_barley['crop'] = 'Spring Barley'

    # Kombiniere alles
    df = pd.concat([maize, winter_wheat, spring_barley])

    # Boxplot
    plt.figure(figsize=(12, 6))
    seaborn.boxplot(x='c_year', y='yield', hue='crop', data=df)

    # Extrahiere die gewünschten Felder
    wanted_fields = ['UA14', 'UA23', 'UA44', 'UA65']
    highlight_df = df[df['field_id'].isin(wanted_fields)]

    # Definiere Farben für jede Region
    region_colors = {
        'UA14': 'red',
        'UA23': 'blue',
        'UA44': 'green',
        'UA65': 'purple'
    }

    # Zeichne die Symbole auf dem Boxplot
    for field_id in wanted_fields:
        field_data = highlight_df[highlight_df['field_id'] == field_id]
        for idx, row in field_data.iterrows():
            x_pos = list(df['c_year'].unique()).index(row['c_year'])
            crop_offsets = {'Maize': -0.25, 'Winter Wheat': 0, 'Spring Barley': 0.25}
            x_pos = x_pos + crop_offsets[row['crop']]
            plt.scatter(x_pos, row['yield'], marker='x',
                        color=region_colors[field_id], s=100, zorder=10, label=field_id)

    plt.xlabel("Year")
    plt.ylabel("Yield [t/ha]")

    # Separate crop and region handles/labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Crop types are first 3, regions are the rest (unique ones)
    crop_handles = handles[:3]
    crop_labels = labels[:3]

    region_dict = dict(zip(labels[3:], handles[3:]))
    region_handles = list(region_dict.values())
    region_labels = list(region_dict.keys())

    # Combine with crops first, then regions
    all_handles = region_handles + crop_handles
    all_labels = region_labels + crop_labels

    plt.legend(all_handles, all_labels, ncol=2)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'Figures/Paper/Fig_3.png', dpi=300)

if __name__ == '__main__':
    # Fig. 1 Köppen-Geiger
    # overview_map_ua()

    # Fig. 2 Crop histogram
    # plot_crop()

    # Fig. 3
    # yearly_hist()

    # Fig 4ff based on:
    path = "Results/SC2/nuts2/ann4"
    # run_ann_ua(results_path=path)
    # run_xgb_ua(path)
    # for crop in ['winter_wheat', 'spring_barley', 'maize']: run_fi(crop, path=path)

    # Fig. 4
    # plot_loocv(path, xgb=False)

    # Fig. 5
    plot_fi(path)

    # Fig. 6 Scatterplot Ukraine 2022
    # plot_ua(path)

    # Fig. 7 Warzones leave-one-YEAR-out cv
    # ua_line(path)

    # Fig 8
    # plot_scatter_ua(path)



    print(f'calculation stopped and took {datetime.now() - start_pro}')

