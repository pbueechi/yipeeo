import pandas as pd
import numpy as np
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
# import seaborn as sns
# from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
import warnings
import os
import geopandas as gpd
import matplotlib.gridspec as gridspec
# import matplotlib.patches as mpatches
# import os
# from typing import List, Dict, Tuple, Optional
# from datetime import datetime
# from XGB_pred import XGBYieldPredictionPipeline
# import geopandas as gpd

warnings.filterwarnings('ignore')

def plot_spatially():
    path_out_t = 'Results/Validation/dl/202510_revision'
    basepath = os.path.join(path_out_t, 'run3_loocv')

    fields = gpd.read_file(r'M:\Projects\YIPEEO\07_data\Crop yield\Database\field_scale_050524.shp')
    fields = fields[fields.farm_code == 'rost']
    fields = fields.drop_duplicates(subset='field_id')

    # Define crops
    crops = ['maize', 'winter_wheat', 'spring_barley']
    crop_labels = ['Maize', 'Winter Wheat', 'Spring Barley']

    # Create figure with gridspec for better control
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.05], hspace=0.35, wspace=0.3)

    # Create axes for plots
    axes = []
    for i in range(2):
        row_axes = []
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes.append(row_axes)

    # Create axes for colorbars
    cbar_axes = []
    for j in range(3):
        cbar_ax = fig.add_subplot(gs[2, j])
        cbar_axes.append(cbar_ax)

    # Iterate through crops (columns)
    for j, (crop, crop_label) in enumerate(zip(crops, crop_labels)):
        # Load the CSV file for this crop
        csv_file = f"{basepath}/{crop}_values_1_1.csv"
        df = pd.read_csv(csv_file)

        # Filter for year 2022
        df = df[df.year == 2022]

        # Merge the dataframe with the shapefile based on field_id
        fields_merged = fields.merge(df[['field_id', 'yield_obs', 'yield_forecast']],
                                     on='field_id', how='left')

        # Determine common color scale for this crop (across obs and forecast)
        vmin = min(fields_merged['yield_obs'].min(), fields_merged['yield_forecast'].min())
        vmax = max(fields_merged['yield_obs'].max(), fields_merged['yield_forecast'].max())

        # Plot yield_obs in the first row (without legend)
        ax_obs = axes[0][j]
        fields_merged.plot(column='yield_obs',
                           ax=ax_obs,
                           legend=False,
                           cmap='YlGn',
                           edgecolor='black',
                           linewidth=0.5,
                           vmin=vmin,
                           vmax=vmax)
        ax_obs.set_title(f'{crop_label} - Observed Yield', fontsize=11, fontweight='bold')
        ax_obs.set_xlabel('Longitude', fontsize=9)
        if j == 0:
            ax_obs.set_ylabel('Latitude', fontsize=9)
        ax_obs.tick_params(axis='both', labelsize=8)
        # Format x-axis (longitude) to 2 decimal places
        ax_obs.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_obs.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

        # Plot yield_forecast in the second row
        ax_forecast = axes[1][j]
        im = fields_merged.plot(column='yield_forecast',
                                ax=ax_forecast,
                                legend=False,
                                cmap='YlGn',
                                edgecolor='black',
                                linewidth=0.5,
                                vmin=vmin,
                                vmax=vmax)
        ax_forecast.set_title(f'{crop_label} - Forecasted Yield', fontsize=11, fontweight='bold')
        ax_forecast.set_xlabel('Longitude', fontsize=9)
        if j==0:
            ax_forecast.set_ylabel('Latitude', fontsize=9)
        ax_forecast.tick_params(axis='both', labelsize=8)
        # Format x-axis (longitude) to 2 decimal places
        ax_forecast.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_forecast.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

        # Add colorbar in the dedicated colorbar row
        cbar = plt.colorbar(im.collections[0], cax=cbar_axes[j], orientation='horizontal')
        cbar.set_label('Yield (t/ha)', fontsize=9)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
    # Save the figure
    output_file = f"{basepath}/crop_yield_spatial_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Display the plot
    plt.show()


# Example usage
if __name__ == "__main__":
    plot_spatially()
    pass