import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class LOYOResultsAnalyzer:
    """
    Analyzer for Leave-One-Year-Out Cross-Validation Results
    """

    def __init__(self, results_files: dict):
        """
        Initialize with dictionary of crop names and their result file paths

        Parameters:
        - results_files: dict like {'maize': 'path/to/maize_results.csv', ...}
        """
        self.results_files = results_files
        self.data = {}
        self.r2_data = None

        # Define the 9 model setups
        self.feature_types = ['all', 'eo', 'era5']
        self.training_modes = ['transfer_learning', 'direct_training', 'direct_training_all_countries']

        # Load and process data
        self.load_data()
        self.calculate_r2_scores()

    def load_data(self):
        """Load CSV files for each crop"""
        for crop, file_path in self.results_files.items():
            try:
                df = pd.read_csv(file_path)
                self.data[crop] = df
                print(f"Loaded {crop}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  Years: {sorted(df['year'].unique())}")
                print(f"  Regions: {sorted(df['region'].unique())}")
            except Exception as e:
                print(f"Error loading {crop} data from {file_path}: {e}")

    def calculate_r2_scores(self):
        """Calculate R² scores for each year and model setup"""
        from sklearn.metrics import r2_score

        r2_results = []

        for crop in self.data.keys():
            df = self.data[crop]

            # Get unique years
            years = sorted(df['year'].unique())

            for year in years:
                year_data = df[df['year'] == year]
                observed = year_data['observed_yield'].values

                # Skip if no observed data
                if len(observed) == 0 or np.all(np.isnan(observed)):
                    continue

                for feature_type in self.feature_types:
                    for training_mode in self.training_modes:
                        pred_col = f'predicted_{feature_type}_{training_mode}'

                        if pred_col in df.columns:
                            predicted = year_data[pred_col].values

                            # Calculate R² only if we have valid predictions
                            if len(predicted) > 0 and not np.all(np.isnan(predicted)):
                                # Remove NaN pairs
                                valid_mask = ~(np.isnan(observed) | np.isnan(predicted))
                                if np.sum(valid_mask) > 1:  # Need at least 2 points
                                    obs_valid = observed[valid_mask]
                                    pred_valid = predicted[valid_mask]

                                    r2 = r2_score(obs_valid, pred_valid)

                                    r2_results.append({
                                        'crop': crop,
                                        'year': year,
                                        'feature_type': feature_type,
                                        'training_mode': training_mode,
                                        'model_setup': f'{feature_type}_{training_mode}',
                                        'r2': r2,
                                        'n_samples': len(obs_valid)
                                    })

        self.r2_data = pd.DataFrame(r2_results)
        print(f"\nCalculated R² scores: {len(self.r2_data)} records")
        print(f"Crops: {self.r2_data['crop'].unique()}")
        print(f"Model setups: {self.r2_data['model_setup'].unique()}")

    def plot_boxplots_by_crop(self, figsize=(20, 15)):
        """
        Create boxplots of R² values for each crop and model setup, grouped by feature type
        """
        if self.r2_data is None or len(self.r2_data) == 0:
            print("No R² data available for plotting")
            return

        # Set up the plot
        fig, axes = plt.subplots(len(self.data), 1, figsize=figsize, sharex=True)
        if len(self.data) == 1:
            axes = [axes]

        # Define colors for different training modes
        training_colors = {
            'transfer_learning': '#1f77b4',
            'direct_training': '#ff7f0e',
            'direct_training_all_countries': '#2ca02c'
        }

        # Feature type labels for better readability
        feature_labels = {
            'all': 'ALL Features\n(EO + ERA5)',
            'eo': 'EO Features\n(Earth Observation)',
            'era5': 'ERA5 Features\n(Meteorological)'
        }

        # Training mode labels
        training_labels = {
            'transfer_learning': 'Transfer',
            'direct_training': 'Direct\n(No UA)',
            'direct_training_all_countries': 'Direct\n(All Countries)'
        }

        for i, crop in enumerate(sorted(self.data.keys())):
            ax = axes[i]

            # Filter data for this crop
            crop_data = self.r2_data[self.r2_data['crop'] == crop]

            if len(crop_data) == 0:
                ax.text(0.5, 0.5, f'No data available for {crop}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{crop.replace("_", " ").title()}')
                continue

            # Prepare data for grouped boxplot
            positions = []
            boxplot_data = []
            colors = []
            labels = []

            pos = 1
            group_positions = []
            group_labels = []

            for feature_type in self.feature_types:
                group_start = pos

                for training_mode in self.training_modes:
                    setup_name = f'{feature_type}_{training_mode}'
                    setup_data = crop_data[crop_data['model_setup'] == setup_name]['r2'].values

                    boxplot_data.append(setup_data if len(setup_data) > 0 else [])
                    positions.append(pos)
                    colors.append(training_colors[training_mode])
                    labels.append(training_labels[training_mode])
                    pos += 1

                # Add group separator
                group_positions.append((group_start + pos - 1) / 2 - 0.5)
                group_labels.append(feature_labels[feature_type])
                pos += 1  # Add space between groups

            # Create boxplot
            bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=0.6)

            # Color the boxes by training mode
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Customize the plot
            ax.set_title(f'{crop.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('R² Score', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Set x-ticks for groups
            ax.set_xticks(group_positions)
            ax.set_xticklabels(group_labels, fontsize=11)

            # Add training mode labels below each box
            for pos, label in zip(positions, labels):
                ax.text(pos, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        label, ha='center', va='top', fontsize=9, rotation=0)

            # Add vertical lines to separate groups
            for j in range(len(self.feature_types) - 1):
                sep_pos = positions[3 + j * 4] + 0.5
                ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)

            # Add horizontal line at R² = 0
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

            # Add summary statistics
            n_years = len(crop_data['year'].unique())
            ax.text(0.02, 0.98, f'Years: {n_years}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Create legend for training modes
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7,
                                         label=training_labels[tm])
                           for tm, color in training_colors.items()]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

        # Set common x-label
        axes[-1].set_xlabel('Feature Type (Training Dataset)', fontsize=12)

        plt.suptitle('Leave-One-Year-Out Cross-Validation Results\nR² Scores by Crop and Feature Type',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_combined_boxplot(self, figsize=(18, 10)):
        """
        Create a single plot with all crops combined, grouped by feature type
        """
        if self.r2_data is None or len(self.r2_data) == 0:
            print("No R² data available for plotting")
            return

        plt.figure(figsize=figsize)

        # Define colors
        crops = sorted(self.data.keys())
        crop_colors = plt.cm.Set1(np.linspace(0, 1, len(crops)))
        crop_color_dict = dict(zip(crops, crop_colors))

        # Feature type labels
        feature_labels = {
            'all': 'ALL Features\n(EO + ERA5)',
            'eo': 'EO Features\n(Earth Observation)',
            'era5': 'ERA5 Features\n(Meteorological)'
        }

        # Training mode labels
        training_labels = {
            'transfer_learning': 'Transfer',
            'direct_training': 'Direct\n(No UA)',
            'direct_training_all_countries': 'Direct\n(All Countries)'
        }

        # Calculate positions for grouped boxplots
        positions = []
        boxplot_data = []
        colors = []

        pos = 1
        group_positions = []
        group_labels = []

        for feature_type in self.feature_types:
            group_start = pos

            for training_mode in self.training_modes:
                for crop in crops:
                    setup_name = f'{feature_type}_{training_mode}'
                    crop_setup_data = self.r2_data[
                        (self.r2_data['crop'] == crop) &
                        (self.r2_data['model_setup'] == setup_name)
                        ]['r2'].values

                    boxplot_data.append(crop_setup_data if len(crop_setup_data) > 0 else [])
                    positions.append(pos)
                    colors.append(crop_color_dict[crop])
                    pos += 1

                # Small gap between training modes within feature type
                pos += 0.5

            # Add group separator
            group_positions.append((group_start + pos - 1) / 2 - 0.25)
            group_labels.append(feature_labels[feature_type])
            pos += 1  # Larger gap between feature types

        # Create boxplot
        bp = plt.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=0.3)

        # Color the boxes by crop
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add training mode labels
        training_positions = []
        training_labels_list = []
        pos_idx = 0

        for feature_type in self.feature_types:
            for training_mode in self.training_modes:
                # Calculate center position for this training mode across all crops
                start_idx = pos_idx
                end_idx = pos_idx + len(crops) - 1
                center_pos = (positions[start_idx] + positions[end_idx]) / 2
                training_positions.append(center_pos)
                training_labels_list.append(training_labels[training_mode])
                pos_idx += len(crops)
                if training_mode != self.training_modes[-1]:  # Skip gap after last training mode
                    pos_idx += 1  # Account for the 0.5 gap
            pos_idx += 1  # Account for the larger gap between feature types

        # Customize the plot
        plt.title('Leave-One-Year-Out Cross-Validation Results\nR² Scores by Feature Type and Crop',
                  fontsize=16, fontweight='bold')
        plt.ylabel('R² Score', fontsize=12)
        plt.xlabel('Feature Type (Training Dataset)', fontsize=12)

        # Set x-ticks for feature groups
        plt.xticks(group_positions, group_labels, fontsize=11)

        # Add training mode labels
        ax = plt.gca()
        for pos, label in zip(training_positions, training_labels_list):
            ax.text(pos, ax.get_ylim()[0] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    label, ha='center', va='top', fontsize=10, fontweight='bold')

        # Add vertical lines to separate feature groups
        for i in range(len(self.feature_types) - 1):
            # Find separator position between feature types
            sep_pos = group_positions[i] + (group_positions[i + 1] - group_positions[i]) / 2
            plt.axvline(x=sep_pos, color='gray', linestyle='-', alpha=0.3, linewidth=2)

        # Add legend for crops
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7,
                                         label=crop.replace('_', ' ').title())
                           for crop, color in crop_color_dict.items()]
        plt.legend(handles=legend_elements, loc='upper right')

        # Add grid and reference line
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def plot_heatmap_summary(self, figsize=(12, 8)):
        """
        Create a heatmap showing mean R² scores
        """
        if self.r2_data is None or len(self.r2_data) == 0:
            print("No R² data available for plotting")
            return

        # Calculate mean R² for each crop and setup
        summary = self.r2_data.groupby(['crop', 'model_setup'])['r2'].agg(['mean', 'std', 'count']).reset_index()

        # Pivot for heatmap
        heatmap_data = summary.pivot(index='crop', columns='model_setup', values='mean')

        # Reorder columns for better grouping
        ordered_cols = []
        for ft in self.feature_types:
            for tm in self.training_modes:
                col_name = f'{ft}_{tm}'
                if col_name in heatmap_data.columns:
                    ordered_cols.append(col_name)

        heatmap_data = heatmap_data[ordered_cols]

        # Create heatmap
        plt.figure(figsize=figsize)

        # Create custom labels
        col_labels = []
        for ft in self.feature_types:
            for tm in self.training_modes:
                col_name = f'{ft}_{tm}'
                if col_name in heatmap_data.columns:
                    label = f"{ft.upper()}\n{tm.replace('_', ' ').replace('training', 'train').title()}"
                    col_labels.append(label)

        row_labels = [crop.replace('_', ' ').title() for crop in heatmap_data.index]

        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlBu_r',
                    center=0,
                    xticklabels=col_labels,
                    yticklabels=row_labels,
                    cbar_kws={'label': 'Mean R² Score'})

        plt.title('Mean R² Scores by Crop and Model Setup', fontsize=14, fontweight='bold')
        plt.xlabel('Model Setup', fontsize=12)
        plt.ylabel('Crop', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def print_summary_statistics(self):
        """Print summary statistics"""
        if self.r2_data is None or len(self.r2_data) == 0:
            print("No R² data available")
            return

        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        # Overall statistics
        overall_stats = self.r2_data.groupby(['crop', 'model_setup'])['r2'].agg(['mean', 'std', 'count'])

        for crop in sorted(self.data.keys()):
            print(f"\n{crop.replace('_', ' ').title().upper()}:")
            print("-" * 40)
            crop_stats = overall_stats.loc[crop] if crop in overall_stats.index else pd.DataFrame()

            if len(crop_stats) > 0:
                # Sort by mean R²
                crop_stats_sorted = crop_stats.sort_values('mean', ascending=False)

                print(f"{'Setup':<35} {'Mean R²':<10} {'Std R²':<10} {'N Years':<8}")
                print("-" * 65)

                for setup, row in crop_stats_sorted.iterrows():
                    setup_name = setup.replace('_', ' ').title()
                    print(f"{setup_name:<35} {row['mean']:<10.3f} {row['std']:<10.3f} {int(row['count']):<8}")

                # Best setup
                best_setup = crop_stats_sorted.index[0]
                best_r2 = crop_stats_sorted.iloc[0]['mean']
                print(f"\nBest setup: {best_setup.replace('_', ' ').title()} (R² = {best_r2:.3f})")
            else:
                print("No data available")

        # Cross-crop comparison
        print(f"\n{'=' * 80}")
        print("CROSS-CROP COMPARISON")
        print("=" * 80)

        crop_means = self.r2_data.groupby('crop')['r2'].mean().sort_values(ascending=False)
        setup_means = self.r2_data.groupby('model_setup')['r2'].mean().sort_values(ascending=False)

        print("\nMean R² by Crop:")
        for crop, r2 in crop_means.items():
            print(f"  {crop.replace('_', ' ').title()}: {r2:.3f}")

        print("\nMean R² by Model Setup:")
        for setup, r2 in setup_means.items():
            print(f"  {setup.replace('_', ' ').title()}: {r2:.3f}")


# Example usage
if __name__ == "__main__":
    # Define your result files
    results_files = {
        'maize': 'Results/SC2/nuts2/ann1/maize.csv',
        'winter_wheat': 'Results/SC2/nuts2/ann1/winter_wheat.csv',
        'spring_barley': 'Results/SC2/nuts2/ann1/spring_barley.csv'
    }

    # Initialize analyzer
    analyzer = LOYOResultsAnalyzer(results_files)

    # Create plots
    print("Creating boxplots by crop...")
    analyzer.plot_boxplots_by_crop()

    # print("\nCreating combined boxplot...")
    # analyzer.plot_combined_boxplot()
    #
    # print("\nCreating heatmap summary...")
    # analyzer.plot_heatmap_summary()

    # Print summary statistics
    analyzer.print_summary_statistics()


