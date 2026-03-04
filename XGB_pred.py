import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
import warnings
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

class XGBYieldPredictionModel:
    """
    XGBoost model for yield prediction with leave-one-year-out cross-validation
    """

    def __init__(self, csv_path: str, yield_column: str = 'yield'):
        """
        Initialize the model with data

        Parameters:
        - csv_path: Path to the CSV file
        - yield_column: Name of the yield column
        """
        self.csv_path = csv_path
        self.yield_column = yield_column
        self.data = None
        self.feature_columns = []
        self.eo_features = []
        self.era5_features = []
        self.scaler_features = StandardScaler()
        self.results = {}

        # Define EO and ERA5 predictors
        self.eo_predictors = ['evi', 'sm', 'vodca']  # EO-based predictors
        self.era5_predictors = []  # Will be populated automatically

        # Load and prepare data
        self.load_data()
        self.prepare_features()

    def load_data(self):
        """Load and basic preprocessing of the data"""
        self.data = pd.read_csv(self.csv_path, index_col=0)
        self.data = self.data.drop(columns=[col for col in self.data.columns if col.startswith('lai')])
        self.data.loc[:, 'country'] = [a_col[:2] for a_col in self.data.field_id]
        # print(f"Data loaded: {self.data.shape}")
        # print(f"Columns: {list(self.data.columns)}")
        # print(f"Available years: {sorted(self.data['c_year'].unique())}")
        # print(f"Available regions: {sorted(self.data['country'].unique())}")

    def prepare_features(self):
        """Identify and prepare feature columns, separating EO and ERA5 features"""
        # Find feature columns (excluding target, year, and region)
        exclude_cols = [self.yield_column, 'field_id', 'country']
        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]

        # Identify predictor types and months
        predictors = set()
        for col in self.feature_columns:
            if any(suffix in col for suffix in ['LT1', 'LT2', 'LT3', 'LT4']):
                predictor = col.replace('LT1', '').replace('LT2', '').replace('LT3', '').replace('LT4', '')
                predictor = predictor.rstrip('_')
                predictors.add(predictor)
        predictors.add('c_year')
        # Separate EO and ERA5 features
        self.eo_features = []
        self.era5_features = []

        for col in self.feature_columns:
            is_eo = any(eo_pred in col.lower() for eo_pred in self.eo_predictors)
            if is_eo:
                self.eo_features.append(col)
            else:
                self.era5_features.append(col)

        # Update ERA5 predictors list
        era5_predictors = set()
        for col in self.era5_features:
            for suffix in ['LT1', 'LT2', 'LT3', 'LT4']:
                if suffix in col:
                    predictor = col.replace(suffix, '').rstrip('_')
                    era5_predictors.add(predictor)
        self.era5_predictors = list(era5_predictors)

        print(f"Identified predictors: {sorted(predictors)}")
        print(f"EO predictors: {self.eo_predictors}")
        print(f"ERA5 predictors: {self.era5_predictors}")
        print(
            f"Total features: {len(self.feature_columns)} (EO: {len(self.eo_features)}, ERA5: {len(self.era5_features)})")

    def prepare_data_for_training(self, train_data: pd.DataFrame,
                                  test_data: pd.DataFrame, feature_type: str = 'all') -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and scale data for training

        Parameters:
        - feature_type: 'all', 'eo', or 'era5' to select feature subset
        """
        # Select features based on type
        if feature_type == 'eo':
            selected_features = self.eo_features
        elif feature_type == 'era5':
            selected_features = self.era5_features
        else:  # 'all'
            selected_features = self.feature_columns

        if not selected_features:
            raise ValueError(f"No features found for type '{feature_type}'")

        # Extract features and targets
        X_train = train_data[selected_features].values
        y_train = train_data[self.yield_column].values
        X_test = test_data[selected_features].values
        y_test = test_data[self.yield_column].values

        # Handle missing values
        X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
        X_test = np.nan_to_num(X_test, nan=np.nanmean(X_train))

        # Scale features (XGBoost can work without scaling, but it often helps)
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_test_scaled = self.scaler_features.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def create_xgb_model(self, n_samples: int) -> xgb.XGBRegressor:
        """
        Create XGBoost regressor with parameters suitable for the dataset size

        Parameters:
        - n_samples: Number of training samples for parameter adjustment
        """
        # Adjust parameters based on sample size
        if n_samples < 100:
            params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 1
            }
        elif n_samples < 500:
            params = {
                'n_estimators': 200,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5
            }
        else:
            params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }

        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            n_jobs=1,  # Use single thread for parallel processing compatibility
            verbosity=0
        )

        return model

    def train_model_with_cv(self, X_train: np.ndarray, y_train: np.ndarray,
                            feature_type: str = 'all') -> xgb.XGBRegressor:
        """
        Train XGBoost model with cross-validation for hyperparameter tuning
        """
        # Create base model
        model = self.create_xgb_model(len(X_train))

        # For small datasets, skip hyperparameter tuning
        if len(X_train) < 50:
            model.fit(X_train, y_train)
            return model

        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.05, 0.1, 0.2]
        }

        # Use fewer parameter combinations for smaller datasets
        if len(X_train) < 200:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 4],
                'learning_rate': [0.1, 0.2]
            }

        # Use TimeSeriesSplit for cross-validation (appropriate for time series data)
        cv_splits = min(3, len(X_train) // 20)  # Ensure reasonable splits
        if cv_splits < 2:
            # If too few samples for CV, just fit the model
            model.fit(X_train, y_train)
            return model

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_

    def evaluate_model(self, model: xgb.XGBRegressor, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        """
        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_true': y_test,
            'y_pred': y_pred
        }


def cross_validation_worker(args):
    """
    Worker function for parallel cross-validation
    """
    test_year, csv_path, yield_column, feature_type, training_mode = args

    try:
        # Create model instance for this worker
        model_instance = XGBYieldPredictionModel(csv_path, yield_column)
        data = model_instance.data

        # Filter data based on training mode
        if training_mode == 'ua_only':
            # Training mode 1: Train and test on UA data only
            ua_data = data[data['country'] == 'UA']

            train_data = ua_data[~ua_data['c_year'].isin([test_year, 2022])]
            test_data = ua_data[ua_data['c_year'] == test_year]

            # Check if we have enough data
            if len(train_data) < 5 or len(test_data) < 1:
                return test_year, None

            training_setup = 'ua_only'

        elif training_mode == 'all_countries':
            # Training mode 2: Train on all countries, test on UA
            train_data = data[~data['c_year'].isin([test_year, 2022])]
            test_data = data[
                (data['country'] == 'UA') &
                (data['c_year'] == test_year)
                ]

            # Check if we have enough data
            if len(train_data) < 20 or len(test_data) < 1:
                return test_year, None

            training_setup = 'all_countries'

        # Prepare data
        X_train, X_test, y_train, y_test = model_instance.prepare_data_for_training(
            train_data, test_data, feature_type
        )

        # Train model
        model = model_instance.train_model_with_cv(X_train, y_train, feature_type)

        # Evaluate and get predictions
        results = model_instance.evaluate_model(model, X_test, y_test)
        results['test_year'] = test_year
        results['feature_type'] = feature_type
        results['training_mode'] = training_setup
        results['n_test_samples'] = len(test_data)
        results['n_train_samples'] = len(train_data)

        # Add region and year information for CSV export
        results['test_regions'] = test_data['field_id'].values
        results['test_years'] = test_data['c_year'].values

        return test_year, results

    except Exception as e:
        print(f"Error in year {test_year} with {feature_type} features and {training_mode}: {str(e)}")
        return test_year, None


class XGBYieldPredictionPipeline:
    """
    Main pipeline for XGBoost yield prediction with cross-validation
    """

    def __init__(self, csv_path: str, yield_column: str = 'yield'):
        self.csv_path = csv_path
        self.yield_column = yield_column
        self.data = pd.read_csv(self.csv_path, index_col=0)
        self.data = self.data.drop(columns=[col for col in self.data.columns if col.startswith('lai')])
        self.data.loc[:, 'country'] = [a_col[:2] for a_col in self.data.field_id]
        self.results = {}

    def run_leave_one_year_out_cv(self, feature_type: str = 'all', training_mode: str = 'ua_only',
                                  n_processes: Optional[int] = None) -> Dict:
        """
        Run leave-one-year-out cross-validation with parallel processing

        Parameters:
        - feature_type: 'all', 'eo', or 'era5' to select feature subset
        - training_mode: 'ua_only' or 'all_countries'
        - n_processes: Number of parallel processes (default: CPU count - 1)
        """
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)

        # Get unique years where UA has data
        ua_years = self.data[self.data['country'] == 'UA']['c_year'].unique()
        years_to_test = sorted(ua_years)

        print(f"Running LOYO CV for years: {years_to_test}")
        print(f"Using {n_processes} processes")
        print(f"Feature type: {feature_type}")
        print(f"Training mode: {training_mode}")

        # Prepare arguments for parallel processing
        args_list = [(year, self.csv_path, self.yield_column, feature_type, training_mode)
                     for year in years_to_test]

        # Run parallel cross-validation
        with Pool(n_processes) as pool:
            results = pool.map(cross_validation_worker, args_list)

        # Process results
        cv_results = {}
        for year, result in results:
            if result is not None:
                cv_results[year] = result

        return cv_results

    def run_comprehensive_analysis(self, n_processes: Optional[int] = None) -> Dict:
        """
        Run comprehensive analysis with all feature types and training modes

        Parameters:
        - n_processes: Number of parallel processes
        """
        feature_types = ['all']
        training_modes = ['ua_only', 'all_countries']

        comprehensive_results = {}

        for feature_type in feature_types:
            for training_mode in training_modes:
                print(f"\n{'=' * 60}")
                print(f"Running analysis: {feature_type} features, {training_mode}")
                print(f"{'=' * 60}")

                try:
                    cv_results = self.run_leave_one_year_out_cv(
                        feature_type=feature_type,
                        training_mode=training_mode,
                        n_processes=n_processes
                    )

                    if cv_results:
                        # Calculate summary statistics
                        rmse_values = [r['rmse'] for r in cv_results.values()]
                        r2_values = [r['r2'] for r in cv_results.values()]
                        mae_values = [r['mae'] for r in cv_results.values()]

                        summary = {
                            'mean_rmse': np.mean(rmse_values),
                            'std_rmse': np.std(rmse_values),
                            'mean_r2': np.mean(r2_values),
                            'std_r2': np.std(r2_values),
                            'mean_mae': np.mean(mae_values),
                            'std_mae': np.std(mae_values),
                            'detailed_results': cv_results,
                            'feature_type': feature_type,
                            'training_mode': training_mode
                        }

                        key = f"{feature_type}_{training_mode}"
                        comprehensive_results[key] = summary

                        print(f"Results: RMSE={summary['mean_rmse']:.3f}±{summary['std_rmse']:.3f}, "
                              f"R²={summary['mean_r2']:.3f}±{summary['std_r2']:.3f}")

                except Exception as e:
                    print(f"Error in {feature_type}_{training_mode}: {str(e)}")
                    continue

        return comprehensive_results

    def export_predictions_to_csv(self, comprehensive_results: Dict,
                                  output_filename: str = 'xgb_yield_predictions.csv'):
        """
        Export predictions to CSV with the requested format:
        year, region, observed_yield,
        predicted_all_ua_only, predicted_eo_ua_only, predicted_era5_ua_only,
        predicted_all_all_countries, predicted_eo_all_countries, predicted_era5_all_countries
        """
        print(f"Exporting predictions to {output_filename}...")

        # Collect all prediction data
        prediction_data = []

        # Get all unique year-region combinations from any available results
        all_year_region_combinations = set()
        for results in comprehensive_results.values():
            if 'detailed_results' in results:
                for year, year_results in results['detailed_results'].items():
                    if 'test_regions' in year_results and 'test_years' in year_results:
                        regions = year_results['test_regions']
                        years = year_results['test_years']
                        for region, year_val in zip(regions, years):
                            all_year_region_combinations.add((int(year_val), region))

        # Convert to sorted list
        all_combinations = sorted(list(all_year_region_combinations))

        # Create a dictionary to store predictions by setup
        predictions_dict = {}

        # Extract predictions for each setup
        for key, results in comprehensive_results.items():
            feature_type = results['feature_type']
            training_mode = results['training_mode']

            if 'detailed_results' in results:
                for year, year_results in results['detailed_results'].items():
                    if 'test_regions' in year_results and 'y_pred' in year_results and 'y_true' in year_results:
                        regions = year_results['test_regions']
                        years = year_results['test_years']
                        predictions = year_results['y_pred']
                        observations = year_results['y_true']

                        for i, (region, year_val) in enumerate(zip(regions, years)):
                            combo_key = (int(year_val), region)
                            if combo_key not in predictions_dict:
                                predictions_dict[combo_key] = {
                                    'year': int(year_val),
                                    'region': region,
                                    'observed_yield': observations[i]
                                }

                            # Store prediction based on feature type and training mode
                            pred_key = f"predicted_{feature_type}_{training_mode}"
                            predictions_dict[combo_key][pred_key] = predictions[i]

        # Convert to list of dictionaries for DataFrame
        for combo in all_combinations:
            if combo in predictions_dict:
                row = predictions_dict[combo].copy()

                # Ensure all columns exist with NaN if missing
                column_names = [
                    'predicted_all_ua_only',
                    'predicted_eo_ua_only',
                    'predicted_era5_ua_only',
                    'predicted_all_all_countries',
                    'predicted_eo_all_countries',
                    'predicted_era5_all_countries'
                ]

                for col in column_names:
                    if col not in row:
                        row[col] = np.nan

                prediction_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(prediction_data)

        # Reorder columns as requested
        desired_columns = [
            'year', 'region', 'observed_yield',
            'predicted_all_ua_only', 'predicted_eo_ua_only', 'predicted_era5_ua_only',
            'predicted_all_all_countries', 'predicted_eo_all_countries', 'predicted_era5_all_countries'
        ]

        # Select only existing columns
        existing_columns = [col for col in desired_columns if col in df.columns]
        df = df[existing_columns]

        # Sort by year and region
        df = df.sort_values(['year', 'region']).reset_index(drop=True)

        # Save to CSV
        df.to_csv(output_filename, index=False)

        print(f"Predictions exported to {output_filename}")
        print(f"Shape: {df.shape}")
        print(f"Years covered: {sorted(df['year'].unique())}")
        print(f"Regions covered: {sorted(df['region'].unique())}")
        print(f"\nColumns in output:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"  {col}: {non_null_count} non-null values")

        return df

    def plot_performance_comparison(self, comprehensive_results: Dict, figsize=(15, 10)):
        """
        Plot performance comparison across different setups
        """
        if not comprehensive_results:
            print("No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Prepare data for plotting
        setups = list(comprehensive_results.keys())
        setup_labels = [s.replace('_', ' ').title() for s in setups]
        mean_rmse = [comprehensive_results[s]['mean_rmse'] for s in setups]
        std_rmse = [comprehensive_results[s]['std_rmse'] for s in setups]
        mean_r2 = [comprehensive_results[s]['mean_r2'] for s in setups]
        std_r2 = [comprehensive_results[s]['std_r2'] for s in setups]

        # RMSE comparison
        axes[0, 0].bar(range(len(setups)), mean_rmse, yerr=std_rmse, capsize=5)
        axes[0, 0].set_title('RMSE by Model Setup')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_xticks(range(len(setups)))
        axes[0, 0].set_xticklabels(setup_labels, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)

        # R² comparison
        axes[0, 1].bar(range(len(setups)), mean_r2, yerr=std_r2, capsize=5)
        axes[0, 1].set_title('R² by Model Setup')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_xticks(range(len(setups)))
        axes[0, 1].set_xticklabels(setup_labels, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)

        # Performance by year
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for i, (setup, results) in enumerate(comprehensive_results.items()):
            if 'detailed_results' in results:
                years = list(results['detailed_results'].keys())
                rmse_by_year = [results['detailed_results'][year]['rmse'] for year in years]
                axes[1, 0].plot(years, rmse_by_year, marker='o',
                                label=setup.replace('_', ' ').title(),
                                color=colors[i % len(colors)])

        axes[1, 0].set_title('RMSE by Year and Setup')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # R² by year
        for i, (setup, results) in enumerate(comprehensive_results.items()):
            if 'detailed_results' in results:
                years = list(results['detailed_results'].keys())
                r2_by_year = [results['detailed_results'][year]['r2'] for year in years]
                axes[1, 1].plot(years, r2_by_year, marker='o',
                                label=setup.replace('_', ' ').title(),
                                color=colors[i % len(colors)])

        axes[1, 1].set_title('R² by Year and Setup')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(self, cv_results: Dict, setup_name: str = "", figsize=(10, 8)):
        """
        Plot predictions vs actual values
        """
        if not cv_results:
            print("No results to plot")
            return

        # Combine all years
        y_true_all = np.concatenate([r['y_true'] for r in cv_results.values()])
        y_pred_all = np.concatenate([r['y_pred'] for r in cv_results.values()])

        plt.figure(figsize=figsize)
        plt.scatter(y_true_all, y_pred_all, alpha=0.6)

        # Perfect prediction line
        min_val = min(y_true_all.min(), y_pred_all.min())
        max_val = max(y_true_all.max(), y_pred_all.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Calculate R²
        r2 = r2_score(y_true_all, y_pred_all)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title(f'Predictions vs Actual {setup_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def print_summary_statistics(self, comprehensive_results: Dict):
        """Print summary statistics"""
        if not comprehensive_results:
            print("No results available")
            return

        print("\n" + "=" * 80)
        print("XGBoost MODEL SUMMARY STATISTICS")
        print("=" * 80)

        # Sort results by performance (lowest RMSE first)
        sorted_results = sorted(comprehensive_results.items(),
                                key=lambda x: x[1]['mean_rmse'])

        print(f"{'Rank':<4} {'Setup':<35} {'RMSE':<12} {'R²':<12} {'MAE':<12}")
        print("-" * 80)

        for i, (key, results) in enumerate(sorted_results, 1):
            setup_name = key.replace('_', ' ').title()
            rmse = f"{results['mean_rmse']:.3f}±{results['std_rmse']:.3f}"
            r2 = f"{results['mean_r2']:.3f}±{results['std_r2']:.3f}"
            mae = f"{results['mean_mae']:.3f}±{results['std_mae']:.3f}"

            print(f"{i:<4} {setup_name:<35} {rmse:<12} {r2:<12} {mae:<12}")

        # Best setup overall
        best_overall = sorted_results[0]
        print(f"\nBest setup: {best_overall[0]} with RMSE = {best_overall[1]['mean_rmse']:.3f}")


def run_xgb_ua(path_out):
    for crop in ['maize', 'winter_wheat', 'spring_barley']:
        pipeline = XGBYieldPredictionPipeline(f'Data/SC2/nuts2/{crop}_M.csv', 'yield')

        print("=== XGBoost COMPREHENSIVE ANALYSIS ===")
        # Run comprehensive analysis with all feature types and training modes (6 combinations)
        comprehensive_results = pipeline.run_comprehensive_analysis(n_processes=20)

        # Export predictions to CSV
        pipeline.export_predictions_to_csv(comprehensive_results, output_filename=f'{path_out}/{crop}_xgb.csv'
        )
