import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_validate, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
import itertools
import warnings
import os
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

def detrend_yield_by_region(df, region_col='field_id', year_col='c_year', yield_col='yield'):
    detrended = []
    for region, group in df.groupby(region_col):
        X = group[year_col].values.reshape(-1, 1)
        y = group[yield_col].values
        model = LinearRegression().fit(X, y)
        trend = model.predict(X)
        # The detrended value is the residual (difference)
        detrended_group = y - trend + np.mean(y)
        detrended.extend(detrended_group)
    # Add the detrended values as a new column
    df['yield'] = detrended
    return df


class YieldPredictionModel:
    """
    Advanced ANN model for yield prediction with transfer learning and cross-validation
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
        self.scaler_target = StandardScaler()
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
        print(f"Data loaded: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Available years: {sorted(self.data['c_year'].unique())}")
        print(f"Available regions: {sorted(self.data['country'].unique())}")

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
        self.eo_features.append('c_year')
        # Update ERA5 predictors list
        era5_predictors = set()
        for col in self.era5_features:
            for suffix in ['LT1', 'LT2', 'LT3', 'LT4']:
                if suffix in col:
                    predictor = col.replace(suffix, '').rstrip('_')
                    era5_predictors.add(predictor)
        self.era5_predictors = list(era5_predictors)
        self.era5_predictors.append('c_year')
        self.eo_predictors.append('c_year')

        print(f"Identified predictors: {sorted(predictors)}")
        print(f"EO predictors: {self.eo_predictors}")
        print(f"ERA5 predictors: {self.era5_predictors}")
        print(
            f"Total features: {len(self.feature_columns)} (EO: {len(self.eo_features)}, ERA5: {len(self.era5_features)})")

    def create_model_architecture(self, input_dim: int, dropout_rate: float = 0.3) -> keras.Model:
        """
        Create ANN architecture suitable for 2000-3000 samples

        Parameters:
        - input_dim: Number of input features
        - dropout_rate: Dropout rate for regularization
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),

            # First hidden layer
            # layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            # layers.BatchNormalization(),
            # layers.Dropout(dropout_rate),

            # Second hidden layer
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            # Third hidden layer
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            # Output layer
            layers.Dense(1, activation='linear')
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

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
        y_train = train_data[self.yield_column].values.reshape(-1, 1)
        X_test = test_data[selected_features].values
        y_test = test_data[self.yield_column].values.reshape(-1, 1)

        # Handle missing values
        X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
        X_test = np.nan_to_num(X_test, nan=np.nanmean(X_train))

        # Scale features
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_test_scaled = self.scaler_features.transform(X_test)

        # Scale target
        y_train_scaled = self.scaler_target.fit_transform(y_train)
        y_test_scaled = self.scaler_target.transform(y_test)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def train_base_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int = 200, patience: int = 20) -> keras.Model:
        """
        Train the base model with early stopping
        """
        model = self.create_model_architecture(X_train.shape[1])

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        return model

    def finetune_model(self, base_model: keras.Model, X_finetune: np.ndarray,
                       y_finetune: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                       epochs: int = 100, patience: int = 15) -> keras.Model:
        """
        Fine-tune the model using transfer learning
        """
        # Create a new model with the same architecture
        model = self.create_model_architecture(X_finetune.shape[1])

        # Copy weights from base model
        model.set_weights(base_model.get_weights())

        # Compile with lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )

        # Fine-tune
        history = model.fit(
            X_finetune, y_finetune,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=0
        )

        return model

    def evaluate_model(self, model: keras.Model, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        """
        # Predictions
        y_pred_scaled = model.predict(X_test, verbose=0)

        # Inverse transform to original scale
        y_test_orig = self.scaler_target.inverse_transform(y_test)
        y_pred_orig = self.scaler_target.inverse_transform(y_pred_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_true': y_test_orig.flatten(),
            'y_pred': y_pred_orig.flatten()
        }

    def calculate_permutation_importance(self, model: keras.Model, X: np.ndarray,
                                         y: np.ndarray, feature_names: List[str],
                                         n_repeats: int = 10) -> Dict:
        """
        Calculate permutation importance for neural network model

        Parameters:
        - model: Trained Keras model
        - X: Input features (scaled)
        - y: Target values (scaled)
        - feature_names: Names of features
        - n_repeats: Number of permutation repeats
        """
        # Get baseline score
        y_pred_baseline = model.predict(X, verbose=0).flatten()
        if len(y.shape) > 1:
            y_flat = y.flatten()
        else:
            y_flat = y

        baseline_score = r2_score(y_flat, y_pred_baseline)

        # Calculate permutation importance for each feature
        importances = []

        for feature_idx in range(X.shape[1]):
            feature_scores = []

            for repeat in range(n_repeats):
                # Create a copy of X and permute the feature
                X_permuted = X.copy()
                np.random.seed(42 + repeat)  # For reproducible results
                np.random.shuffle(X_permuted[:, feature_idx])

                # Get predictions with permuted feature
                y_pred_permuted = model.predict(X_permuted, verbose=0).flatten()

                # Calculate score decrease
                permuted_score = r2_score(y_flat, y_pred_permuted)
                score_decrease = baseline_score - permuted_score
                feature_scores.append(score_decrease)

            importances.append(feature_scores)

        # Convert to numpy array and calculate statistics
        importances = np.array(importances)
        importances_mean = np.mean(importances, axis=1)
        importances_std = np.std(importances, axis=1)

        # Create results dictionary
        importance_dict = {
            'feature_names': feature_names,
            'importances_mean': importances_mean,
            'importances_std': importances_std,
            'importances': importances,
            'baseline_score': baseline_score
        }

        return importance_dict

    def train_and_analyze_transfer_learning(self, test_year: int, feature_type: str = 'all',
                                            training_countries: List[str] = 'all',
                                            n_repeats: int = 10) -> Dict:
        """
        Train transfer learning model and analyze feature importance before and after fine-tuning

        Parameters:
        - test_year: Year to hold out for testing
        - feature_type: 'all', 'eo', or 'era5'
        - training_countries: Countries for base training
        - n_repeats: Number of permutation repeats
        """
        print(f"Analyzing transfer learning for test year {test_year}")
        print(f"Feature type: {feature_type}")
        print(f"Training countries: {training_countries}")

        # Filter data
        if training_countries != 'all':
            base_train_data = self.data[
                (self.data['country'].isin(training_countries)) &
                (self.data['c_year'] != test_year)
                ]
        else:
            base_train_data = self.data[
                (self.data['country'] != 'UA') &
                (self.data['c_year'] != test_year)
                ]

        ua_train_data = self.data[
            (self.data['country'] == 'UA') &
            (self.data['c_year'] != test_year)
            ]

        ua_test_data = self.data[
            (self.data['country'] == 'UA') &
            (self.data['c_year'] == test_year)
            ]

        # Check if we have enough data
        if len(base_train_data) < 50 or len(ua_train_data) < 10 or len(ua_test_data) < 5:
            raise ValueError("Insufficient data for analysis")

        # Split UA training data for validation
        ua_val_size = max(1, len(ua_train_data) // 5)
        ua_val_data = ua_train_data.sample(n=ua_val_size, random_state=42)
        ua_train_data = ua_train_data.drop(ua_val_data.index)

        # Prepare data
        X_base, _, y_base, _ = self.prepare_data_for_training(base_train_data, ua_test_data, feature_type)
        X_ua_train, _, y_ua_train, _ = self.prepare_data_for_training(ua_train_data, ua_test_data, feature_type)
        X_ua_val, _, y_ua_val, _ = self.prepare_data_for_training(ua_val_data, ua_test_data, feature_type)
        X_test, _, y_test, _ = self.prepare_data_for_training(ua_test_data, ua_test_data, feature_type)

        # Get feature names
        if feature_type == 'eo':
            feature_names = self.eo_features
        elif feature_type == 'era5':
            feature_names = self.era5_features
        else:
            feature_names = self.feature_columns

        # Train base model
        base_val_size = max(1, len(X_base) // 5)
        val_indices = np.random.choice(len(X_base), base_val_size, replace=False)
        train_indices = np.setdiff1d(np.arange(len(X_base)), val_indices)

        X_base_train, X_base_val = X_base[train_indices], X_base[val_indices]
        y_base_train, y_base_val = y_base[train_indices], y_base[val_indices]

        print("Training base model...")
        base_model = self.train_base_model(X_base_train, y_base_train, X_base_val, y_base_val)

        # Calculate feature importance for base model
        print("Calculating feature importance for base model...")
        base_importance = self.calculate_permutation_importance(
            base_model, X_base_val, y_base_val, feature_names, n_repeats
        )

        # Fine-tune model
        print("Fine-tuning model...")
        finetuned_model = self.finetune_model(
            base_model, X_ua_train, y_ua_train, X_ua_val, y_ua_val
        )

        # Calculate feature importance for fine-tuned model
        print("Calculating feature importance for fine-tuned model...")
        finetuned_importance = self.calculate_permutation_importance(
            finetuned_model, X_ua_val, y_ua_val, feature_names, n_repeats
        )

        # Evaluate both models on test data
        base_results = self.evaluate_model(base_model, X_test, y_test)
        finetuned_results = self.evaluate_model(finetuned_model, X_test, y_test)

        return {
            'test_year': test_year,
            'feature_type': feature_type,
            'training_countries': training_countries,
            'base_model_importance': base_importance,
            'finetuned_model_importance': finetuned_importance,
            'base_model_performance': base_results,
            'finetuned_model_performance': finetuned_results,
            'n_base_samples': len(base_train_data),
            'n_ua_train_samples': len(ua_train_data),
            'n_test_samples': len(ua_test_data)
        }

    def analyze_feature_importance_across_years(self, feature_type: str = 'all',
                                                training_countries: List[str] = 'all',
                                                n_repeats: int = 10) -> Dict:
        """
        Analyze feature importance across multiple years

        Parameters:
        - feature_type: 'all', 'eo', or 'era5'
        - training_countries: Countries for base training
        - n_repeats: Number of permutation repeats
        """
        # Get available years
        ua_years = self.data[self.data['country'] == 'UA']['c_year'].unique()
        years_to_analyze = sorted(ua_years)  # Analyze first 3 years to save time

        print(f"Analyzing feature importance for years: {years_to_analyze}")

        all_results = {}

        for year in years_to_analyze:
            try:
                print(f"\nAnalyzing year {year}...")
                result = self.train_and_analyze_transfer_learning(
                    test_year=year,
                    feature_type=feature_type,
                    training_countries=training_countries,
                    n_repeats=n_repeats
                )
                all_results[year] = result
            except Exception as e:
                print(f"Error analyzing year {year}: {str(e)}")
                continue

        return all_results

    def plot_feature_importance_comparison(self, importance_results: Dict,
                                           figsize: Tuple[int, int] = (16, 12)):
        """
        Plot feature importance comparison between base and fine-tuned models

        Parameters:
        - importance_results: Results from train_and_analyze_transfer_learning
        - figsize: Figure size
        """
        if len(importance_results) == 0:
            print("No results to plot")
            return

        # Number of years to plot
        n_years = len(importance_results)

        # Create subplots
        fig, axes = plt.subplots(n_years, 2, figsize=figsize)
        if n_years == 1:
            axes = axes.reshape(1, -1)

        for i, (year, results) in enumerate(importance_results.items()):
            base_imp = results['base_model_importance']
            finetuned_imp = results['finetuned_model_importance']

            # Get top 15 most important features for better visualization
            n_features = min(15, len(base_imp['feature_names']))

            # Sort by base model importance
            base_indices = np.argsort(base_imp['importances_mean'])[-n_features:]

            # Base model importance
            ax1 = axes[i, 0]
            y_pos = np.arange(n_features)
            base_means = base_imp['importances_mean'][base_indices]
            base_stds = base_imp['importances_std'][base_indices]
            feature_names = [base_imp['feature_names'][j] for j in base_indices]

            bars1 = ax1.barh(y_pos, base_means, xerr=base_stds,
                             capsize=3, alpha=0.7, color='skyblue')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(feature_names, fontsize=8)
            ax1.set_xlabel('Permutation Importance')
            ax1.set_title(
                f'Base Model Feature Importance\nYear {year} (R²={results["base_model_performance"]["r2"]:.3f})')
            ax1.grid(True, alpha=0.3)

            # Fine-tuned model importance (same features for comparison)
            ax2 = axes[i, 1]
            finetuned_means = finetuned_imp['importances_mean'][base_indices]
            finetuned_stds = finetuned_imp['importances_std'][base_indices]

            bars2 = ax2.barh(y_pos, finetuned_means, xerr=finetuned_stds,
                             capsize=3, alpha=0.7, color='lightcoral')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(feature_names, fontsize=8)
            ax2.set_xlabel('Permutation Importance')
            ax2.set_title(
                f'Fine-tuned Model Feature Importance\nYear {year} (R²={results["finetuned_model_performance"]["r2"]:.3f})')
            ax2.grid(True, alpha=0.3)

        plt.suptitle('Feature Importance Comparison: Base vs Fine-tuned Models',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_importance_change_analysis(self, importance_results: Dict,
                                        figsize: Tuple[int, int] = (14, 10)):
        """
        Plot analysis of how feature importance changes after fine-tuning

        Parameters:
        - importance_results: Results from train_and_analyze_transfer_learning
        - figsize: Figure size
        """
        if len(importance_results) == 0:
            print("No results to plot")
            return

        # Aggregate importance changes across years
        all_feature_names = set()
        importance_changes = {}

        for year, results in importance_results.items():
            base_imp = results['base_model_importance']
            finetuned_imp = results['finetuned_model_importance']

            for i, feature in enumerate(base_imp['feature_names']):
                all_feature_names.add(feature)
                if feature not in importance_changes:
                    importance_changes[feature] = []

                # Calculate change in importance
                base_val = base_imp['importances_mean'][i]
                finetuned_val = finetuned_imp['importances_mean'][i]
                change = finetuned_val - base_val
                importance_changes[feature].append(change)

        # Calculate average changes
        avg_changes = {}
        std_changes = {}
        for feature, changes in importance_changes.items():
            avg_changes[feature] = np.mean(changes)
            std_changes[feature] = np.std(changes)

        # Sort by absolute average change
        sorted_features = sorted(avg_changes.keys(),
                                 key=lambda x: abs(avg_changes[x]),
                                 reverse=True)

        # Plot top 20 features with largest changes
        n_features = min(20, len(sorted_features))
        top_features = sorted_features[:n_features]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Average importance change
        y_pos = np.arange(n_features)
        changes = [avg_changes[f] for f in top_features]
        stds = [std_changes[f] for f in top_features]
        colors = ['green' if c > 0 else 'red' for c in changes]

        bars = ax1.barh(y_pos, changes, xerr=stds, capsize=3,
                        color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_features, fontsize=8)
        ax1.set_xlabel('Change in Importance (Fine-tuned - Base)')
        ax1.set_title('Average Change in Feature Importance\nAfter Fine-tuning')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Scatter plot of base vs fine-tuned importance
        # Use data from the first year for visualization
        first_year_result = list(importance_results.values())[0]
        base_imp = first_year_result['base_model_importance']
        finetuned_imp = first_year_result['finetuned_model_importance']

        ax2.scatter(base_imp['importances_mean'], finetuned_imp['importances_mean'],
                    alpha=0.6, s=50)

        # Add diagonal line (no change)
        max_imp = max(base_imp['importances_mean'].max(),
                      finetuned_imp['importances_mean'].max())
        min_imp = min(base_imp['importances_mean'].min(),
                      finetuned_imp['importances_mean'].min())
        ax2.plot([min_imp, max_imp], [min_imp, max_imp], 'r--', alpha=0.8,
                 label='No Change')

        ax2.set_xlabel('Base Model Importance')
        ax2.set_ylabel('Fine-tuned Model Importance')
        ax2.set_title(f'Importance Correlation\n(Year {first_year_result["test_year"]})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(base_imp['importances_mean'],
                           finetuned_imp['importances_mean'])[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                 transform=ax2.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def analyze_predictor_groups(self, importance_results: Dict) -> pd.DataFrame:
        """
        Analyze importance by predictor groups (EO vs ERA5, different time periods)

        Parameters:
        - importance_results: Results from analyze_feature_importance_across_years
        """
        # Group features by type and time period
        predictor_analysis = []

        for year, results in importance_results.items():
            base_imp = results['base_model_importance']
            finetuned_imp = results['finetuned_model_importance']

            for i, feature in enumerate(base_imp['feature_names']):
                # Determine predictor type
                is_eo = any(eo_pred in feature.lower() for eo_pred in self.eo_predictors)
                predictor_type = 'EO' if is_eo else 'ERA5'

                # Determine time period
                time_period = 'Unknown'
                for suffix in ['LT1', 'LT2', 'LT3', 'LT4']:
                    if suffix in feature:
                        time_period = suffix
                        break

                # Extract base predictor name
                base_predictor = feature
                for suffix in ['LT1', 'LT2', 'LT3', 'LT4']:
                    base_predictor = base_predictor.replace(suffix, '').rstrip('_')

                predictor_analysis.append({
                    'year': year,
                    'feature': feature,
                    'predictor_type': predictor_type,
                    'time_period': time_period,
                    'base_predictor': base_predictor,
                    'base_importance': base_imp['importances_mean'][i],
                    'finetuned_importance': finetuned_imp['importances_mean'][i],
                    'importance_change': finetuned_imp['importances_mean'][i] - base_imp['importances_mean'][i]
                })

        df = pd.DataFrame(predictor_analysis)

        # Print summary
        print("\n" + "=" * 80)
        print("PREDICTOR GROUP ANALYSIS")
        print("=" * 80)

        # Average importance by predictor type
        type_summary = df.groupby('predictor_type').agg({
            'base_importance': ['mean', 'std'],
            'finetuned_importance': ['mean', 'std'],
            'importance_change': ['mean', 'std']
        }).round(4)

        print("\nAverage Importance by Predictor Type:")
        print(type_summary)

        # Average importance by time period
        time_summary = df.groupby('time_period').agg({
            'base_importance': ['mean', 'std'],
            'finetuned_importance': ['mean', 'std'],
            'importance_change': ['mean', 'std']
        }).round(4)

        print("\nAverage Importance by Time Period:")
        print(time_summary)

        # Top predictors by base importance
        print("\nTop 10 Most Important Predictors (Base Model):")
        top_base = df.groupby('base_predictor')['base_importance'].mean().sort_values(ascending=False).head(10)
        for pred, imp in top_base.items():
            print(f"  {pred}: {imp:.4f}")

        # Top predictors by fine-tuned importance
        print("\nTop 10 Most Important Predictors (Fine-tuned Model):")
        top_finetuned = df.groupby('base_predictor')['finetuned_importance'].mean().sort_values(ascending=False).head(
            10)
        for pred, imp in top_finetuned.items():
            print(f"  {pred}: {imp:.4f}")

        # Predictors with largest changes
        print("\nPredictors with Largest Importance Changes:")
        change_summary = df.groupby('base_predictor')['importance_change'].mean().sort_values(key=abs,
                                                                                              ascending=False).head(10)
        for pred, change in change_summary.items():
            direction = "increased" if change > 0 else "decreased"
            print(f"  {pred}: {change:+.4f} ({direction})")

        return df

    def export_feature_importance_results(self, importance_results: Dict,
                                          output_filename: str = 'feature_importance_analysis.csv'):
        """
        Export feature importance results to CSV

        Parameters:
        - importance_results: Results from analyze_feature_importance_across_years
        - output_filename: Output CSV filename
        """
        all_data = []

        for year, results in importance_results.items():
            base_imp = results['base_model_importance']
            finetuned_imp = results['finetuned_model_importance']

            for i, feature in enumerate(base_imp['feature_names']):
                all_data.append({
                    'year': year,
                    'feature': feature,
                    'base_importance_mean': base_imp['importances_mean'][i],
                    'base_importance_std': base_imp['importances_std'][i],
                    'finetuned_importance_mean': finetuned_imp['importances_mean'][i],
                    'finetuned_importance_std': finetuned_imp['importances_std'][i],
                    'importance_change': finetuned_imp['importances_mean'][i] - base_imp['importances_mean'][i],
                    'base_model_r2': results['base_model_performance']['r2'],
                    'finetuned_model_r2': results['finetuned_model_performance']['r2'],
                    'feature_type': results['feature_type']
                })

        df = pd.DataFrame(all_data)
        df.to_csv(output_filename, index=False)

        print(f"\nFeature importance results exported to {output_filename}")
        print(f"Shape: {df.shape}")

        return df

def cross_validation_worker(args):
    """
    Worker function for parallel cross-validation
    """
    test_year, training_countries, csv_path, yield_column, feature_type, training_mode = args

    try:
        # Create model instance for this worker
        model_instance = YieldPredictionModel(csv_path, yield_column)
        data = model_instance.data

        # Filter data based on training mode
        if training_mode == 'transfer_learning':
            # Original transfer learning setup
            if training_countries != 'all':
                # base_train_data = data[
                #     (data['country'].isin(training_countries)) &
                #     (data['c_year'] != test_year)
                #     ]
                base_train_data = data[
                    (data['country'].isin(training_countries)) &
                    (~data['c_year'].isin([test_year, 2022]))
                    ]
            else:
                # base_train_data = data[
                #     (data['country'] != 'UA') &
                #     (data['c_year'] != test_year)
                #     ]
                base_train_data = data[
                    (data['country'] != 'UA') &
                    (~data['c_year'].isin([test_year, 2022]))
                    ]

            # ua_train_data = data[
            #     (data['country'] == 'UA') &
            #     (data['c_year'] != test_year)
            #     ]

            ua_train_data = data[
                (data['country'] == 'UA') &
                (~data['c_year'].isin([test_year, 2022]))
                ]

            ua_test_data = data[
                (data['country'] == 'UA') &
                (data['c_year'] == test_year)
                ]

            # Check if we have enough data
            if len(base_train_data) < 50 or len(ua_train_data) < 10 or len(ua_test_data) < 5:
                return test_year, None

            # Split UA training data for validation
            ua_val_size = max(1, len(ua_train_data) // 5)  # 20% for validation
            ua_val_data = ua_train_data.sample(n=ua_val_size, random_state=42)
            ua_train_data = ua_train_data.drop(ua_val_data.index)

            # Prepare data
            X_base, _, y_base, _ = model_instance.prepare_data_for_training(base_train_data, ua_test_data, feature_type)
            X_ua_train, _, y_ua_train, _ = model_instance.prepare_data_for_training(ua_train_data, ua_test_data,
                                                                                    feature_type)
            X_ua_val, _, y_ua_val, _ = model_instance.prepare_data_for_training(ua_val_data, ua_test_data, feature_type)
            X_test, _, y_test, _ = model_instance.prepare_data_for_training(ua_test_data, ua_test_data, feature_type)

            # Train base model
            base_val_size = max(1, len(X_base) // 5)
            val_indices = np.random.choice(len(X_base), base_val_size, replace=False)
            train_indices = np.setdiff1d(np.arange(len(X_base)), val_indices)

            X_base_train, X_base_val = X_base[train_indices], X_base[val_indices]
            y_base_train, y_base_val = y_base[train_indices], y_base[val_indices]

            base_model = model_instance.train_base_model(
                X_base_train, y_base_train, X_base_val, y_base_val
            )

            # Fine-tune model
            final_model = model_instance.finetune_model(
                base_model, X_ua_train, y_ua_train, X_ua_val, y_ua_val
            )

            training_setup = 'transfer_learning'

        elif training_mode == 'direct_training':
            # Direct training on all regions except UA and except test year
            if training_countries != 'all':
                all_train_data = data[
                    (data['country'].isin(training_countries)) &
                    (data['c_year'] != test_year)
                    ]
            else:
                all_train_data = data[
                    (data['country'] != 'UA') &
                    (data['c_year'] != test_year)
                    ]

            ua_test_data = data[
                (data['country'] == 'UA') &
                (data['c_year'] == test_year)
                ]

            # Check if we have enough data
            if len(all_train_data) < 50 or len(ua_test_data) < 5:
                return test_year, None

            # Prepare data
            X_train, X_test, y_train, y_test = model_instance.prepare_data_for_training(
                all_train_data, ua_test_data, feature_type
            )

            # Split training data for validation
            val_size = max(1, len(X_train) // 5)
            val_indices = np.random.choice(len(X_train), val_size, replace=False)
            train_indices = np.setdiff1d(np.arange(len(X_train)), val_indices)

            X_train_split, X_val = X_train[train_indices], X_train[val_indices]
            y_train_split, y_val = y_train[train_indices], y_train[val_indices]

            # Train model directly
            final_model = model_instance.train_base_model(
                X_train_split, y_train_split, X_val, y_val
            )

            training_setup = 'direct_training'

        elif training_mode == 'direct_training_all_countries':
            # NEW: Direct training on ALL countries including UA (except test year)
            if training_countries != 'all':
                all_train_data = data[
                    (data['country'].isin(training_countries + ['UA'])) &
                    (data['c_year'] != test_year)
                    ]
            else:
                all_train_data = data[data['c_year'] != test_year]

            ua_test_data = data[
                (data['country'] == 'UA') &
                (data['c_year'] == test_year)
                ]

            # Check if we have enough data
            if len(all_train_data) < 50 or len(ua_test_data) < 5:
                return test_year, None

            # Prepare data
            X_train, X_test, y_train, y_test = model_instance.prepare_data_for_training(
                all_train_data, ua_test_data, feature_type
            )

            # Split training data for validation
            val_size = max(1, len(X_train) // 5)
            val_indices = np.random.choice(len(X_train), val_size, replace=False)
            train_indices = np.setdiff1d(np.arange(len(X_train)), val_indices)

            X_train_split, X_val = X_train[train_indices], X_train[val_indices]
            y_train_split, y_val = y_train[train_indices], y_train[val_indices]

            # Train model directly on all countries
            final_model = model_instance.train_base_model(
                X_train_split, y_train_split, X_val, y_val
            )

            training_setup = 'direct_training_all_countries'

        # Evaluate and get predictions
        results = model_instance.evaluate_model(final_model, X_test, y_test)
        results['test_year'] = test_year
        results['feature_type'] = feature_type
        results['training_mode'] = training_setup
        results['n_test_samples'] = len(ua_test_data)

        # Add region information for CSV export
        results['test_regions'] = ua_test_data['field_id'].values
        results['test_years'] = ua_test_data['c_year'].values

        if training_mode == 'transfer_learning':
            results['n_base_samples'] = len(base_train_data)
            results['n_ua_train_samples'] = len(ua_train_data)
        else:
            results['n_train_samples'] = len(all_train_data)

        return test_year, results

    except Exception as e:
        print(f"Error in year {test_year} with {feature_type} features and {training_mode}: {str(e)}")
        return test_year, None


class YieldPredictionPipeline:
    """
    Main pipeline for yield prediction with cross-validation and country comparison
    """

    def __init__(self, csv_path: str, yield_column: str = 'yield'):
        self.csv_path = csv_path
        self.yield_column = yield_column
        self.data = pd.read_csv(self.csv_path, index_col=0)
        self.data = self.data.drop(columns=[col for col in self.data.columns if col.startswith('lai')])
        self.data.loc[:, 'country'] = [a_col[:2] for a_col in self.data.field_id]
        self.results = {}

    def run_leave_one_year_out_cv(self, training_countries: List[str] = 'all',
                                  feature_type: str = 'all', training_mode: str = 'transfer_learning',
                                  n_processes: Optional[int] = None) -> Dict:
        """
        Run leave-one-year-out cross-validation with parallel processing

        Parameters:
        - training_countries: List of countries for base training or 'all' for all except UA
        - feature_type: 'all', 'eo', or 'era5' to select feature subset
        - training_mode: 'transfer_learning', 'direct_training', or 'direct_training_all_countries'
        - n_processes: Number of parallel processes (default: CPU count - 1)
        """
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)

        # Get unique years where UA has data
        ua_years = self.data[self.data['country'] == 'UA']['c_year'].unique()
        years_to_test = sorted(ua_years)

        print(f"Running LOYO CV for years: {years_to_test}")
        print(f"Using {n_processes} processes")
        print(f"Training countries: {training_countries}")
        print(f"Feature type: {feature_type}")
        print(f"Training mode: {training_mode}")

        # Prepare arguments for parallel processing
        args_list = [(year, training_countries, self.csv_path, self.yield_column, feature_type, training_mode)
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

    def run_comprehensive_analysis(self, training_countries: List[str] = 'all',
                                   n_processes: Optional[int] = None) -> Dict:
        """
        Run comprehensive analysis with all feature types and training modes

        Parameters:
        - training_countries: List of countries for base training or 'all' for all except UA
        - n_processes: Number of parallel processes
        """
        feature_types = ['all', 'eo', 'era5']
        training_modes = ['transfer_learning', 'direct_training', 'direct_training_all_countries']

        comprehensive_results = {}

        for feature_type in feature_types:
            for training_mode in training_modes:
                print(f"\n{'=' * 60}")
                print(f"Running analysis: {feature_type} features, {training_mode}")
                print(f"{'=' * 60}")

                try:
                    cv_results = self.run_leave_one_year_out_cv(
                        training_countries=training_countries,
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

    def compare_countries(self, country_combinations: List[List[str]] = None,
                          feature_type: str = 'all', training_mode: str = 'transfer_learning',
                          n_processes: Optional[int] = None) -> Dict:
        """
        Compare model performance using different country combinations for base training

        Parameters:
        - country_combinations: List of country combinations to test
        - feature_type: 'all', 'eo', or 'era5' to select feature subset
        - training_mode: 'transfer_learning', 'direct_training', or 'direct_training_all_countries'
        - n_processes: Number of parallel processes
        """
        if country_combinations is None:
            # Get available countries (excluding UA)
            available_countries = [c for c in self.data['country'].unique() if c != 'UA']

            # Create different combinations
            country_combinations = [
                ['all'],  # All countries except UA
                available_countries[:2] if len(available_countries) >= 2 else available_countries,
                available_countries[:3] if len(available_countries) >= 3 else available_countries,
                available_countries  # All available countries
            ]

        results = {}

        for i, countries in enumerate(country_combinations):
            print(f"\nTesting combination {i + 1}/{len(country_combinations)}: {countries}")

            cv_results = self.run_leave_one_year_out_cv(
                training_countries=countries,
                feature_type=feature_type,
                training_mode=training_mode,
                n_processes=n_processes
            )

            # Calculate summary statistics
            if cv_results:
                rmse_values = [r['rmse'] for r in cv_results.values()]
                r2_values = [r['r2'] for r in cv_results.values()]

                summary = {
                    'mean_rmse': np.mean(rmse_values),
                    'std_rmse': np.std(rmse_values),
                    'mean_r2': np.mean(r2_values),
                    'std_r2': np.std(r2_values),
                    'detailed_results': cv_results,
                    'countries': countries,
                    'feature_type': feature_type,
                    'training_mode': training_mode
                }

                results[str(countries)] = summary

        return results

    def plot_comprehensive_results(self, comprehensive_results: Dict):
        """
        Plot comprehensive results comparing feature types and training modes
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Prepare data
        feature_types = ['all', 'eo', 'era5']
        training_modes = ['transfer_learning', 'direct_training']

        # Extract metrics for plotting
        rmse_data = {}
        r2_data = {}

        for feature_type in feature_types:
            rmse_data[feature_type] = []
            r2_data[feature_type] = []

            for training_mode in training_modes:
                key = f"{feature_type}_{training_mode}"
                if key in comprehensive_results:
                    rmse_data[feature_type].append(comprehensive_results[key]['mean_rmse'])
                    r2_data[feature_type].append(comprehensive_results[key]['mean_r2'])
                else:
                    rmse_data[feature_type].append(np.nan)
                    r2_data[feature_type].append(np.nan)

        # Plot 1: RMSE comparison by feature type and training mode
        x = np.arange(len(training_modes))
        width = 0.25

        for i, feature_type in enumerate(feature_types):
            axes[0, 0].bar(x + i * width, rmse_data[feature_type], width,
                           label=f'{feature_type.upper()} features', alpha=0.8)

        axes[0, 0].set_title('RMSE by Feature Type and Training Mode')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_xlabel('Training Mode')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(['Transfer Learning', 'Direct Training'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: R² comparison by feature type and training mode
        for i, feature_type in enumerate(feature_types):
            axes[0, 1].bar(x + i * width, r2_data[feature_type], width,
                           label=f'{feature_type.upper()} features', alpha=0.8)

        axes[0, 1].set_title('R² by Feature Type and Training Mode')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_xlabel('Training Mode')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels(['Transfer Learning', 'Direct Training'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Feature type comparison (average across training modes)
        avg_rmse = [np.nanmean(rmse_data[ft]) for ft in feature_types]
        avg_r2 = [np.nanmean(r2_data[ft]) for ft in feature_types]

        axes[0, 2].bar(feature_types, avg_rmse, alpha=0.8, color='skyblue')
        axes[0, 2].set_title('Average RMSE by Feature Type')
        axes[0, 2].set_ylabel('RMSE')
        axes[0, 2].set_xlabel('Feature Type')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Performance by year for each setup
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for i, (key, results) in enumerate(comprehensive_results.items()):
            if 'detailed_results' in results:
                years = list(results['detailed_results'].keys())
                rmse_by_year = [results['detailed_results'][year]['rmse'] for year in years]
                axes[1, 0].plot(years, rmse_by_year, marker='o',
                                label=key.replace('_', ' ').title(),
                                color=colors[i % len(colors)])

        axes[1, 0].set_title('RMSE by Year and Setup')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: R² by year for each setup
        for i, (key, results) in enumerate(comprehensive_results.items()):
            if 'detailed_results' in results:
                years = list(results['detailed_results'].keys())
                r2_by_year = [results['detailed_results'][year]['r2'] for year in years]
                axes[1, 1].plot(years, r2_by_year, marker='o',
                                label=key.replace('_', ' ').title(),
                                color=colors[i % len(colors)])

        axes[1, 1].set_title('R² by Year and Setup')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Summary statistics heatmap
        summary_data = []
        labels = []

        for key, results in comprehensive_results.items():
            summary_data.append([results['mean_rmse'], results['mean_r2'], results['mean_mae']])
            labels.append(key.replace('_', ' ').title())

        if summary_data:
            im = axes[1, 2].imshow(summary_data, cmap='RdYlBu_r', aspect='auto')
            axes[1, 2].set_title('Performance Metrics Heatmap')
            axes[1, 2].set_xticks([0, 1, 2])
            axes[1, 2].set_xticklabels(['RMSE', 'R²', 'MAE'])
            axes[1, 2].set_yticks(range(len(labels)))
            axes[1, 2].set_yticklabels(labels)

            # Add colorbar
            plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()
        plt.show()

        # Print detailed summary
        self.print_comprehensive_summary(comprehensive_results)

    def print_comprehensive_summary(self, comprehensive_results: Dict):
        """
        Print detailed summary of comprehensive results
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
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

        print("\n" + "=" * 80)
        print("FEATURE TYPE ANALYSIS")
        print("=" * 80)

        # Analyze by feature type
        feature_analysis = {}
        for key, results in comprehensive_results.items():
            feature_type = results['feature_type']
            if feature_type not in feature_analysis:
                feature_analysis[feature_type] = []
            feature_analysis[feature_type].append(results['mean_rmse'])

        for feature_type, rmse_values in feature_analysis.items():
            avg_rmse = np.mean(rmse_values)
            print(f"{feature_type.upper()} features: Average RMSE = {avg_rmse:.3f}")

        print("\n" + "=" * 80)
        print("TRAINING MODE ANALYSIS")
        print("=" * 80)

        # Analyze by training mode
        mode_analysis = {}
        for key, results in comprehensive_results.items():
            training_mode = results['training_mode']
            if training_mode not in mode_analysis:
                mode_analysis[training_mode] = []
            mode_analysis[training_mode].append(results['mean_rmse'])

        for training_mode, rmse_values in mode_analysis.items():
            avg_rmse = np.mean(rmse_values)
            print(f"{training_mode.replace('_', ' ').title()}: Average RMSE = {avg_rmse:.3f}")

    def plot_performance_comparison(self, comparison_results: Dict):
        """
        Plot performance comparison across different country combinations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Prepare data for plotting
        combinations = list(comparison_results.keys())
        mean_rmse = [comparison_results[combo]['mean_rmse'] for combo in combinations]
        std_rmse = [comparison_results[combo]['std_rmse'] for combo in combinations]
        mean_r2 = [comparison_results[combo]['mean_r2'] for combo in combinations]
        std_r2 = [comparison_results[combo]['std_r2'] for combo in combinations]

        # RMSE comparison
        axes[0, 0].bar(range(len(combinations)), mean_rmse, yerr=std_rmse, capsize=5)
        axes[0, 0].set_title('RMSE by Country Combination')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_xticks(range(len(combinations)))
        axes[0, 0].set_xticklabels([f'Combo {i + 1}' for i in range(len(combinations))], rotation=45)

        # R² comparison
        axes[0, 1].bar(range(len(combinations)), mean_r2, yerr=std_r2, capsize=5)
        axes[0, 1].set_title('R² by Country Combination')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_xticks(range(len(combinations)))
        axes[0, 1].set_xticklabels([f'Combo {i + 1}' for i in range(len(combinations))], rotation=45)

        # Detailed RMSE by year for each combination
        for i, combo in enumerate(combinations):
            years = list(comparison_results[combo]['detailed_results'].keys())
            rmse_by_year = [comparison_results[combo]['detailed_results'][year]['rmse'] for year in years]
            axes[1, 0].plot(years, rmse_by_year, marker='o', label=f'Combo {i + 1}')

        axes[1, 0].set_title('RMSE by Year and Country Combination')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()

        # Detailed R² by year for each combination
        for i, combo in enumerate(combinations):
            years = list(comparison_results[combo]['detailed_results'].keys())
            r2_by_year = [comparison_results[combo]['detailed_results'][year]['r2'] for year in years]
            axes[1, 1].plot(years, r2_by_year, marker='o', label=f'Combo {i + 1}')

        axes[1, 1].set_title('R² by Year and Country Combination')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\nPerformance Summary:")
        print("-" * 60)
        for i, combo in enumerate(combinations):
            countries = comparison_results[combo]['countries']
            mean_rmse = comparison_results[combo]['mean_rmse']
            mean_r2 = comparison_results[combo]['mean_r2']
            feature_type = comparison_results[combo]['feature_type']
            training_mode = comparison_results[combo]['training_mode']
            print(f"Combo {i + 1} ({countries}, {feature_type} features, {training_mode}): "
                  f"RMSE={mean_rmse:.3f}, R²={mean_r2:.3f}")

    def plot_predictions_vs_actual(self, cv_results: Dict, year: int = None):
        """
        Plot predictions vs actual values
        """
        if year is None:
            # Combine all years
            y_true_all = np.concatenate([r['y_true'] for r in cv_results.values()])
            y_pred_all = np.concatenate([r['y_pred'] for r in cv_results.values()])
            title = "Predictions vs Actual (All Years)"
        else:
            if year not in cv_results:
                print(f"Year {year} not found in results")
                return
            y_true_all = cv_results[year]['y_true']
            y_pred_all = cv_results[year]['y_pred']
            title = f"Predictions vs Actual (Year {year})"

        plt.figure(figsize=(10, 8))
        plt.scatter(y_true_all, y_pred_all, alpha=0.6)

        # Perfect prediction line
        min_val = min(y_true_all.min(), y_pred_all.min())
        max_val = max(y_true_all.max(), y_pred_all.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def export_predictions_to_csv(self, comprehensive_results: Dict,
                                  output_filename: str = 'yield_predictions.csv'):
        """
        Export predictions to CSV with the requested format:
        year, region, predicted_all_transfer, predicted_eo_transfer, predicted_era5_transfer,
        predicted_all_direct, predicted_eo_direct, predicted_era5_direct,
        predicted_all_direct_all, predicted_eo_direct_all, predicted_era5_direct_all,
        observed_yield
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
                    'predicted_all_transfer_learning',
                    'predicted_eo_transfer_learning',
                    'predicted_era5_transfer_learning',
                    'predicted_all_direct_training',
                    'predicted_eo_direct_training',
                    'predicted_era5_direct_training',
                    'predicted_all_direct_training_all_countries',
                    'predicted_eo_direct_training_all_countries',
                    'predicted_era5_direct_training_all_countries'
                ]

                for col in column_names:
                    if col not in row:
                        row[col] = np.nan

                prediction_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(prediction_data)

        # Reorder columns as requested
        desired_columns = [
            'year', 'region',
            'predicted_all_transfer_learning', 'predicted_eo_transfer_learning', 'predicted_era5_transfer_learning',
            'predicted_all_direct_training', 'predicted_eo_direct_training', 'predicted_era5_direct_training',
            'predicted_all_direct_training_all_countries', 'predicted_eo_direct_training_all_countries',
            'predicted_era5_direct_training_all_countries',
            'observed_yield'
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

def run_ann_ua(results_path="Results/SC2/nuts2/ann1"):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for crop in ['winter_wheat', 'spring_barley', 'maize']:  # , 'maize'
        pipeline = YieldPredictionPipeline(f'Data/SC2/nuts2/{crop}_M.csv', 'yield')

        print("=== COMPREHENSIVE ANALYSIS ===")
        # Run comprehensive analysis with all feature types and training modes (now 9 combinations)
        comprehensive_results = pipeline.run_comprehensive_analysis(
            training_countries='all',
            # training_countries=['FR', 'DE', 'PL', 'UA'],
            n_processes=20
        )

        # Plot comprehensive results
        pipeline.plot_comprehensive_results(comprehensive_results)

        # Export predictions to CSV
        predictions_df = pipeline.export_predictions_to_csv(
            comprehensive_results,
            output_filename=f'{results_path}/{crop}.csv'
        )

def xgb_run_ua(crop):
    # Load and prepare data
    file = pd.read_csv(f'Data/nuts2/{crop}_M.csv', index_col=0)
    file = file.dropna(axis=0, how='any')
    file.index = range(len(file))

    file.loc[:, 'country'] = [f[:2] for f in file.field_id]
    countries = np.unique(file.country)

    predictors = list(file.columns[3:-1])
    predictors.append('c_year')
    preds = ('t2m', 'tp', 'swvl1', 'pev', 'lai_lv', 'evavt', 'ssr', 'evi', 'sm', 'VODCA_CXKu')

    used_predictors = [a for a in predictors if a.startswith(preds)]
    met_pred = [a for a in used_predictors if not a.startswith(('evi', 'sm', 'VODCA'))]
    eo_pred = [a for a in used_predictors if a.startswith(('evi', 'sm', 'VODCA', 'c_year'))]

    for country in ['UA']:
        inds_c = np.where(file.country == country)[0]
        # inds_not_c = np.where(file.country != country)[0]
        X = file.loc[inds_c, used_predictors]
        X_eo = file.loc[inds_c, eo_pred]
        X_met = file.loc[inds_c, met_pred]
        years = file.loc[inds_c, 'c_year']
        regs = file.loc[inds_c, 'field_id']

        X.index = pd.to_datetime(years.values, format="%Y")
        X_eo.index = pd.to_datetime(years.values, format="%Y")
        X_met.index = pd.to_datetime(years.values,format="%Y")
        y = file.loc[inds_c, 'yield']

        estimator = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=50)
        pipe_xgb = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])

        groups = years
        logo = LeaveOneGroupOut()
        preds = cross_val_predict(pipe_xgb, X, y, groups=groups, cv=logo, n_jobs=30)
        preds_met = cross_val_predict(pipe_xgb, X_met, y, groups=groups, cv=logo, n_jobs=30)
        preds_eo = cross_val_predict(pipe_xgb, X_eo, y, groups=groups, cv=logo, n_jobs=30)

        result_df = pd.DataFrame({'region': regs.values, 'year': X['c_year'].values, 'xgb': preds,
                                  'xgb_eo': preds_eo, 'xgb_met': preds_met, 'observed': y.values})

        result_df.to_csv(f'Results/SC2/nuts2/xgb/{country}_{crop}_all.csv')

def run_fi(crop, path, var="all"):
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

    # Initialize model for feature importance analysis
    model = YieldPredictionModel(f'Data/SC2/nuts2/{crop}_M.csv', 'yield')

    # Analyze feature importance for a single year
    # print("\n1. Single year feature importance analysis:")
    # single_year_results = model.train_and_analyze_transfer_learning(
    #     test_year=2020,  # Replace with an available year
    #     feature_type='all',
    #     training_countries='all',
    #     n_repeats=5  # Reduce for faster execution
    # )
    #
    # # Plot single year results
    # single_year_dict = {2020: single_year_results}
    # model.plot_feature_importance_comparison(single_year_dict)
    # model.plot_importance_change_analysis(single_year_dict)

    # Analyze feature importance across multiple years
    print("\n2. Multi-year feature importance analysis:")
    multi_year_results = model.analyze_feature_importance_across_years(
        feature_type=var,
        training_countries='all',
        n_repeats=5  # Reduce for faster execution
    )

    # Plot multi-year results
    model.plot_feature_importance_comparison(multi_year_results)
    model.plot_importance_change_analysis(multi_year_results)

    # Analyze predictor groups
    predictor_df = model.analyze_predictor_groups(multi_year_results)

    # Export feature importance results
    importance_df = model.export_feature_importance_results(
        multi_year_results,
        f'{path}/{crop}_feature_importance_{var}.csv'
    )

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    # crop = 'maize'
    # run_fi(crop)
    for crop in ['winter_wheat', 'spring_barley', 'maize']: #, 'maize'
        run_fi(crop, path='Results/SC2/nuts2/ann4', var="eo")
        run_fi(crop, path='Results/SC2/nuts2/ann4', var="era5")
        # pipeline = YieldPredictionPipeline(f'Data/SC2/nuts2/{crop}_M.csv', 'yield')
        #
        # print("=== COMPREHENSIVE ANALYSIS ===")
        # # Run comprehensive analysis with all feature types and training modes (now 9 combinations)
        # comprehensive_results = pipeline.run_comprehensive_analysis(
        #     # training_countries='all',  # or specify countries like ['DE', 'FR', 'ES']
        #     training_countries=['FR', 'DE', 'PL', 'UA'],
        #     n_processes=20
        # )
        #
        # # Plot comprehensive results
        # pipeline.plot_comprehensive_results(comprehensive_results)
        #
        # # Export predictions to CSV
        # predictions_df = pipeline.export_predictions_to_csv(
        #     comprehensive_results,
        #     output_filename=f'Results/SC2/nuts2/ann1/{crop}_best_c.csv'
        # )

        # print("\n=== SPECIFIC ANALYSIS EXAMPLES ===")
        #
        # # Example 1: Transfer learning with EO features only
        # print("\n1. Transfer Learning with EO features:")
        # eo_transfer_results = pipeline.run_leave_one_year_out_cv(
        #     training_countries='all',
        #     feature_type='eo',
        #     training_mode='transfer_learning'
        # )
        #
        # # Example 2: Direct training (all countries) with ERA5 features only
        # print("\n2. Direct Training (all countries) with ERA5 features:")
        # era5_direct_all_results = pipeline.run_leave_one_year_out_cv(
        #     training_countries='all',
        #     feature_type='era5',
        #     training_mode='direct_training_all_countries'
        # )
        #
        # # Example 3: Compare countries with specific feature type and training mode
        # print("\n3. Country comparison with all features and transfer learning:")
        # country_comparison = pipeline.compare_countries(
        #     country_combinations=[
        #         ['all'],
        #         ['AT', 'CZ', 'FR', 'HU', 'SI', 'SK', 'HR'],
        #         ['HU', 'SI', 'SK', 'PL', 'CZ']
        #     ],
        #     feature_type='all',
        #     training_mode='transfer_learning'
        # )
        # pipeline.plot_performance_comparison(country_comparison)
        #
        # # Plot predictions vs actual for best performing setup
        # if comprehensive_results:
        #     best_setup = min(comprehensive_results.items(), key=lambda x: x[1]['mean_rmse'])
        #     print(f"\nBest performing setup: {best_setup[0]} (RMSE: {best_setup[1]['mean_rmse']:.3f})")
        #     pipeline.plot_predictions_vs_actual(best_setup[1]['detailed_results'])
        #
        # # Summary of all setups
        # print("\n=== FINAL SUMMARY ===")
        # if comprehensive_results:
        #     print(f"Total setups tested: {len(comprehensive_results)}")
        #
        #     # Best setup overall
        #     best_overall = min(comprehensive_results.items(), key=lambda x: x[1]['mean_rmse'])
        #     print(f"Best setup: {best_overall[0]} with RMSE = {best_overall[1]['mean_rmse']:.3f}")
        #
        #     # Best by feature type
        #     print("\nBest by feature type:")
        #     for feature_type in ['all', 'eo', 'era5']:
        #         feature_results = {k: v for k, v in comprehensive_results.items()
        #                            if v['feature_type'] == feature_type}
        #         if feature_results:
        #             best_feature = min(feature_results.items(), key=lambda x: x[1]['mean_rmse'])
        #             print(f"  {feature_type.upper()}: {best_feature[0]} (RMSE: {best_feature[1]['mean_rmse']:.3f})")
        #
        #     # Best by training mode
        #     print("\nBest by training mode:")
        #     for mode in ['transfer_learning', 'direct_training', 'direct_training_all_countries']:
        #         mode_results = {k: v for k, v in comprehensive_results.items()
        #                         if v['training_mode'] == mode}
        #         if mode_results:
        #             best_mode = min(mode_results.items(), key=lambda x: x[1]['mean_rmse'])
        #             print(f"  {mode.replace('_', ' ').title()}: {best_mode[0]} (RMSE: {best_mode[1]['mean_rmse']:.3f})")
        #
        #     print(f"\nPredictions saved to: yield_predictions_results.csv")
        #     print(f"CSV contains {len(predictions_df)} rows with predictions from all {len(comprehensive_results)} model setups")
        #     pipeline.plot_performance_comparison(country_comparison)
        #
        #     # Plot predictions vs actual for best performing setup
        #     if comprehensive_results:
        #         best_setup = min(comprehensive_results.items(), key=lambda x: x[1]['mean_rmse'])
        #     print(f"\nBest performing setup: {best_setup[0]} (RMSE: {best_setup[1]['mean_rmse']:.3f})")
        #     pipeline.plot_predictions_vs_actual(best_setup[1]['detailed_results'])
        #
        #     # Summary of all setups
        #     print("\n=== FINAL SUMMARY ===")
        #     if comprehensive_results:
        #         print(f"Total setups tested: {len(comprehensive_results)}")
        #
        #     # Best setup overall
        #     best_overall = min(comprehensive_results.items(), key=lambda x: x[1]['mean_rmse'])
        #     print(f"Best setup: {best_overall[0]} with RMSE = {best_overall[1]['mean_rmse']:.3f}")
        #
        #     # Best by feature type
        #     print("\nBest by feature type:")
        #     for feature_type in ['all', 'eo', 'era5']:
        #         feature_results = {k: v for k, v in comprehensive_results.items()
        #                            if v['feature_type'] == feature_type}
        #     if feature_results:
        #         best_feature = min(feature_results.items(), key=lambda x: x[1]['mean_rmse'])
        #     print(f"  {feature_type.upper()}: {best_feature[0]} (RMSE: {best_feature[1]['mean_rmse']:.3f})")
        #
        #     # Best by training mode
        #     print("\nBest by training mode:")
        #     for mode in ['transfer_learning', 'direct_training']:
        #         mode_results = {k: v for k, v in comprehensive_results.items()
        #                         if v['training_mode'] == mode}
        #     if mode_results:
        #         best_mode = min(mode_results.items(), key=lambda x: x[1]['mean_rmse'])
        #     print(f"  {mode.replace('_', ' ').title()}: {best_mode[0]} (RMSE: {best_mode[1]['mean_rmse']:.3f})")