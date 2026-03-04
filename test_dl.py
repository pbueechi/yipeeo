import datetime
import random
import os
import csv
import itertools
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2

import matplotlib.pyplot as plt


seed_value = 1

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

class LSTMPredictor:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.history = None

    def load_and_prepare_data(self, csv_file_path):
        """
        Load CSV data and reshape it for LSTM input
        Expected format: predictor1_LT1, predictor1_LT2, ..., predictor2_LT1, ..., effort, year
        """
        # Load the data
        df = pd.read_csv(csv_file_path)
        # Too little data in Hungary, Croatia, Slovakia and Slovenia -> merge SK and HU as well as HR and SL
        di = {'HR': 'HI', 'HU': 'UK', 'SI': 'HI', 'SK': 'UK'}
        df.loc[:, 'country'] = [di[a[:2]] if a[:2] in di.keys() else a[:2] for a in df.field_id]

        # Find all predictor columns that follow the pattern *_LT[1-8]
        predictor_cols = [col for col in df.columns if
                          col.endswith(('_LT1', '_LT2', '_LT3', '_LT4', '_LT5', '_LT6', '_LT7', '_LT8'))]
        predictor_cols.sort()  # Sort to ensure consistent ordering

        # Group predictors by their base names
        predictor_groups = {}
        for col in predictor_cols:
            base_name = col.rsplit('_LT', 1)[0]
            timestep = int(col.rsplit('_LT', 1)[1])
            if base_name not in predictor_groups:
                predictor_groups[base_name] = {}
            predictor_groups[base_name][timestep] = col

        print(f"Found {len(predictor_groups)} predictor variables:")
        for pred_name in predictor_groups.keys():
            print(f"  - {pred_name}")

        # Check required columns
        if 'yield_anom' not in df.columns:
            raise ValueError("Target column 'yield_anom' not found in the dataset")
        if 'c_year' not in df.columns:
            raise ValueError("Year column 'c_year' not found in the dataset")

        # Organize data for LSTM format
        n_samples = len(df)
        n_predictors = len(predictor_groups)
        n_timesteps = 8

        # Initialize X array: (n_samples, timesteps, features)
        X = np.zeros((n_samples, n_timesteps, n_predictors))

        # Fill X with predictor values
        for pred_idx, (pred_name, timestep_cols) in enumerate(predictor_groups.items()):
            for timestep in range(1, 9):
                if timestep in timestep_cols:
                    col_name = timestep_cols[timestep]
                    X[:, timestep - 1, pred_idx] = df[col_name].values

        # Extract target and year
        y = df['yield_anom'].values.reshape(-1, 1)
        years = df['c_year'].values
        countries = df['country'].values

        print(f"Data shape - X: {X.shape}, y: {y.shape}")
        print(f"Years range: {years.min()} to {years.max()}")
        print(X.shape)
        print(y.shape)
        return X, y, years, countries, list(predictor_groups.keys())

    def leave_one_year_out_cv(self, X, y, years):
        """
        Perform leave-one-year-out cross-validation
        """
        unique_years = np.sort(np.unique(years))
        print(f"Performing LOYO CV with {len(unique_years)} years: {unique_years}")

        cv_results = []
        fold_predictions = []
        fold_actuals = []

        for test_year in unique_years:
            print(f"\nFold: Testing on year {test_year}")

            # Split data by year
            train_mask = years != test_year
            test_mask = years == test_year

            X_train_fold = X[train_mask]
            y_train_fold = y[train_mask]
            X_test_fold = X[test_mask]
            y_test_fold = y[test_mask]

            print(f"  Train samples: {X_train_fold.shape[0]}, Test samples: {X_test_fold.shape[0]}")

            # Create separate scalers for this fold
            scaler_X_fold = StandardScaler()
            scaler_y_fold = StandardScaler()

            # Scale the target variable
            y_train_scaled = scaler_y_fold.fit_transform(y_train_fold)
            y_test_scaled = scaler_y_fold.transform(y_test_fold)

            # Scale the features (reshape for scaling, then reshape back)
            X_train_reshaped = X_train_fold.reshape(-1, X_train_fold.shape[-1])
            X_test_reshaped = X_test_fold.reshape(-1, X_test_fold.shape[-1])

            X_train_scaled = scaler_X_fold.fit_transform(X_train_reshaped)
            X_test_scaled = scaler_X_fold.transform(X_test_reshaped)

            # Reshape back to LSTM format
            X_train_scaled = X_train_scaled.reshape(X_train_fold.shape)
            X_test_scaled = X_test_scaled.reshape(X_test_fold.shape)

            # Split training data for validation (use 20% of training data)
            train_indices = np.arange(len(X_train_scaled))
            train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=42)

            X_train_final = X_train_scaled[train_idx]
            y_train_final = y_train_scaled[train_idx]
            X_val = X_train_scaled[val_idx]
            y_val = y_train_scaled[val_idx]

            # Build and train model for this fold
            model = self.build_model(
                timesteps=X.shape[1],
                features=X.shape[2]
            )

            # Set up early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )

            # Train the model
            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )

            # Make predictions
            predictions_scaled = model.predict(X_test_scaled, verbose=0)
            predictions = scaler_y_fold.inverse_transform(predictions_scaled)
            y_test_original = scaler_y_fold.inverse_transform(y_test_scaled)

            # Calculate metrics for this fold
            mse = mean_squared_error(y_test_original, predictions)
            mae = mean_absolute_error(y_test_original, predictions)
            r2 = r2_score(y_test_original, predictions)
            r = pearsonr(y_test_original.ravel(), predictions.ravel())[0]

            fold_result = {
                'test_year': test_year,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'n_train': len(X_train_fold),
                'n_test': len(X_test_fold)
            }

            cv_results.append(fold_result)
            fold_predictions.extend(predictions.flatten())
            fold_actuals.extend(y_test_original.flatten())

            print(f"  Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, R: {r:.3f}")

        # Calculate overall CV metrics
        overall_mse = np.mean([r['mse'] for r in cv_results])
        overall_mae = np.mean([r['mae'] for r in cv_results])
        overall_r2 = np.mean([r['r2'] for r in cv_results])

        std_mse = np.std([r['mse'] for r in cv_results])
        std_mae = np.std([r['mae'] for r in cv_results])
        std_r2 = np.std([r['r2'] for r in cv_results])

        print(f"\n{'=' * 50}")
        print(f"LEAVE-ONE-YEAR-OUT CROSS-VALIDATION RESULTS")
        print(f"{'=' * 50}")
        print(f"Mean MSE: {overall_mse:.4f} ± {std_mse:.4f}")
        print(f"Mean MAE: {overall_mae:.4f} ± {std_mae:.4f}")
        print(f"Mean R²:  {overall_r2:.4f} ± {std_r2:.4f}")

        return {
            'fold_results': cv_results,
            'overall_metrics': {
                'mse_mean': overall_mse, 'mse_std': std_mse,
                'mae_mean': overall_mae, 'mae_std': std_mae,
                'r2_mean': overall_r2, 'r2_std': std_r2
            },
            'all_predictions': np.array(fold_predictions),
            'all_actuals': np.array(fold_actuals)
        }

    def build_model(self, timesteps=8, features=1, lstm_units=50, dropout_rate=0.2, orig=False):
        """
        Build the LSTM model
        """
        if orig:
            model = Sequential([
                LSTM(lstm_units, return_sequences=True, input_shape=(timesteps, features)),
                # LayerNormalization(),
                Dropout(dropout_rate),
                LSTM(lstm_units//2, return_sequences=False),
                # LayerNormalization(),
                Dropout(dropout_rate),
                Dense(25, activation='relu'),
                # Dropout(dropout_rate*0.5),
                Dense(1, activation=None)
            ])

            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae'],
            )
        else:
            model = Sequential([
                # First bidirectional LSTM layer
                Bidirectional(LSTM(lstm_units, return_sequences=True,
                                   kernel_regularizer=l2(0.001),
                                   recurrent_regularizer=l2(0.001)),
                              input_shape=(timesteps, features)),
                LayerNormalization(),  # Better than BatchNorm for sequences
                Dropout(dropout_rate),

                # Second bidirectional LSTM layer
                Bidirectional(LSTM(lstm_units // 2, return_sequences=False,
                                   kernel_regularizer=l2(0.001),
                                   recurrent_regularizer=l2(0.001))),
                LayerNormalization(),
                Dropout(dropout_rate),

                # Dense layers with improved structure
                # Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                # Dropout(dropout_rate * 0.5),  # Lower dropout for dense layers
                Dense(25, activation='relu'),
                Dense(1, activation=None)  # No activation for regression
            ])

            # Improved optimizer configuration
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
        return model

    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                    epochs=100, batch_size=32, patience=5):
        """
        Train the LSTM model
        """
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True
        )

        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        return self.history

    def predict(self, X):
        """
        Make predictions and inverse transform
        """
        predictions_scaled = self.model.predict(X)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        """
        predictions = self.predict(X_test)
        y_test_original = self.scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_test_original, predictions)
        mae = mean_absolute_error(y_test_original, predictions)
        r2 = r2_score(y_test_original, predictions)

        print(f"Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'actual': y_test_original
        }

    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_cv_results(self, cv_results):
        """
        Plot cross-validation results
        """
        fold_results = cv_results['fold_results']
        years = [r['test_year'] for r in fold_results]
        mse_values = [r['mse'] for r in fold_results]
        mae_values = [r['mae'] for r in fold_results]
        r2_values = [r['r2'] for r in fold_results]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # MSE by year
        axes[0, 0].bar(years, mse_values, alpha=0.7, color='red')
        axes[0, 0].set_title('MSE by Test Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # MAE by year
        axes[0, 1].bar(years, mae_values, alpha=0.7, color='orange')
        axes[0, 1].set_title('MAE by Test Year')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # R² by year
        axes[1, 0].bar(years, r2_values, alpha=0.7, color='green')
        axes[1, 0].set_title('R² by Test Year')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Overall predictions vs actuals
        all_predictions = cv_results['all_predictions']
        all_actuals = cv_results['all_actuals']

        axes[1, 1].scatter(all_actuals, all_predictions, alpha=0.6)
        axes[1, 1].plot([all_actuals.min(), all_actuals.max()],
                        [all_actuals.min(), all_actuals.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('All CV Predictions vs Actual Values')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print detailed results table
        print(f"\nDetailed Results by Year:")
        print(f"{'Year':<6} {'MSE':<8} {'MAE':<8} {'R²':<8} {'N_Train':<8} {'N_Test':<8}")
        print("-" * 50)
        for result in fold_results:
            print(
                f"{result['test_year']:<6} {result['mse']:<8.4f} {result['mae']:<8.4f} {result['r2']:<8.4f} {result['n_train']:<8} {result['n_test']:<8}")

    def transfer_learning_cv(self, X, y, years, countries, predictors, crop, path_out, target_country='UA',
                             lstm_units=50, dropout_rate=0.4, tl_train_layers=2, early_stopping_patience=10):
        """
        Perform transfer learning with leave-one-year-out cross-validation
        1. Train base model on all countries except target_country
        2. Fine-tune on target_country data (excluding test year)
        3. Test on target_country data from test year only
        """
        if not os.path.exists(os.path.join(path_out, f'configs_{crop}_{target_country}.txt')):
            with open(os.path.join(path_out, f'configs_{crop}_{target_country}.txt'), 'w') as file:
                file.write(f'used configs: lstm_units={lstm_units}\n dropout_rate={dropout_rate}\n tl_train_layers='
                           f'{tl_train_layers}\n early_stopping_patience={early_stopping_patience}')

        epochs = 150
        batch_size = 32

        # Get unique years from target country
        target_mask = countries == target_country
        target_years = np.unique(years[target_mask])
        target_years = np.sort(target_years)
        target_years = [y for y in target_years if y>2006]

        i_eo = [predictors.index(a) for a in ['VODCA_CXKu', 'evi_median', 'sm', 'year']]
        i_met = [predictors.index(a) for a in ['evavt', 'pev', 'ssr', 'swvl1', 't2m', 'tp', 'year']]

        X_eo, X_met = X[:, :, i_eo], X[:, :, i_met]

        print(f"Transfer Learning Setup:")
        print(f"Available years for {target_country}: {target_years}")
        print(f"Total samples in {target_country}: {np.sum(target_mask)}")


        cv_results = []
        fold_predictions = []
        fold_actuals = []

        ret_vals = pd.DataFrame(columns=['year', 'region', 'forecasted', 'forecasted_tl', 'forecasted_no_tl',
                                         'forecasted_all_tl', 'forecasted_eo', 'forecasted_met', 'observed'])

        for test_year in target_years:
            print(f"\n{'=' * 60}")
            print(f"FOLD: Testing on {target_country} data from year {test_year}")
            print(f"{'=' * 60}")

            # Prepare base training data (all countries except target)
            train_country_mask = countries != target_country
            train_year_mask = years != test_year
            test_country_mask = countries == target_country
            test_year_mask = years == test_year

            train_mask = train_country_mask & train_year_mask
            test_mask = test_country_mask & test_year_mask
            finetune_mask = test_country_mask & train_year_mask
            train_no_ft_mask = train_year_mask

            X_base, y_base = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            X_finetune, y_finetune = X[finetune_mask], y[finetune_mask]
            X_base_all, y_base_all = X[train_no_ft_mask], y[train_no_ft_mask]
            X_base_met, X_base_eo = X_met[train_mask], X_eo[train_mask]
            X_finetune_met, X_finetune_eo = X_met[finetune_mask], X_eo[finetune_mask]
            X_test_met, X_test_eo = X_met[test_mask], X_eo[test_mask]

            print(f"Base training samples: {X_base.shape[0]}")
            print(f"Fine-tuning samples: {X_finetune.shape[0]}")
            print(f"Test samples: {X_test.shape[0]}")

            if X_test.shape[0] == 0:
                print(f"No test data for {target_country} in year {test_year}, skipping...")
                continue

            # Step 1: Train base model on all countries except target
            print("\nStep 1: Training base model...")
            base_scaler_X = StandardScaler()
            base_scaler_X_eo = StandardScaler()
            base_scaler_X_met = StandardScaler()
            base_scaler_X_all = StandardScaler()
            base_scaler_y = StandardScaler()
            base_scaler_y_all = StandardScaler()

            # Scale base training data
            y_base_scaled = base_scaler_y.fit_transform(y_base)
            y_base_all_scaled = base_scaler_y_all.fit_transform(y_base_all)
            base_scaler_X, X_base_scaled = scale_x(scaler=base_scaler_X, X=X_base, fit=True)
            base_scaler_X_eo, X_base_eo_scaled = scale_x(scaler=base_scaler_X_eo, X=X_base_eo, fit=True)
            base_scaler_X_met, X_base_met_scaled = scale_x(scaler=base_scaler_X_met, X=X_base_met, fit=True)
            base_scaler_X_all, X_base_all_scaled = scale_x(scaler=base_scaler_X_all, X=X_base_all, fit=True)

            X_base_train, y_base_train, X_base_val, y_base_val = train_val_split(X_base_scaled, y_base_scaled)
            X_base_train_eo, _, X_base_val_eo, _ = train_val_split(X_base_eo_scaled, y_base_all_scaled)
            X_base_train_met, _, X_base_val_met, _ = train_val_split(X_base_met_scaled, y_base_all_scaled)
            X_base_train_all, y_base_train_all, X_base_val_all, y_base_val_all = train_val_split(X_base_all_scaled, y_base_all_scaled)

            # Build and train base model
            base_model = self.build_model(timesteps=X.shape[1], features=X.shape[2], lstm_units=lstm_units, dropout_rate=dropout_rate)
            base_model_eo = self.build_model(timesteps=X_base_eo.shape[1], features=X_base_eo.shape[2], lstm_units=lstm_units, dropout_rate=dropout_rate)
            base_model_met = self.build_model(timesteps=X_base_met.shape[1], features=X_base_met.shape[2], lstm_units=lstm_units, dropout_rate=dropout_rate)
            base_model_all = self.build_model(timesteps=X_base_all.shape[1], features=X_base_all.shape[2], lstm_units=lstm_units, dropout_rate=dropout_rate)

            early_stopping_base = EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )

            base_history = base_model.fit(X_base_train, y_base_train, validation_data=(X_base_val, y_base_val),
                epochs=epochs, batch_size=batch_size, callbacks=[early_stopping_base], verbose=0)
            print("\nStep 1: Training eo model...")
            base_history_eo = base_model_eo.fit(X_base_train_eo, y_base_train, validation_data=(X_base_val_eo, y_base_val),
                                          epochs=epochs, batch_size=batch_size, callbacks=[early_stopping_base], verbose=0)
            print("\nStep 1: Training met model...")
            base_history_met = base_model_met.fit(X_base_train_met, y_base_train, validation_data=(X_base_val_met, y_base_val),
                                          epochs=epochs, batch_size=batch_size, callbacks=[early_stopping_base], verbose=0)
            print("\nStep 1: Training all model...")
            base_history_all = base_model_all.fit(X_base_train_all, y_base_train_all, validation_data=(X_base_val_all, y_base_val_all),
                                          epochs=epochs, batch_size=batch_size, callbacks=[early_stopping_base], verbose=0)

            # Scale test data using base scalers
            y_test_scaled = base_scaler_y.transform(y_test)
            _, X_test_scaled = scale_x(scaler=base_scaler_X, X=X_test, fit=False)
            _, X_test_scaled_eo = scale_x(scaler=base_scaler_X_eo, X=X_test_eo, fit=False)
            _, X_test_scaled_met = scale_x(scaler=base_scaler_X_met, X=X_test_met, fit=False)


            # Make predictions
            predictions_scaled_nft = base_model.predict(X_test_scaled, verbose=0)
            predictions_nft = base_scaler_y.inverse_transform(predictions_scaled_nft)
            y_test_original_nft = base_scaler_y.inverse_transform(y_test_scaled)

            # Make predictions train
            predictions_scaled_nft_train = base_model.predict(X_base_train, verbose=0)
            predictions_nft_train = base_scaler_y.inverse_transform(predictions_scaled_nft_train)
            y_train_original_nft = base_scaler_y.inverse_transform(y_base_train)

            # # Make predictions all
            predictions_scaled_all = base_model_all.predict(X_test_scaled, verbose=0)
            predictions_all = base_scaler_y.inverse_transform(predictions_scaled_all)

            # Calculate metrics for this fold
            mse_nft = mean_squared_error(y_test_original_nft, predictions_nft)
            mae_nft = mean_absolute_error(y_test_original_nft, predictions_nft)
            r2_nft = r2_score(y_test_original_nft, predictions_nft)
            r_nft = pearsonr(y_test_original_nft.ravel(), predictions_nft.ravel())[0]
            mbias_nft = np.median(y_test_original_nft-predictions_nft)

            # Calculate metrics for this fold train
            mse_nft_train = mean_squared_error(y_train_original_nft, predictions_nft_train)
            mae_nft_train = mean_absolute_error(y_train_original_nft, predictions_nft_train)
            r2_nft_train = r2_score(y_train_original_nft, predictions_nft_train)
            r_nft_train = pearsonr(y_train_original_nft.ravel(), predictions_nft_train.ravel())[0]
            mbias_nft_train = np.median(y_train_original_nft - predictions_nft_train)

            print(f"Base model trained. Final val_loss: {min(base_history.history['val_loss']):.4f}")
            print(f"  Results - MSE: {mse_nft:.4f}, MAE: {mae_nft:.4f}, R²: {r2_nft:.4f}, R: {r_nft:.3f}, med bias: {mbias_nft:.3f}")
            print(f"  Results train - MSE: {mse_nft_train:.4f}, MAE: {mae_nft_train:.4f}, R²: {r2_nft_train:.4f}, "
                  f"R: {r_nft_train:.3f}, med bias: {mbias_nft_train:.3f}")

            # Step 2: Fine-tune on target country data (if available)
            if X_finetune.shape[0] > 0:
                print(f"\nStep 2: Fine-tuning on {target_country} data...")

                # Scale fine-tuning data using base scalers (important for transfer learning)
                y_finetune_scaled = base_scaler_y.transform(y_finetune)
                _, X_finetune_scaled = scale_x(scaler=base_scaler_X, X=X_finetune, fit=False)
                _, X_finetune_scaled_met = scale_x(scaler=base_scaler_X_met, X=X_finetune_met, fit=False)
                _, X_finetune_scaled_eo = scale_x(scaler=base_scaler_X_eo, X=X_finetune_eo, fit=False)

                if not tl_train_layers=='all':
                    if tl_train_layers>len(base_model.layers):
                        tl_train_layers=len(base_model.layers)
                    for i in range(len(base_model.layers)-tl_train_layers):
                        base_model.layers[i].trainable = False
                        base_model_met.layers[i].trainable = False
                        base_model_eo.layers[i].trainable = False
                        base_model_all.layers[i].trainable = False

                # Split fine-tuning data for validation (if enough samples)
                if X_finetune_scaled.shape[0] > 5:
                    X_ft_train, y_ft_train, X_ft_val, y_ft_val = train_val_split(X_finetune_scaled, y_finetune_scaled)
                    validation_data_ft = (X_ft_val, y_ft_val)
                    X_ft_train_eo, y_ft_train_eo, X_ft_val_eo, y_ft_val_eo = train_val_split(X_finetune_scaled_eo, y_finetune_scaled)
                    validation_data_ft_eo = (X_ft_val_eo, y_ft_val_eo)
                    X_ft_train_met, y_ft_train_met, X_ft_val_met, y_ft_val_met = train_val_split(X_finetune_scaled_met, y_finetune_scaled)
                    validation_data_ft_met = (X_ft_val_met, y_ft_val_met)
                    monitor_ft = 'val_loss'
                else:
                    # Use all data for training if too few samples
                    X_ft_train, y_ft_train = X_finetune_scaled, y_finetune_scaled
                    X_ft_train_eo, y_ft_train_eo = X_finetune_scaled_eo, y_finetune_scaled
                    X_ft_train_met, y_ft_train_met = X_finetune_scaled_met, y_finetune_scaled
                    validation_data_ft = None
                    monitor_ft = 'loss'

                # Fine-tune with lower learning rate
                base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
                base_model_eo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
                base_model_met.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
                base_model_all.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

                early_stopping_ft = EarlyStopping(monitor=monitor_ft, patience=early_stopping_patience-5, restore_best_weights=True, verbose=0)

                ft_history = base_model.fit(
                    X_ft_train, y_ft_train,
                    validation_data=validation_data_ft,
                    epochs=int(epochs*0.5),
                    batch_size=int(batch_size*0.5),  # Smaller batch size for fine-tuning
                    callbacks=[early_stopping_ft],
                    verbose=0
                )
                base_model_eo.fit(
                    X_ft_train_eo, y_ft_train,
                    validation_data=validation_data_ft_eo,
                    epochs=int(epochs*0.5),
                    batch_size=int(batch_size*0.5),  # Smaller batch size for fine-tuning
                    callbacks=[early_stopping_ft],
                    verbose=0
                )
                base_model_met.fit(
                    X_ft_train_met, y_ft_train,
                    validation_data=validation_data_ft_met,
                    epochs=int(epochs*0.5),
                    batch_size=int(batch_size*0.5),  # Smaller batch size for fine-tuning
                    callbacks=[early_stopping_ft],
                    verbose=0
                )
                base_model_all.fit(
                    X_ft_train, y_ft_train,
                    validation_data=validation_data_ft,
                    epochs=int(epochs*0.5),
                    batch_size=int(batch_size*0.5),  # Smaller batch size for fine-tuning
                    callbacks=[early_stopping_ft],
                    verbose=0
                )

                print(f"Fine-tuning completed. Final loss: {ft_history.history['loss'][-1]:.4f}")
            else:
                print(f"\nStep 2: No fine-tuning data available for {target_country}")

            # Step 3: Test on target country test year
            print(f"\nStep 3: Testing on {target_country} year {test_year}...")

            # Make predictions
            predictions_scaled = base_model.predict(X_test_scaled, verbose=0)
            predictions = base_scaler_y.inverse_transform(predictions_scaled)
            y_test_original = base_scaler_y.inverse_transform(y_test_scaled)

            # Make predictions eo
            predictions_scaled_eo = base_model_eo.predict(X_test_scaled_eo, verbose=0)
            predictions_eo = base_scaler_y.inverse_transform(predictions_scaled_eo)

            # Make predictions met
            predictions_scaled_met = base_model_met.predict(X_test_scaled_met, verbose=0)
            predictions_met = base_scaler_y.inverse_transform(predictions_scaled_met)

            # Make predictions all_ft
            predictions_scaled_all_ft = base_model_all.predict(X_test_scaled, verbose=0)
            predictions_all_ft = base_scaler_y.inverse_transform(predictions_scaled_all_ft)

            # Calculate metrics for this fold
            mse = mean_squared_error(y_test_original, predictions)
            mae = mean_absolute_error(y_test_original, predictions)
            r2 = r2_score(y_test_original, predictions)
            mbias = np.median(y_test_original - predictions)
            r = pearsonr(y_test_original.ravel(), predictions.ravel())[0]

            fold_result = {
                'test_year': test_year,
                'target_country': target_country,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'n_base_train': len(X_base),
                'n_finetune': len(X_finetune) if X_finetune.shape[0] > 0 else 0,
                'n_test': len(X_test)
            }

            cv_results.append(fold_result)
            fold_predictions.extend(predictions.flatten())
            fold_actuals.extend(y_test_original.flatten())

            print(f"  Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, R: {r:.3f}, med bias: {mbias:.3f}")

            dict = {'year': [test_year] * len(y_test_original),
                    'region': [target_country] * len(y_test_original),
                    'observed': y_test_original.ravel(),
                    'forecasted': predictions_nft.ravel(),
                    'forecasted_no_tl': predictions_all.ravel(),
                    'forecasted_all_tl': predictions_all_ft.ravel(),
                    'forecasted_tl': predictions.ravel(),
                    'forecasted_eo': predictions_eo.ravel(),
                    'forecasted_met': predictions_met.ravel(),
                    # 'forecasted_xgb': predictions,
                    # 'forecasted_xgb_all': predictions
                    }

            df = pd.DataFrame(dict)
            ret_vals = pd.concat([ret_vals, df])
        ret_vals.to_csv(os.path.join(path_out, f'{crop}_{target_country}_1.csv'))
        # Calculate overall CV metrics
        if cv_results:
            overall_mse = np.mean([r['mse'] for r in cv_results])
            overall_mae = np.mean([r['mae'] for r in cv_results])
            overall_r2 = np.mean([r['r2'] for r in cv_results])

            std_mse = np.std([r['mse'] for r in cv_results])
            std_mae = np.std([r['mae'] for r in cv_results])
            std_r2 = np.std([r['r2'] for r in cv_results])

            print(f"\n{'=' * 70}")
            print(f"TRANSFER LEARNING CROSS-VALIDATION RESULTS ({target_country})")
            print(f"{'=' * 70}")
            print(f"Mean MSE: {overall_mse:.4f} ± {std_mse:.4f}")
            print(f"Mean MAE: {overall_mae:.4f} ± {std_mae:.4f}")
            print(f"Mean R²:  {overall_r2:.4f} ± {std_r2:.4f}")
            print(f"Number of test years: {len(cv_results)}")
        else:
            print("No valid test results obtained.")
            overall_mse = overall_mae = overall_r2 = std_mse = std_mae = std_r2 = 0

        return {
            'fold_results': cv_results,
            'overall_metrics': {
                'mse_mean': overall_mse, 'mse_std': std_mse,
                'mae_mean': overall_mae, 'mae_std': std_mae,
                'r2_mean': overall_r2, 'r2_std': std_r2
            },
            'all_predictions': np.array(fold_predictions),
            'all_actuals': np.array(fold_actuals),
            'target_country': target_country
        }

    def hyper_tune(self, X, y, years, countries, crop, path_out, hp_settings, target_country='UA'):
        """
        Perform transfer learning with leave-one-year-out cross-validation
        1. Train base model on all countries except target_country
        2. Fine-tune on target_country data (excluding test year)
        3. Test on target_country data from test year only
        """
        file_path = os.path.join(path_out, f'hp_tune_{crop}_{target_country}.csv')
        row_head = ['year', 'l_unit', 'do_rate', 'esp', 'tl_tr_layer', 'train_r2', 'test_r2', 'train_mae', 'test_mae',
                    'train_bias', 'test_bias', 'train_r', 'test_r', 'train_ft_r2', 'test_ft_r2', 'train_ft_mae',
                    'test_ft_mae', 'train_ft_bias', 'test_ft_bias', 'train_ft_r', 'test_ft_r']

        epochs = 150
        batch_size = 32
        hps = [hp_settings[a] for a in hp_settings.keys()]
        hp_combs = list(itertools.product(*hps))

        if not os.path.exists(path_out): os.makedirs(path_out)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(row_head)

        # Get unique years from target country
        target_mask = countries == target_country
        target_years = np.unique(years[target_mask])
        target_years = np.sort(target_years)

        print(f"Transfer Learning Setup:")
        print(f"Target country: {target_country}")
        print(f"Available years for {target_country}: {target_years}")
        print(f"Total samples in {target_country}: {np.sum(target_mask)}")

        for test_year in [2008, 2016]:
            print(f"\n{'=' * 60}")
            print(f"FOLD: Testing on {target_country} data from year {test_year}")
            print(f"{'=' * 60}")

            # Prepare base training data (all countries except target)
            train_country_mask = countries != target_country
            train_year_mask = years != test_year
            test_country_mask = countries == target_country
            test_year_mask = years == test_year

            train_mask = train_country_mask & train_year_mask
            test_mask = test_country_mask & test_year_mask
            finetune_mask = test_country_mask & train_year_mask

            X_base, y_base = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            X_finetune, y_finetune = X[finetune_mask], y[finetune_mask]

            print(f"Base training samples: {X_base.shape[0]}")
            print(f"Fine-tuning samples: {X_finetune.shape[0]}")
            print(f"Test samples: {X_test.shape[0]}")

            if (X_base.shape[0] > 100) and (X_finetune.shape[0] > 10) and (X_test.shape[0] > 10):
                # Step 1: Train base model on all countries except target
                print("\nStep 1: Training base model...")
                base_scaler_X = StandardScaler()
                base_scaler_y = StandardScaler()

                # Scale base training data
                y_base_scaled = base_scaler_y.fit_transform(y_base)
                base_scaler_X, X_base_scaled = scale_x(scaler=base_scaler_X, X=X_base, fit=True)

                X_base_train, y_base_train, X_base_val, y_base_val = train_val_split(X_base_scaled, y_base_scaled)

                # Build and train base model
                for hps in hp_combs:
                    lstm_units, dropout_rate, tl_train_layers, early_stopping_patience = hps

                    base_model = self.build_model(timesteps=X.shape[1], features=X.shape[2], lstm_units=lstm_units, dropout_rate=dropout_rate)

                    early_stopping_base = EarlyStopping(
                        monitor='val_loss',
                        patience=early_stopping_patience,
                        restore_best_weights=True,
                        verbose=0
                    )

                    base_history = base_model.fit(X_base_train, y_base_train, validation_data=(X_base_val, y_base_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping_base], verbose=0)

                    # Scale test data using base scalers
                    y_test_scaled = base_scaler_y.transform(y_test)
                    _, X_test_scaled = scale_x(scaler=base_scaler_X, X=X_test, fit=False)

                    # Make predictions
                    predictions_scaled_nft = base_model.predict(X_test_scaled, verbose=0)
                    predictions_nft = base_scaler_y.inverse_transform(predictions_scaled_nft)
                    y_test_original_nft = base_scaler_y.inverse_transform(y_test_scaled)

                    # Make predictions train
                    predictions_scaled_nft_train = base_model.predict(X_base_train, verbose=0)
                    predictions_nft_train = base_scaler_y.inverse_transform(predictions_scaled_nft_train)
                    y_train_original_nft = base_scaler_y.inverse_transform(y_base_train)

                    # Calculate metrics for this fold
                    mae_nft = mean_absolute_error(y_test_original_nft, predictions_nft)
                    r2_nft = r2_score(y_test_original_nft, predictions_nft)
                    r_nft = pearsonr(y_test_original_nft.ravel(), predictions_nft.ravel())[0]
                    mbias_nft = np.median(y_test_original_nft-predictions_nft)

                    # Calculate metrics for this fold train
                    mae_nft_train = mean_absolute_error(y_train_original_nft, predictions_nft_train)
                    r2_nft_train = r2_score(y_train_original_nft, predictions_nft_train)
                    r_nft_train = pearsonr(y_train_original_nft.ravel(), predictions_nft_train.ravel())[0]
                    mbias_nft_train = np.median(y_train_original_nft - predictions_nft_train)

                    # Step 2: Fine-tune on target country data (if available)
                    if X_finetune.shape[0] > 0:
                        print(f"\nStep 2: Fine-tuning on {target_country} data...")

                        # Scale fine-tuning data using base scalers (important for transfer learning)
                        y_finetune_scaled = base_scaler_y.transform(y_finetune)
                        _, X_finetune_scaled = scale_x(scaler=base_scaler_X, X=X_finetune, fit=False)

                        if not tl_train_layers=='all':
                            if tl_train_layers>len(base_model.layers):
                                tl_train_layers=len(base_model.layers)
                            for i in range(len(base_model.layers)-tl_train_layers):
                                base_model.layers[i].trainable = False

                        # Split fine-tuning data for validation (if enough samples)
                        if X_finetune_scaled.shape[0] > 5:
                            X_ft_train, y_ft_train, X_ft_val, y_ft_val = train_val_split(X_finetune_scaled, y_finetune_scaled)
                            validation_data_ft = (X_ft_val, y_ft_val)
                            monitor_ft = 'val_loss'
                        else:
                            # Use all data for training if too few samples
                            X_ft_train, y_ft_train = X_finetune_scaled, y_finetune_scaled
                            validation_data_ft = None
                            monitor_ft = 'loss'

                        # Fine-tune with lower learning rate
                        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

                        early_stopping_ft = EarlyStopping(monitor=monitor_ft, patience=early_stopping_patience-5, restore_best_weights=True, verbose=0)

                        ft_history = base_model.fit(
                            X_ft_train, y_ft_train,
                            validation_data=validation_data_ft,
                            epochs=int(epochs*0.5),
                            batch_size=int(batch_size*0.5),  # Smaller batch size for fine-tuning
                            callbacks=[early_stopping_ft],
                            verbose=0
                        )
                        print(f"Fine-tuning completed. Final loss: {ft_history.history['loss'][-1]:.4f}")
                    else:
                        print(f"\nStep 2: No fine-tuning data available for {target_country}")

                    # Step 3: Test on target country test year
                    print(f"\nStep 3: Testing on {target_country} year {test_year}...")

                    # Make predictions
                    predictions_scaled = base_model.predict(X_test_scaled, verbose=0)
                    predictions = base_scaler_y.inverse_transform(predictions_scaled)
                    y_test_original = base_scaler_y.inverse_transform(y_test_scaled)

                    # Make predictions train_after fine_tune
                    predictions_scaled_nft_train = base_model.predict(X_base_train, verbose=0)
                    predictions_nft_train = base_scaler_y.inverse_transform(predictions_scaled_nft_train)
                    y_train_original_nft = base_scaler_y.inverse_transform(y_base_train)

                    # Calculate metrics for this fold
                    mae_ft = mean_absolute_error(y_test_original, predictions)
                    r2_ft = r2_score(y_test_original, predictions)
                    mbias_ft = np.median(y_test_original - predictions)
                    r_ft = pearsonr(y_test_original.ravel(), predictions.ravel())[0]

                    # Calculate metrics for this fold train
                    mae_ft_train = mean_absolute_error(y_train_original_nft, predictions_nft_train)
                    r2_ft_train = r2_score(y_train_original_nft, predictions_nft_train)
                    r_ft_train = pearsonr(y_train_original_nft.ravel(), predictions_nft_train.ravel())[0]
                    mbias_ft_train = np.median(y_train_original_nft - predictions_nft_train)

                    print(f"  Results - MAE: {mae_ft:.4f}, R²: {r2_ft:.4f}, R: {r_ft:.3f}, med bias: {mbias_ft:.3f}")

                    resi = [test_year, lstm_units, dropout_rate, early_stopping_patience, tl_train_layers, r2_nft_train,
                            r2_nft, mae_nft_train, mae_nft, mbias_nft_train, mbias_nft, r_nft_train, r_nft,
                            r2_ft_train, r2_ft, mae_ft_train, mae_ft, mbias_ft_train, mbias_ft, r_ft_train, r_ft]

                    with open(file_path, 'a') as f1:
                        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                        writer.writerow(resi)
                    base_model = []
            else:
                print(f'not sufficient data available in {test_year}')


    def plot_transfer_learning_results(self, cv_results):
        """
        Plot transfer learning cross-validation results
        """
        fold_results = cv_results['fold_results']
        target_country = cv_results['target_country']

        if not fold_results:
            print("No results to plot.")
            return

        years = [r['test_year'] for r in fold_results]
        mse_values = [r['mse'] for r in fold_results]
        mae_values = [r['mae'] for r in fold_results]
        r2_values = [r['r2'] for r in fold_results]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Transfer Learning Results - Target Country: {target_country}', fontsize=16)

        # MSE by year
        axes[0, 0].bar(years, mse_values, alpha=0.7, color='red')
        axes[0, 0].set_title('MSE by Test Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # MAE by year
        axes[0, 1].bar(years, mae_values, alpha=0.7, color='orange')
        axes[0, 1].set_title('MAE by Test Year')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # R² by year
        axes[1, 0].bar(years, r2_values, alpha=0.7, color='green')
        axes[1, 0].set_title('R² by Test Year')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Overall predictions vs actuals
        all_predictions = cv_results['all_predictions']
        all_actuals = cv_results['all_actuals']

        if len(all_predictions) > 0:
            axes[1, 1].scatter(all_actuals, all_predictions, alpha=0.6, color='blue')
            axes[1, 1].plot([all_actuals.min(), all_actuals.max()],
                            [all_actuals.min(), all_actuals.max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('Actual Values')
            axes[1, 1].set_ylabel('Predicted Values')
            axes[1, 1].set_title(f'All Predictions vs Actual ({target_country})')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print detailed results table
        print(f"\nDetailed Transfer Learning Results for {target_country}:")
        print(f"{'Year':<6} {'MSE':<8} {'MAE':<8} {'R²':<8} {'N_Base':<8} {'N_FT':<8} {'N_Test':<8}")
        print("-" * 60)
        for result in fold_results:
            print(f"{result['test_year']:<6} {result['mse']:<8.4f} {result['mae']:<8.4f} "
                  f"{result['r2']:<8.4f} {result['n_base_train']:<8} {result['n_finetune']:<8} {result['n_test']:<8}")

        print(f"\nN_Base: Base training samples, N_FT: Fine-tuning samples, N_Test: Test samples")


# class ANNPredictor:
#     def __init__(self):
#         self.model = None
#         self.history = None
#
#     def load_and_prepare_data(self, csv_file_path):
#         """
#         Load CSV data and reshape it for LSTM input
#         Expected format: predictor1_LT1, predictor1_LT2, ..., predictor2_LT1, ..., effort, year
#         """
#         # Load the data
#         df = pd.read_csv(csv_file_path)
#         df.loc[:, 'country'] = [field_id[:2] for field_id in df.field_id]
#
#         # Find all predictor columns that follow the pattern *_LT[1-8]
#         predictor_cols = [col for col in df.columns if
#                           col.endswith(('_LT1', '_LT2', '_LT3', '_LT4', '_LT5', '_LT6', '_LT7', '_LT8'))]
#
#         X = df.loc[:, predictor_cols].to_numpy()
#
#         # Check required columns
#         if 'yield_anom' not in df.columns:
#             raise ValueError("Target column 'yield_anom' not found in the dataset")
#         if 'c_year' not in df.columns:
#             raise ValueError("Year column 'c_year' not found in the dataset")
#
#         # Extract target and year
#         y = df['yield_anom'].values.reshape(-1, 1)
#         years = df['c_year'].values
#         countries = df['country'].values
#
#         print(f"Data shape - X: {X.shape}, y: {y.shape}")
#         print(f"Years range: {years.min()} to {years.max()}")
#         print(X.shape)
#         print(y.shape)
#
#         return X, y, years, countries, predictor_cols
#
#     def build_model(self, input_shape, hidden_layer_sizes=[100,50,50,1], dropout_rate=0.1):
#         """
#         Build the LSTM model
#         """
#         model = Sequential()
#         for i, neur in enumerate(hidden_layer_sizes):
#             if i == 0:
#                 model.add(Dense(neur, input_shape=input_shape, activation='relu'))
#                 if dropout_rate:
#                     model.add(Dropout(dropout_rate))
#             elif (i>0) and (i<len(hidden_layer_sizes)):
#                 model.add(Dense(neur, input_shape=input_shape, activation='relu'))
#                 if dropout_rate:
#                     model.add(Dropout(dropout_rate))
#             else:
#                 model.add(Dense(neur, activation='relu'))
#
#         model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#
#         return model
#
#     def train_model(self, X_train, y_train, X_val=None, y_val=None,
#                     epochs=100, batch_size=32, patience=5):
#         """
#         Train the LSTM model
#         """
#         # Set up early stopping
#         early_stopping = EarlyStopping(
#             monitor='val_loss' if X_val is not None else 'loss',
#             patience=patience,
#             restore_best_weights=True
#         )
#
#         # Prepare validation data
#         validation_data = (X_val, y_val) if X_val is not None else None
#
#         # Train the model
#         self.history = self.model.fit(
#             X_train, y_train,
#             validation_data=validation_data,
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=[early_stopping],
#             verbose=1
#         )
#
#         return self.history
#
#     def predict(self, X):
#         """
#         Make predictions and inverse transform
#         """
#         predictions_scaled = self.model.predict(X)
#         predictions = self.scaler_y.inverse_transform(predictions_scaled)
#         return predictions
#
#     def evaluate_model(self, X_test, y_test):
#         """
#         Evaluate the model performance
#         """
#         predictions = self.predict(X_test)
#         y_test_original = self.scaler_y.inverse_transform(y_test)
#
#         mse = mean_squared_error(y_test_original, predictions)
#         mae = mean_absolute_error(y_test_original, predictions)
#         r2 = r2_score(y_test_original, predictions)
#
#         print(f"Model Performance:")
#         print(f"MSE: {mse:.4f}")
#         print(f"MAE: {mae:.4f}")
#         print(f"R²: {r2:.4f}")
#
#         return {
#             'mse': mse,
#             'mae': mae,
#             'r2': r2,
#             'predictions': predictions,
#             'actual': y_test_original
#         }
#
#     def transfer_learning_cv(self, X, y, years, countries, predictors, crop, path_out, target_country='UA',
#                              hidden_layer_sizes=[100,50,50,1], dropout_rate=0.4, tl_train_layers=2,
#                              early_stopping_patience=10):
#         """
#         Perform transfer learning with leave-one-year-out cross-validation
#         1. Train base model on all countries except target_country
#         2. Fine-tune on target_country data (excluding test year)
#         3. Test on target_country data from test year only
#         """
#         if not os.path.exists(os.path.join(path_out, f'configs_{crop}.txt')):
#             with open(os.path.join(path_out, f'configs_{crop}.txt'), 'w') as file:
#                 file.write(f'used configs: hidden layer size={hidden_layer_sizes}\n dropout_rate={dropout_rate}\n tl_train_layers='
#                            f'{tl_train_layers}\n early_stopping_patience={early_stopping_patience}')
#
#         # Get unique years from target country
#         target_mask = countries == target_country
#         target_years = np.unique(years[target_mask])
#         target_years = np.sort(target_years)
#
#         eo_preds = [a for a in predictors if a.startswith(('evi', 'sm', 'VODCA'))]
#         met_preds = [a for a in predictors if not a.startswith(('evi', 'sm', 'VODCA'))]
#         i_eo = [predictors.index(a) for a in eo_preds]
#         i_met = [predictors.index(a) for a in met_preds]
#         print(i_met, i_eo)
#         X_eo, X_met = X[:, i_eo], X[:, i_met]
#
#         print(f"Transfer Learning Setup:")
#         print(f"Target country: {target_country}")
#         print(f"Available years for {target_country}: {target_years}")
#         print(f"Total samples in {target_country}: {np.sum(target_mask)}")
#
#         cv_results = []
#         fold_predictions = []
#         fold_actuals = []
#
#         ret_vals = pd.DataFrame(columns=['year', 'region', 'forecasted', 'forecasted_tl', 'forecasted_no_tl',
#                                          'forecasted_eo', 'forecasted_met', 'observed'])
#
#         for test_year in target_years:
#             print(f"\n{'=' * 60}")
#             print(f"FOLD: Testing on {target_country} data from year {test_year}")
#             print(f"{'=' * 60}")
#
#             # Prepare base training data (all countries except target)
#             train_country_mask = countries != target_country
#             train_year_mask = years != test_year
#             test_country_mask = countries == target_country
#             test_year_mask = years == test_year
#
#             train_mask = train_country_mask & train_year_mask
#             test_mask = test_country_mask & test_year_mask
#             finetune_mask = test_country_mask & train_year_mask
#             train_no_ft_mask = train_year_mask
#
#             X_base, y_base = X[train_mask], y[train_mask]
#             X_test, y_test = X[test_mask], y[test_mask]
#             X_finetune, y_finetune = X[finetune_mask], y[finetune_mask]
#             X_base_all, y_base_all = X[train_no_ft_mask], y[train_no_ft_mask]
#             X_base_met, X_base_eo = X_met[train_mask], X_eo[train_mask]
#             X_finetune_met, X_finetune_eo = X_met[finetune_mask], X_eo[finetune_mask]
#             X_test_met, X_test_eo = X_met[test_mask], X_eo[test_mask]
#
#             print(f"Base training samples: {X_base.shape[0]}")
#             print(f"Fine-tuning samples: {X_finetune.shape[0]}")
#             print(f"Test samples: {X_test.shape[0]}")
#
#             if X_test.shape[0] == 0:
#                 print(f"No test data for {target_country} in year {test_year}, skipping...")
#                 continue
#
#             # Step 1: Train base model on all countries except target
#             print("\nStep 1: Training base model...")
#             base_scaler_X = StandardScaler()
#             base_scaler_X_eo = StandardScaler()
#             base_scaler_X_met = StandardScaler()
#             base_scaler_X_all = StandardScaler()
#             base_scaler_y = StandardScaler()
#             base_scaler_y_all = StandardScaler()
#
#             # Scale base training data
#             y_base_scaled = base_scaler_y.fit_transform(y_base)
#             y_base_all_scaled = base_scaler_y_all.fit_transform(y_base_all)
#             base_scaler_X, X_base_scaled = scale_x(scaler=base_scaler_X, X=X_base, fit=True)
#             base_scaler_X_eo, X_base_eo_scaled = scale_x(scaler=base_scaler_X_eo, X=X_base_eo, fit=True)
#             base_scaler_X_met, X_base_met_scaled = scale_x(scaler=base_scaler_X_met, X=X_base_met, fit=True)
#             base_scaler_X_all, X_base_all_scaled = scale_x(scaler=base_scaler_X_all, X=X_base_all, fit=True)
#
#             X_base_train, y_base_train, X_base_val, y_base_val = train_val_split(X_base_scaled, y_base_scaled)
#             X_base_train_eo, _, X_base_val_eo, _ = train_val_split(X_base_eo_scaled, y_base_all_scaled)
#             X_base_train_met, _, X_base_val_met, _ = train_val_split(X_base_met_scaled, y_base_all_scaled)
#             X_base_train_all, y_base_train_all, X_base_val_all, y_base_val_all = train_val_split(X_base_all_scaled,
#                                                                                                  y_base_all_scaled)
#
#             # Build and train base model
#             base_model = self.build_model(input_shape=X_base.shape[1:], hidden_layer_sizes=hidden_layer_sizes,
#                                           dropout_rate=dropout_rate)
#             base_model_eo = self.build_model(input_shape=X_base_eo.shape[1:], hidden_layer_sizes=hidden_layer_sizes,
#                                           dropout_rate=dropout_rate)
#             base_model_met = self.build_model(input_shape=X_base_met.shape[1:], hidden_layer_sizes=hidden_layer_sizes,
#                                           dropout_rate=dropout_rate)
#             base_model_all = self.build_model(input_shape=X_base_all.shape[1:], hidden_layer_sizes=hidden_layer_sizes,
#                                           dropout_rate=dropout_rate)
#
#             early_stopping_base = EarlyStopping(
#                 monitor='val_loss',
#                 patience=early_stopping_patience,
#                 restore_best_weights=True,
#                 verbose=0
#             )
#
#             base_history = base_model.fit(X_base_train, y_base_train, validation_data=(X_base_val, y_base_val),
#                                           epochs=150, batch_size=32, callbacks=[early_stopping_base], verbose=0)
#             print("\nStep 1: Training eo model...")
#             base_history_eo = base_model_eo.fit(X_base_train_eo, y_base_train,
#                                                 validation_data=(X_base_val_eo, y_base_val),
#                                                 epochs=150, batch_size=32, callbacks=[early_stopping_base], verbose=0)
#             print("\nStep 1: Training met model...")
#             base_history_met = base_model_met.fit(X_base_train_met, y_base_train,
#                                                   validation_data=(X_base_val_met, y_base_val),
#                                                   epochs=150, batch_size=32, callbacks=[early_stopping_base], verbose=0)
#             print("\nStep 1: Training all model...")
#             base_history_all = base_model_all.fit(X_base_train_all, y_base_train_all,
#                                                   validation_data=(X_base_val_all, y_base_val_all),
#                                                   epochs=150, batch_size=32, callbacks=[early_stopping_base], verbose=0)
#
#             # Scale test data using base scalers
#             y_test_scaled = base_scaler_y.transform(y_test)
#             _, X_test_scaled = scale_x(scaler=base_scaler_X, X=X_test, fit=False)
#             _, X_test_scaled_eo = scale_x(scaler=base_scaler_X_eo, X=X_test_eo, fit=False)
#             _, X_test_scaled_met = scale_x(scaler=base_scaler_X_met, X=X_test_met, fit=False)
#
#             # Make predictions
#             predictions_scaled_nft = base_model.predict(X_test_scaled, verbose=0)
#             predictions_nft = base_scaler_y.inverse_transform(predictions_scaled_nft)
#             y_test_original_nft = base_scaler_y.inverse_transform(y_test_scaled)
#
#             # Make predictions train
#             predictions_scaled_nft_train = base_model.predict(X_base_train, verbose=0)
#             predictions_nft_train = base_scaler_y.inverse_transform(predictions_scaled_nft_train)
#             y_train_original_nft = base_scaler_y.inverse_transform(y_base_train)
#
#             # # Make predictions all
#             predictions_scaled_all = base_model_all.predict(X_test_scaled, verbose=0)
#             predictions_all = base_scaler_y.inverse_transform(predictions_scaled_all)
#
#             # Calculate metrics for this fold
#             mse_nft = mean_squared_error(y_test_original_nft, predictions_nft)
#             mae_nft = mean_absolute_error(y_test_original_nft, predictions_nft)
#             r2_nft = r2_score(y_test_original_nft, predictions_nft)
#             r_nft = pearsonr(y_test_original_nft.ravel(), predictions_nft.ravel())[0]
#             mbias_nft = np.median(y_test_original_nft - predictions_nft)
#
#             # Calculate metrics for this fold train
#             mse_nft_train = mean_squared_error(y_train_original_nft, predictions_nft_train)
#             mae_nft_train = mean_absolute_error(y_train_original_nft, predictions_nft_train)
#             r2_nft_train = r2_score(y_train_original_nft, predictions_nft_train)
#             r_nft_train = pearsonr(y_train_original_nft.ravel(), predictions_nft_train.ravel())[0]
#             mbias_nft_train = np.median(y_train_original_nft - predictions_nft_train)
#
#             print(f"Base model trained. Final val_loss: {min(base_history.history['val_loss']):.4f}")
#             print(
#                 f"  Results - MSE: {mse_nft:.4f}, MAE: {mae_nft:.4f}, R²: {r2_nft:.4f}, R: {r_nft:.3f}, med bias: {mbias_nft:.3f}")
#             print(f"  Results train - MSE: {mse_nft_train:.4f}, MAE: {mae_nft_train:.4f}, R²: {r2_nft_train:.4f}, "
#                   f"R: {r_nft_train:.3f}, med bias: {mbias_nft_train:.3f}")
#
#             # Step 2: Fine-tune on target country data (if available)
#             if X_finetune.shape[0] > 0:
#                 print(f"\nStep 2: Fine-tuning on {target_country} data...")
#
#                 # Scale fine-tuning data using base scalers (important for transfer learning)
#                 y_finetune_scaled = base_scaler_y.transform(y_finetune)
#                 _, X_finetune_scaled = scale_x(scaler=base_scaler_X, X=X_finetune, fit=False)
#                 _, X_finetune_scaled_met = scale_x(scaler=base_scaler_X_met, X=X_finetune_met, fit=False)
#                 _, X_finetune_scaled_eo = scale_x(scaler=base_scaler_X_eo, X=X_finetune_eo, fit=False)
#
#                 if not tl_train_layers == 'all':
#                     if tl_train_layers > len(base_model.layers):
#                         tl_train_layers = len(base_model.layers)
#                     for i in range(len(base_model.layers) - tl_train_layers):
#                         base_model.layers[i].trainable = False
#                         base_model_met.layers[i].trainable = False
#                         base_model_eo.layers[i].trainable = False
#
#                 # Split fine-tuning data for validation (if enough samples)
#                 if X_finetune_scaled.shape[0] > 5:
#                     X_ft_train, y_ft_train, X_ft_val, y_ft_val = train_val_split(X_finetune_scaled, y_finetune_scaled)
#                     validation_data_ft = (X_ft_val, y_ft_val)
#                     X_ft_train_eo, y_ft_train_eo, X_ft_val_eo, y_ft_val_eo = train_val_split(X_finetune_scaled_eo,
#                                                                                              y_finetune_scaled)
#                     validation_data_ft_eo = (X_ft_val_eo, y_ft_val_eo)
#                     X_ft_train_met, y_ft_train_met, X_ft_val_met, y_ft_val_met = train_val_split(X_finetune_scaled_met,
#                                                                                                  y_finetune_scaled)
#                     validation_data_ft_met = (X_ft_val_met, y_ft_val_met)
#                     monitor_ft = 'val_loss'
#                 else:
#                     # Use all data for training if too few samples
#                     X_ft_train, y_ft_train = X_finetune_scaled, y_finetune_scaled
#                     X_ft_train_eo, y_ft_train_eo = X_finetune_scaled_eo, y_finetune_scaled
#                     X_ft_train_met, y_ft_train_met = X_finetune_scaled_met, y_finetune_scaled
#                     validation_data_ft = None
#                     monitor_ft = 'loss'
#
#                 # Fine-tune with lower learning rate
#                 base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse',
#                                    metrics=['mae'])
#                 base_model_eo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse',
#                                       metrics=['mae'])
#                 base_model_met.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse',
#                                        metrics=['mae'])
#
#                 early_stopping_ft = EarlyStopping(monitor=monitor_ft, patience=early_stopping_patience - 5,
#                                                   restore_best_weights=True, verbose=0)
#
#                 ft_history = base_model.fit(
#                     X_ft_train, y_ft_train,
#                     validation_data=validation_data_ft,
#                     epochs=100,
#                     batch_size=16,  # Smaller batch size for fine-tuning
#                     callbacks=[early_stopping_ft],
#                     verbose=0
#                 )
#                 base_model_eo.fit(
#                     X_ft_train_eo, y_ft_train,
#                     validation_data=validation_data_ft_eo,
#                     epochs=100,
#                     batch_size=16,  # Smaller batch size for fine-tuning
#                     callbacks=[early_stopping_ft],
#                     verbose=0
#                 )
#                 base_model_met.fit(
#                     X_ft_train_met, y_ft_train,
#                     validation_data=validation_data_ft_met,
#                     epochs=100,
#                     batch_size=16,  # Smaller batch size for fine-tuning
#                     callbacks=[early_stopping_ft],
#                     verbose=0
#                 )
#
#                 print(f"Fine-tuning completed. Final loss: {ft_history.history['loss'][-1]:.4f}")
#             else:
#                 print(f"\nStep 2: No fine-tuning data available for {target_country}")
#
#             # Step 3: Test on target country test year
#             print(f"\nStep 3: Testing on {target_country} year {test_year}...")
#
#             # Make predictions
#             predictions_scaled = base_model.predict(X_test_scaled, verbose=0)
#             predictions = base_scaler_y.inverse_transform(predictions_scaled)
#             y_test_original = base_scaler_y.inverse_transform(y_test_scaled)
#
#             # Make predictions eo
#             predictions_scaled_eo = base_model_eo.predict(X_test_scaled_eo, verbose=0)
#             predictions_eo = base_scaler_y.inverse_transform(predictions_scaled_eo)
#
#             # Make predictions met
#             predictions_scaled_met = base_model_met.predict(X_test_scaled_met, verbose=0)
#             predictions_met = base_scaler_y.inverse_transform(predictions_scaled_met)
#
#             # Calculate metrics for this fold
#             mse = mean_squared_error(y_test_original, predictions)
#             mae = mean_absolute_error(y_test_original, predictions)
#             r2 = r2_score(y_test_original, predictions)
#             mbias = np.median(y_test_original - predictions)
#             r = pearsonr(y_test_original.ravel(), predictions.ravel())[0]
#
#             fold_result = {
#                 'test_year': test_year,
#                 'target_country': target_country,
#                 'mse': mse,
#                 'mae': mae,
#                 'r2': r2,
#                 'n_base_train': len(X_base),
#                 'n_finetune': len(X_finetune) if X_finetune.shape[0] > 0 else 0,
#                 'n_test': len(X_test)
#             }
#
#             cv_results.append(fold_result)
#             fold_predictions.extend(predictions.flatten())
#             fold_actuals.extend(y_test_original.flatten())
#
#             print(f"  Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, R: {r:.3f}, med bias: {mbias:.3f}")
#
#             dict = {'year': [test_year] * len(y_test_original),
#                     'region': [target_country] * len(y_test_original),
#                     'observed': y_test_original.ravel(),
#                     'forecasted': predictions_nft.ravel(),
#                     'forecasted_no_tl': predictions_all.ravel(),
#                     'forecasted_tl': predictions.ravel(),
#                     'forecasted_eo': predictions_eo.ravel(),
#                     'forecasted_met': predictions_met.ravel(),
#                     # 'forecasted_xgb': predictions,
#                     # 'forecasted_xgb_all': predictions
#                     }
#
#             df = pd.DataFrame(dict)
#             ret_vals = pd.concat([ret_vals, df])
#         ret_vals.to_csv(os.path.join(path_out, f'{crop}_1.csv'))
#         # Calculate overall CV metrics
#         if cv_results:
#             overall_mse = np.mean([r['mse'] for r in cv_results])
#             overall_mae = np.mean([r['mae'] for r in cv_results])
#             overall_r2 = np.mean([r['r2'] for r in cv_results])
#
#             std_mse = np.std([r['mse'] for r in cv_results])
#             std_mae = np.std([r['mae'] for r in cv_results])
#             std_r2 = np.std([r['r2'] for r in cv_results])
#
#             print(f"\n{'=' * 70}")
#             print(f"TRANSFER LEARNING CROSS-VALIDATION RESULTS ({target_country})")
#             print(f"{'=' * 70}")
#             print(f"Mean MSE: {overall_mse:.4f} ± {std_mse:.4f}")
#             print(f"Mean MAE: {overall_mae:.4f} ± {std_mae:.4f}")
#             print(f"Mean R²:  {overall_r2:.4f} ± {std_r2:.4f}")
#             print(f"Number of test years: {len(cv_results)}")
#         else:
#             print("No valid test results obtained.")
#             overall_mse = overall_mae = overall_r2 = std_mse = std_mae = std_r2 = 0
#
#         return {
#             'fold_results': cv_results,
#             'overall_metrics': {
#                 'mse_mean': overall_mse, 'mse_std': std_mse,
#                 'mae_mean': overall_mae, 'mae_std': std_mae,
#                 'r2_mean': overall_r2, 'r2_std': std_r2
#             },
#             'all_predictions': np.array(fold_predictions),
#             'all_actuals': np.array(fold_actuals),
#             'target_country': target_country
#         }
#
#     def hyper_tune(self, X, y, years, countries, crop, path_out, hp_settings):
#         """
#         Perform transfer learning with leave-one-year-out cross-validation
#         1. Train base model on all countries except target_country
#         2. Fine-tune on target_country data (excluding test year)
#         3. Test on target_country data from test year only
#         """
#         file_path = os.path.join(path_out, f'hp_tune_{crop}.csv')
#         row_head = ['year', 'layers', 'do_rate', 'esp', 'tl_tr_layer', 'train_r2', 'test_r2', 'train_mae', 'test_mae',
#                     'train_bias', 'test_bias', 'train_r', 'test_r', 'train_ft_r2', 'test_ft_r2', 'train_ft_mae',
#                     'test_ft_mae', 'train_ft_bias', 'test_ft_bias', 'train_ft_r', 'test_ft_r']
#         target_country = 'UA'
#
#         hps = [hp_settings[a] for a in hp_settings.keys()]
#         hp_combs = list(itertools.product(*hps))
#         print(hp_combs)
#
#         if not os.path.exists(path_out): os.makedirs(path_out)
#         if not os.path.exists(file_path):
#             with open(file_path, 'w') as f1:
#                 writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
#                 writer.writerow(row_head)
#
#         # Get unique years from target country
#         target_mask = countries == target_country
#         target_years = np.unique(years[target_mask])
#         target_years = np.sort(target_years)
#
#         print(f"Transfer Learning Setup:")
#         print(f"Target country: {target_country}")
#         print(f"Available years for {target_country}: {target_years}")
#         print(f"Total samples in {target_country}: {np.sum(target_mask)}")
#
#         for test_year in [2008, 2015, 2021]:
#             print(f"\n{'=' * 60}")
#             print(f"FOLD: Testing on {target_country} data from year {test_year}")
#             print(f"{'=' * 60}")
#
#             # Prepare base training data (all countries except target)
#             train_country_mask = countries != target_country
#             train_year_mask = years != test_year
#             test_country_mask = countries == target_country
#             test_year_mask = years == test_year
#
#             train_mask = train_country_mask & train_year_mask
#             test_mask = test_country_mask & test_year_mask
#             finetune_mask = test_country_mask & train_year_mask
#
#             X_base, y_base = X[train_mask], y[train_mask]
#             X_test, y_test = X[test_mask], y[test_mask]
#             X_finetune, y_finetune = X[finetune_mask], y[finetune_mask]
#
#             print(f"Base training samples: {X_base.shape[0]}")
#             print(f"Fine-tuning samples: {X_finetune.shape[0]}")
#             print(f"Test samples: {X_test.shape[0]}")
#
#             # Step 1: Train base model on all countries except target
#             print("\nStep 1: Training base model...")
#             base_scaler_X = StandardScaler()
#             base_scaler_y = StandardScaler()
#
#             # Scale base training data
#             y_base_scaled = base_scaler_y.fit_transform(y_base)
#             base_scaler_X, X_base_scaled = scale_x(scaler=base_scaler_X, X=X_base, fit=True)
#
#             X_base_train, y_base_train, X_base_val, y_base_val = train_val_split(X_base_scaled, y_base_scaled)
#
#             # Build and train base model
#             for hps in hp_combs:
#                 lstm_units, dropout_rate, tl_train_layers, early_stopping_patience = hps
#
#                 base_model = self.build_model(input_shape=X.shape[1:], hidden_layer_sizes=lstm_units,
#                                           dropout_rate=dropout_rate)
#
#                 early_stopping_base = EarlyStopping(
#                     monitor='val_loss',
#                     patience=early_stopping_patience,
#                     restore_best_weights=True,
#                     verbose=0
#                 )
#
#                 base_history = base_model.fit(X_base_train, y_base_train, validation_data=(X_base_val, y_base_val),
#                                               epochs=100, batch_size=32, callbacks=[early_stopping_base], verbose=0)
#
#                 # Scale test data using base scalers
#                 y_test_scaled = base_scaler_y.transform(y_test)
#                 _, X_test_scaled = scale_x(scaler=base_scaler_X, X=X_test, fit=False)
#
#                 # Make predictions
#                 predictions_scaled_nft = base_model.predict(X_test_scaled, verbose=0)
#                 predictions_nft = base_scaler_y.inverse_transform(predictions_scaled_nft)
#                 y_test_original_nft = base_scaler_y.inverse_transform(y_test_scaled)
#
#                 # Make predictions train
#                 predictions_scaled_nft_train = base_model.predict(X_base_train, verbose=0)
#                 predictions_nft_train = base_scaler_y.inverse_transform(predictions_scaled_nft_train)
#                 y_train_original_nft = base_scaler_y.inverse_transform(y_base_train)
#
#                 # Calculate metrics for this fold
#                 mae_nft = mean_absolute_error(y_test_original_nft, predictions_nft)
#                 r2_nft = r2_score(y_test_original_nft, predictions_nft)
#                 r_nft = pearsonr(y_test_original_nft.ravel(), predictions_nft.ravel())[0]
#                 mbias_nft = np.median(y_test_original_nft - predictions_nft)
#
#                 # Calculate metrics for this fold train
#                 mae_nft_train = mean_absolute_error(y_train_original_nft, predictions_nft_train)
#                 r2_nft_train = r2_score(y_train_original_nft, predictions_nft_train)
#                 r_nft_train = pearsonr(y_train_original_nft.ravel(), predictions_nft_train.ravel())[0]
#                 mbias_nft_train = np.median(y_train_original_nft - predictions_nft_train)
#
#                 # Step 2: Fine-tune on target country data (if available)
#                 if X_finetune.shape[0] > 0:
#                     print(f"\nStep 2: Fine-tuning on {target_country} data...")
#
#                     # Scale fine-tuning data using base scalers (important for transfer learning)
#                     y_finetune_scaled = base_scaler_y.transform(y_finetune)
#                     _, X_finetune_scaled = scale_x(scaler=base_scaler_X, X=X_finetune, fit=False)
#
#                     if not tl_train_layers == 'all':
#                         if tl_train_layers > len(base_model.layers):
#                             tl_train_layers = len(base_model.layers)
#                         for i in range(len(base_model.layers) - tl_train_layers):
#                             base_model.layers[i].trainable = False
#
#                     # Split fine-tuning data for validation (if enough samples)
#                     if X_finetune_scaled.shape[0] > 5:
#                         X_ft_train, y_ft_train, X_ft_val, y_ft_val = train_val_split(X_finetune_scaled,
#                                                                                      y_finetune_scaled)
#                         validation_data_ft = (X_ft_val, y_ft_val)
#                         monitor_ft = 'val_loss'
#                     else:
#                         # Use all data for training if too few samples
#                         X_ft_train, y_ft_train = X_finetune_scaled, y_finetune_scaled
#                         validation_data_ft = None
#                         monitor_ft = 'loss'
#
#                     # Fine-tune with lower learning rate
#                     base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse',
#                                        metrics=['mae'])
#
#                     early_stopping_ft = EarlyStopping(monitor=monitor_ft, patience=early_stopping_patience - 5,
#                                                       restore_best_weights=True, verbose=0)
#
#                     ft_history = base_model.fit(
#                         X_ft_train, y_ft_train,
#                         validation_data=validation_data_ft,
#                         epochs=100,
#                         batch_size=16,  # Smaller batch size for fine-tuning
#                         callbacks=[early_stopping_ft],
#                         verbose=0
#                     )
#                     print(f"Fine-tuning completed. Final loss: {ft_history.history['loss'][-1]:.4f}")
#                 else:
#                     print(f"\nStep 2: No fine-tuning data available for {target_country}")
#
#                 # Step 3: Test on target country test year
#                 print(f"\nStep 3: Testing on {target_country} year {test_year}...")
#
#                 # Make predictions
#                 predictions_scaled = base_model.predict(X_test_scaled, verbose=0)
#                 predictions = base_scaler_y.inverse_transform(predictions_scaled)
#                 y_test_original = base_scaler_y.inverse_transform(y_test_scaled)
#
#                 # Make predictions train_after fine_tune
#                 predictions_scaled_nft_train = base_model.predict(X_base_train, verbose=0)
#                 predictions_nft_train = base_scaler_y.inverse_transform(predictions_scaled_nft_train)
#                 y_train_original_nft = base_scaler_y.inverse_transform(y_base_train)
#
#                 # Calculate metrics for this fold
#                 mae_ft = mean_absolute_error(y_test_original, predictions)
#                 r2_ft = r2_score(y_test_original, predictions)
#                 mbias_ft = np.median(y_test_original - predictions)
#                 r_ft = pearsonr(y_test_original.ravel(), predictions.ravel())[0]
#
#                 # Calculate metrics for this fold train
#                 mae_ft_train = mean_absolute_error(y_train_original_nft, predictions_nft_train)
#                 r2_ft_train = r2_score(y_train_original_nft, predictions_nft_train)
#                 r_ft_train = pearsonr(y_train_original_nft.ravel(), predictions_nft_train.ravel())[0]
#                 mbias_ft_train = np.median(y_train_original_nft - predictions_nft_train)
#
#                 print(f"  Results - MAE: {mae_ft:.4f}, R²: {r2_ft:.4f}, R: {r_ft:.3f}, med bias: {mbias_ft:.3f}")
#
#                 resi = [test_year, lstm_units, dropout_rate, early_stopping_patience, tl_train_layers, r2_nft_train,
#                         r2_nft, mae_nft_train, mae_nft, mbias_nft_train, mbias_nft, r_nft_train, r_nft,
#                         r2_ft_train, r2_ft, mae_ft_train, mae_ft, mbias_ft_train, mbias_ft, r_ft_train, r_ft]
#
#                 with open(file_path, 'a') as f1:
#                     writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
#                     writer.writerow(resi)
#                 base_model = []


def scale_x(scaler, X, fit=False):
    X_reshaped = X.reshape(-1, X.shape[-1])
    if fit:
        X_scaled = scaler.fit_transform(X_reshaped)
    else:
        X_scaled = scaler.transform(X_reshaped)
    return scaler, X_scaled.reshape(X.shape)


def train_val_split(X_base, y_base):
    base_train_idx, base_val_idx = train_test_split(
        np.arange(len(X_base)), test_size=0.2, random_state=42
    )

    X_base_train = X_base[base_train_idx]
    y_base_train = y_base[base_train_idx]
    X_base_val = X_base[base_val_idx]
    y_base_val = y_base[base_val_idx]

    return X_base_train, y_base_train, X_base_val, y_base_val

# Usage example
def lstm_run(path_out, crop):
    # Initialize the LSTM predictor
    lstm_predictor = LSTMPredictor()
    if not os.path.exists(path_out): os.makedirs(path_out)

    # Load and prepare data
    try:
        # Replace 'your_data.csv' with your actual file path
        X, y, years, countries, predictors = lstm_predictor.load_and_prepare_data(f'Data/SC2/2W/final/{crop}_all_abs_fin_year_det.csv')

        print(f"Data loaded successfully!")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Years shape: {years.shape}")

        print("\nStarting HP-Tuning")
        hp_tuning_elements = {'lstm_units': [50, 70, 100],
                              'dropout_rate': [0.2, 0.3, 0.4],
                              'tl_train_layers': [4],
                              'early_stopping_patience': [20, 30, 40]}
        # hp_tuning_elements = {'lstm_units': [30, 50],
        #                       'dropout_rate': [0.2],
        #                       'tl_train_layers': [4],
        #                       'early_stopping_patience': [10, 20]}

        # lstm_predictor.hyper_tune(X=X, y=y, years=years, countries=countries, path_out=path_out,
        #                           crop=crop, hp_settings=hp_tuning_elements)
        # l_unit, do_rate, esp, tl_tr_layer = get_hps(crop, path=path_out)
        l_unit, do_rate, esp, tl_tr_layer = 70, 0.2, 30, 4
        # Perform leave-one-year-out cross-validation
        print("\nStarting Leave-One-Year-Out Cross-Validation...")

        cv_results = lstm_predictor.transfer_learning_cv(X=X, y=y, years=years, countries=countries,
                                                         path_out=path_out, predictors=predictors, crop=crop,
                                                         lstm_units=l_unit, dropout_rate=do_rate,
                                                         tl_train_layers=tl_tr_layer, early_stopping_patience=esp)

        print(cv_results)
        # Plot cross-validation results
        # lstm_predictor.plot_cv_results(cv_results)
        #
        # # Optional: Train final model on all data for future predictions
        # print("\nTraining final model on all data...")
        #
        # # Scale all data
        # scaler_X_final = StandardScaler()
        # scaler_y_final = StandardScaler()
        #
        # y_scaled = scaler_y_final.fit_transform(y)
        # X_reshaped = X.reshape(-1, X.shape[-1])
        # X_scaled = scaler_X_final.fit_transform(X_reshaped)
        # X_scaled = X_scaled.reshape(X.shape)
        #
        # # Split for validation
        # X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        #     X_scaled, y_scaled, test_size=0.2, random_state=42
        # )
        #
        # # Build and train final model
        # final_model = lstm_predictor.build_model(
        #     timesteps=X.shape[1],
        #     features=X.shape[2]
        # )
        #
        # early_stopping = EarlyStopping(
        #     monitor='val_loss',
        #     patience=15,
        #     restore_best_weights=True
        # )
        #
        # final_history = final_model.fit(
        #     X_train_final, y_train_final,
        #     validation_data=(X_val_final, y_val_final),
        #     epochs=100,
        #     batch_size=32,
        #     callbacks=[early_stopping],
        #     verbose=1
        # )
        #
        # print("Final model trained successfully!")

        # Save the final model and scalers if needed
        # final_model.save('lstm_model_final.h5')
        # joblib.dump(scaler_X_final, 'scaler_X_final.pkl')
        # joblib.dump(scaler_y_final, 'scaler_y_final.pkl')

    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def lstm_run_all_countries(path_out, crop):
    # Initialize the LSTM predictor
    lstm_predictor = LSTMPredictor()
    if not os.path.exists(path_out): os.makedirs(path_out)

    # Load and prepare data
    # Replace 'your_data.csv' with your actual file path
    X, y, years, countries, predictors = lstm_predictor.load_and_prepare_data(f'Data/SC2/2W/final/{crop}_all_abs_fin_year_det.csv')

    print(f"Data loaded successfully!")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Years shape: {years.shape}")

    print("\nStarting HP-Tuning")
    hp_tuning_elements = {'lstm_units': [50, 64, 80],
                          'dropout_rate': [0.2, 0.3, 0.4],
                          'tl_train_layers': [4],
                          'early_stopping_patience': [10, 25]}
    # hp_tuning_elements = {'lstm_units': [30, 50],
    #                       'dropout_rate': [0.2],
    #                       'tl_train_layers': [4],
    #                       'early_stopping_patience': [25]}

    ind_country = np.unique(countries)
    print(ind_country, len(countries))
    if crop=='maize': ind_country=ind_country[2:]

    for tc, target_country in enumerate(ind_country):
        print(f'start {target_country} number {tc+1}/{len(ind_country)}')
        lstm_predictor.hyper_tune(X=X, y=y, years=years, countries=countries, path_out=path_out,
                                  crop=crop, hp_settings=hp_tuning_elements, target_country=target_country)
        l_unit, do_rate, esp, tl_tr_layer = get_hps(crop, path=path_out, target_country=target_country)
        # l_unit, do_rate, esp, tl_tr_layer = 70, 0.2, 30, 4
        # Perform leave-one-year-out cross-validation
        print("\nStarting Leave-One-Year-Out Cross-Validation...")

        cv_results = lstm_predictor.transfer_learning_cv(X=X, y=y, years=years, countries=countries,
                                                         path_out=path_out, predictors=predictors, crop=crop,
                                                         lstm_units=l_unit, dropout_rate=do_rate,
                                                         tl_train_layers=tl_tr_layer, early_stopping_patience=esp,
                                                         target_country=target_country)
        print(target_country, cv_results)

#
# def ann_run(path_out, crop):
#     # Initialize the LSTM predictor
#     ann_predictor = ANNPredictor()
#
#     if not os.path.exists(path_out): os.makedirs(path_out)
#
#     # Load and prepare data
#     X, y, years, countries, predictors = ann_predictor.load_and_prepare_data(f'Data/SC2/2W/final/{crop}_all_abs_fin_year_det.csv')
#
#     print(f"Data loaded successfully!")
#     print(f"Input shape: {X.shape}")
#     print(f"Target shape: {y.shape}")
#     print(f"Years shape: {years.shape}")
#
#     print("\nStarting HP-Tuning")
#     hp_tuning_elements = {'hidden_layer_sizes': [[100,50,50,25,25,1], [100,50,50,50,50,50,50,1], [80,60,40,40,20,10,1]],
#                           'dropout_rate': [0.2, 0.3, 0.4],
#                           'tl_train_layers': [1, 2, 4],
#                           'early_stopping_patience': [10, 20, 30]}
#
#     # ann_predictor.hyper_tune(X=X, y=y, years=years, countries=countries, path_out=path_out,
#     #                           crop=crop, hp_settings=hp_tuning_elements)
#     layers, do_rate, esp, tl_tr_layer = get_hps(crop, path=path_out, ann=True)
#     # layers, do_rate, esp, tl_tr_layer = [100,50,50,50,50,1], 0.2, 20, 1
#     # Perform leave-one-year-out cross-validation
#     print("\nStarting Leave-One-Year-Out Cross-Validation...")
#
#     cv_results = ann_predictor.transfer_learning_cv(X=X, y=y, years=years, countries=countries,
#                                                      path_out=path_out, predictors=predictors, crop=crop,
#                                                      hidden_layer_sizes=layers, dropout_rate=do_rate,
#                                                      tl_train_layers=tl_tr_layer, early_stopping_patience=esp)
#     print(cv_results)

def get_hps(crop, path, target_country='UA', ann=False):
    val = 'test_ft_mae'

    df = pd.read_csv(os.path.join(path, f'hp_tune_{crop}_{target_country}.csv'))
    if df.shape[0]==0:
        l_unit, do_rate, esp, tl_tr_layer = 64, 0.3, 25, 4
    else:
        years = np.unique(df.year)
        perf_dict = {a: [] for a in range(int(df.shape[0]/len(years)))}

        for year in years:
            df_sub = df.iloc[np.where(df.year==year)[0], :]
            for i in range(df_sub.shape[0]):
                perf_dict[i].append(df_sub.loc[:, val].iloc[i])

        perfs = pd.DataFrame(index=perf_dict.keys(), columns=['mean'])
        for i in perfs.index:
            perfs.iloc[i,0] = np.mean(perf_dict[i])

        asc = False
        if val.endswith('mae'): asc=True
        top_perf = perfs.sort_values('mean', ascending=asc).head(3)
        [print(perf_dict[i], np.mean(perf_dict[i])) for i in top_perf.index]

        if ann:
            hps = df.iloc[top_perf.index[0], :][['layers', 'do_rate', 'esp', 'tl_tr_layer']]
            l_unit, do_rate, esp, tl_tr_layer = hps
            l_unit, esp, tl_tr_layer = json.loads(l_unit), int(esp), int(tl_tr_layer)
        else:
            hps = df.iloc[top_perf.index[0],:][['l_unit', 'do_rate', 'esp', 'tl_tr_layer']]
            l_unit, do_rate, esp, tl_tr_layer = hps
            l_unit, esp, tl_tr_layer = int(l_unit), int(esp), int(tl_tr_layer)
        print(l_unit)
    return l_unit, do_rate, esp, tl_tr_layer

if __name__ == "__main__":
    start_pro = datetime.datetime.now()
    # for crop in ['spring_barley', 'winter_wheat', 'maize']: lstm_run(crop)
    # for crop in ['maize']:
    #     ann_run()
        # l_unit, do_rate, esp, tl_tr_layer = get_hps(crop, path='Results/SC2/202506/loocv/country/lstm_test3')
        # print(l_unit, do_rate, esp, tl_tr_layer)
    end_pro = datetime.datetime.now()
    print(f'calculation stopped at {end_pro} and took {end_pro - start_pro}')
