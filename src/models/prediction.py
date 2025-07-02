import os
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from prophet import Prophet

class ModelPredictor:
    """Handles model prediction operations"""
    def __init__(self, data_extractor = None, data_processor = None, model_manager = None, 
                 list_date = None, metric_Id_list = None, metric_Id_list_tar = None,
                 entity_cov = None, entity_tar = None):
        self.data_extractor = data_extractor
        self.data_processor = data_processor
        self.model_manager = model_manager
        self.list_date = list_date
        self.metric_Id_list = metric_Id_list
        self.metric_Id_list_tar = metric_Id_list_tar
        self.entity_cov = entity_cov
        self.entity_tar = entity_tar
        #self.model = self.model_manager.load_serialized_model(serialized_model)
    
    def rolling_forecast(self, df, horizon=3, train_size=0.7):
        """Perform rolling window forecasting."""
        # Split data into train and test sets
        train_size = int(train_size * len(df))
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

        # Initialize storage for results
        all_predictions = []
        all_true_values = []
        forecast_dates = []
        horizon_labels = []

        # Iterate through the test set in steps of 'horizon'
        for i in range(0, len(test), horizon):
            # Current training data (all past + up to current test point)
            current_train = pd.concat([train, test.iloc[:i]], axis=0)

            # Test data for the next 'horizon' steps
            current_test = test.iloc[i:i + horizon]
            if current_test.empty:
                break

            # Create Prophet Model
            model = Prophet(
                  yearly_seasonality=True,
                  weekly_seasonality=False,
                  daily_seasonality=False,
                  )
            
            # Add all exogenous regressors
            regressor_cols = [col for col in current_train.columns if col not in ["ds", "y"]]
            for col in regressor_cols:
              model.add_regressor(col)
            # Fit model
            model.fit(current_train)

            future = model.make_future_dataframe(
                periods=len(current_test),
                freq='M',
                include_history=False
            )

            # Add exogenous variables
            future_regressors = current_test[regressor_cols + ['ds']].copy()
            future = future.merge(future_regressors, on="ds", how="left")

            # Predict
            forecast = model.predict(future)

            # Store results
            all_predictions.extend(forecast["yhat"].values)
            all_true_values.extend(current_test["y"].values)
            forecast_dates.extend(current_test["ds"].values)
            horizon_labels.extend([f"Month {h + 1}" for h in range(len(current_test))])

        results = {
            "predictions": all_predictions,
            "true_values": all_true_values,
            "forecast_dates": forecast_dates,
            "horizon_labels": horizon_labels,
            "model": model,  # Returns the last trained model
            'test_data': test
        }

        return results

    def generate_out_of_sample_forecast(self, model, test_df, date_backtesting, future_periods=3):
        """Generate out-of-sample forecast with exogenous variables."""
        date_backtesting = pd.to_datetime(date_backtesting)
        #date_backtesting = date_backtesting.strftime("%Y-%m-%d")
        # Get last date from historical data
        if test_df['ds'].iloc[-1] < date_backtesting:
            last_date = test_df['ds'].iloc[-1]

            # Create block matrix for exogenous variables
            row_input_list = np.array(test_df.iloc[-1][2:]).reshape(5,3)
            result_matrix = self._create_block_matrix(row_input_list, num_blocks=5)

            # Create future exogenous DataFrame
            dates = pd.date_range(start=last_date, periods=3, freq='M')
            df_future_exog = pd.DataFrame(
                result_matrix,
                columns=test_df.columns[2:],  # Assuming exog columns start from index 2
                index=dates
            )

            # Interpolate missing values
            df_future_exog_interpolate = df_future_exog.interpolate(method='linear', limit_direction='forward')
            df_future_exog_interpolate.reset_index(inplace=True)
            df_future_exog_interpolate.rename(columns={'index': 'ds'}, inplace=True)
            forecast = model.predict(df_future_exog_interpolate)
        else:
            print(test_df.head())
            #date_backtesting = date_backtesting.strftime("%Y-%m-%d")
            target_date = date_backtesting
            print(f"Target date for prediction: {target_date}")
            mask = test_df['ds'] == target_date
            if not mask.any():
                print(f"Warning: target_date {target_date} not found in test_df['ds'].")
                print("Available dates:", test_df['ds'].unique())
                return None, None
            idx_pos = test_df[mask].index[0]
            #idx_pos = test_df[test_df['ds'] == target_date].index[0]
            test_df = test_df.drop(columns='y')
            df_future_exog_interpolate = test_df.iloc[idx_pos-3:idx_pos]
            forecast = model.predict(df_future_exog_interpolate)

        return forecast, df_future_exog_interpolate

    def _create_block_matrix(self, row_list, num_blocks=5):
        """Creates a 3x(3*num_blocks) matrix by concatenating Toeplitz blocks."""
        if num_blocks > len(row_list):
            print(f"Warning: num_blocks ({num_blocks}) exceeds the number of provided row lists ({len(row_list)}). Using {len(row_list)} blocks.")
            num_blocks = len(row_list)

        # Create first block
        first_row = row_list[0]
        toeplitz_block = self._create_toeplitz(first_row)
        block_size = len(first_row)
        mask = np.tri(block_size, k=-1, dtype=bool)  # Lower triangular (excluding diagonal)
        upper_block = np.where(~mask, toeplitz_block, np.nan)

        # Concatenate blocks horizontally
        full_matrix = upper_block.copy()
        for i in range(1,num_blocks):
            current_row = row_list[i]
            if len(current_row) != block_size:
                raise ValueError(f"Row list at index {i} has inconsistent length. Expected {block_size}, got {len(current_row)}.")
            new_block = self._create_toeplitz(current_row)
            new_block = np.where(~mask, new_block, np.nan)
            full_matrix = np.hstack((full_matrix, new_block))

        return full_matrix

    def _create_toeplitz(self, row_list):
        """Constructs a Toeplitz matrix from the given list representing the first row."""
        num_cols = len(row_list)
        num_rows = num_cols  # For a square Toeplitz matrix
        toeplitz_matrix = np.zeros((num_rows, num_cols), dtype=row_list[0].__class__)

        for i in range(num_rows):
            for j in range(num_cols):
                if j-i >= 0:
                    toeplitz_matrix[i, j] = row_list[j-i]
                elif i-j < num_cols:
                    toeplitz_matrix[i, j] = row_list[i-j]
                else:
                    toeplitz_matrix[i, j] = np.nan
        return toeplitz_matrix
    
    def make_prediction(self, start_date = None):
        if start_date is None:
          prediction_date  = dt.datetime.now()
          prediction_date = prediction_date.strftime("%Y-%m-%d")
        else:
          prediction_date = start_date
          prediction_date = pd.to_datetime(prediction_date)
          prediction_date = prediction_date.strftime("%Y-%m-%d")

         # To avoid connection to API and fetching data, we will use a pre-saved monthly data file.
        '''
        raw_data = self.data_extractor.fetch_data(
            self.list_date, 
            self.metric_Id_list, 
            self.metric_Id_list_tar, 
            self.entity_cov, 
            self.entity_tar
        )
        _, monthly_data = self.data_processor.process_data(raw_data)
        '''
        PROJECT_ROOT = Path(__file__).parent.parent
        
        monthly_data_path = PROJECT_ROOT / 'data' / 'monthly_data.parquet'
        
        #monthly = self.model_manager.load_and_concatenate(
        #str(monthly_data_path), monthly_data, 'Fecha', True
        #)
        monthly = self.model_manager.load_and_concatenate(
        str(monthly_data_path), None, 'Fecha', True
        )
        modeling_data, _, y_mean, y_std = self.data_processor.preprocess_for_modeling(monthly)
        #raw_data = self.fetch_data(list_date, metric_Id_list, metric_Id_list_tar, entity_cov, entity_tar)
        #_, monthly_data = self.process_data()
        #monthly = self.load_and_concatenate('monthly_data.parquet', monthly_data, 'Fecha', True)
        #modeling_data, _ , _ , _ = self.preprocess_for_modeling(monthly)
        delay = 1
        number_of_lags = 3
        final_lag = delay + number_of_lags
        mod_modeling_data = self.data_processor.mod_df(
            modeling_data, delay, number_of_lags, final_lag
        )
        print(mod_modeling_data.head())
        print(f'prediction_date: {prediction_date}')
        serialized_model_path = PROJECT_ROOT / 'data' / 'serialized_prophet'
        model = self.model_manager.load_serialized_model(str(serialized_model_path))
        forecast_data, _ = self.generate_out_of_sample_forecast(
            model, mod_modeling_data, prediction_date
        )
        # Generate out-of-sample forecast
        #forecast_data, _ = self.generate_out_of_sample_forecast(self.model, mod_modeling_data, prediction_date)
        return forecast_data, monthly, modeling_data, y_mean, y_std

    def forecast_data_rescaled(self,forecast_data, columns, y_mean, y_std):
        forecast_data_rescale = pd.DataFrame(index=forecast_data.index)
        for col in columns:
          forecast_data_rescale[col] = forecast_data[col] * y_std + y_mean
        forecast_data_rescale['ds'] = forecast_data['ds']
        return forecast_data_rescale