import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

class ModelManager:
    """Handles model serialization and data management"""
    
    @staticmethod
    def load_serialized_model(filepath):
        """Load a pickled Prophet model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def load_serialized_figure(filepath):
        """Load a pickled Plotly figure"""
        with open(filepath, 'rb') as f:
            fig = pickle.load(f)
        return fig

    @staticmethod
    def load_json_metrics(filepath):
        df = pd.read_json(filepath, orient='columns')
        return df

    def save_forecast_data(self, forecast_data, columns, output_file):
        """Save forecast data to a file."""
        columns_to_save = columns
        selected_data = forecast_data[columns_to_save]
        PROJECT_ROOT= Path(__file__).parent.parent
        output_path_file =  PROJECT_ROOT / 'data' / output_file
        #output_path = Path(output_path_file)
        output_path_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            selected_data.to_csv(
                output_path_file,
                index=True,
                date_format="%Y-%m-%d"
            )
        except IOError as e:
            raise IOError(f"Failed to write CSV file: {e}")

    def save_data(self, df, file_path):
        """Save a DataFrame to NPZ file."""
        np.savez(
            file_path,
            index=df.index.values,
            **{col: df[col].values for col in df.columns}
        )

    def load_data(self, file_path):
        """Load a DataFrame from NPZ file."""
        with np.load(file_path, allow_pickle=True) as data:
            df = pd.DataFrame(
                {col: data[col] for col in data.files if col != 'index'},
                index=data['index'] if 'index' in data.files else None
            )
        return df

    def save_to_parquet_by_date_range(self, df, output_path, start_date, end_date):
        df_filtered = df.copy()
        if start_date:
            start_dt = start_date
            df_filtered = df_filtered[df_filtered['Fecha'] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df_filtered = df_filtered[df_filtered['Fecha'] <= end_dt]

        # Save to Parquet
        df_filtered.to_parquet(output_path)
        print(f"Successfully saved {len(df_filtered)} records to {output_path}")

    def load_and_concatenate(self, full_path, next_df, date_column, ensure_continuity):
        # Load the Parquet file
        current_df = pd.read_parquet(full_path)

        # Ensure date column exists and is datetime
        if date_column not in current_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in Parquet file")

        current_df[date_column] = pd.to_datetime(current_df[date_column])

        # If no next_df provided, just return the loaded DataFrame
        if next_df is None:
            return current_df

        # Validate next_df structure matches current_df
        if set(current_df.columns) != set(next_df.columns):
            raise ValueError("DataFrames must have identical column structures")

        # Check for datetime continuity if required
        if ensure_continuity:
            current_max_date = current_df[date_column].max()
            next_min_date = next_df[date_column].min()

            if next_min_date <= current_max_date:
                raise ValueError(
                    f"Date continuity violation: Next DataFrame starts at {next_min_date} "
                    f"which is before/equal to current DataFrame's max date {current_max_date}"
                )

        # Concatenate while preserving original column order
        concatenated = pd.concat(
            [current_df, next_df],
            axis=0,
            ignore_index=True
        )

        # Restore original column order
        concatenated = concatenated[current_df.columns]

        return concatenated
    
    def generate_descriptive_stats(self, data, periods):
        """
        Generate descriptive statistics for specified time periods

        Args:
            data: pandas DataFrame with datetime index
            periods: list of tuples with (period_name, years)

        Returns:
            Dictionary of DataFrames with descriptive statistics
        """
        #data = data.set_index('Fecha',inplace=True)
        data.set_index('Fecha',inplace=True)
        results = {}
        #current_date = data.index.max()  # Most recent date in data
        current_date = data.index.max()

        for period_name, years in periods:
            # Calculate cutoff date
            cutoff = current_date - pd.DateOffset(years=years)

            # Filter data for the period
            period_data = data.loc[cutoff:current_date]

            # Generate descriptive stats
            stats = period_data.describe(percentiles=[.25, .5, .75])

            # Add additional statistics if needed
            stats.loc['skew'] = period_data.skew()
            stats.loc['kurtosis'] = period_data.kurtosis()
            #stats.loc['count_null'] = period_data.isnull().sum()

            results[f"{period_name} ({years} year)"] = stats
            
            data_statistics = []

            for period, stats in results.items():
                # Add period to the stats dictionary
                stats_with_period = {'period': period}
                stats_with_period.update(stats)
                data_statistics.append(stats_with_period)

            # Convert to DataFrame
            df_statistics = pd.DataFrame(data_statistics)

            # Set period as the index if desired
            df_statistics.set_index('period', inplace=True)

            for period, stats in results.items():
                print(f"\n{period} Statistics:")
                print(stats)
                df_stats = pd.DataFrame(stats)

        return results, df_stats