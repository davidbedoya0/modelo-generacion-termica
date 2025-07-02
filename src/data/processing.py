import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Handles all data processing and transformation operations"""
    
    def process_data(self, raw_data):
        """Process and transform the raw data into a format suitable for modeling."""
        if raw_data is None:
            raise ValueError("No data available. Run fetch_data() first.")

        # Unpack the raw data
        df_volumen_ener = raw_data[0]
        df_apor_ener = raw_data[1]
        df_porc_vol_util_ = raw_data[2]
        df_genreal = raw_data[3]
        df_ONI = raw_data[4]

        # Process generation data
        df_genreal_tipo = df_genreal.groupby(['Fecha','TipoGeneracion'])['GeneracionRealEstimada'].sum().reset_index()
        Gigafactor_gene = 1e6  # Conversion factor from kWh to Gwh
        df_genreal_tipo['GeneracionRealEstimada'] = df_genreal_tipo['GeneracionRealEstimada'].div(Gigafactor_gene)

        # Pivot and rename columns
        df_genreal_tipo = df_genreal_tipo.pivot_table(
            columns='TipoGeneracion',
            index='Fecha',
            values='GeneracionRealEstimada',
            fill_value=0
        ).reset_index()
        df_genreal_tipo['Fecha'] = pd.to_datetime(df_genreal_tipo['Fecha'], format='%Y-%m-%d')
        df_genreal_tipo.set_index('Fecha', inplace=True)
        df_genreal_tipo = df_genreal_tipo[['Termica']]
        df_genreal_tipo.rename(columns={'Termica':'TERMICA'}, inplace=True)

        # Process volume data
        df_volumen_ener.rename(columns={'Value':'Volume in Energy'}, inplace=True)
        Gigafactor_volu = 1e6
        df_volumen_ener['Volume in Energy'] = df_volumen_ener['Volume in Energy'].div(Gigafactor_volu)
        df_volumen_ener.drop(columns=['Id'], inplace=True)
        df_volumen_ener.set_index('Date', inplace=True)

        # Process apor energy data
        df_apor_ener.rename(columns={'Value':'AporEnergia'}, inplace=True)
        Gigafactor_apor = 1e6
        df_apor_ener['AporEnergia'] = df_apor_ener['AporEnergia'].div(Gigafactor_apor)
        df_apor_ener.drop(columns=['Id'], inplace=True)
        df_apor_ener.set_index('Date', inplace=True)

        # Process volume utilization data
        df_porc_vol_util_.rename(columns={'Value':'PorcVoluUtilDiario'}, inplace=True)
        df_porc_vol_util_.drop(columns=['Id'], inplace=True)
        df_porc_vol_util_.set_index('Date', inplace=True)

        # Combine all data
        df_thermalgen_cov = pd.concat([
            df_genreal_tipo,
            df_volumen_ener,
            df_ONI,
            df_apor_ener,
            df_porc_vol_util_
        ], axis=1)

        # Create monthly aggregated data
        df_thermalcovariates_monthly = df_thermalgen_cov.resample('M').agg({
            'TERMICA': 'sum',
            'Volume in Energy': 'sum',
            'ONI Anomaly': 'max',
            'AporEnergia': 'sum',
            'PorcVoluUtilDiario': 'mean'
        })

        # Reset indices and rename columns
        df_thermalgen_cov.reset_index(inplace=True)
        df_thermalgen_cov.rename(columns={'index':'Fecha'}, inplace=True)
        df_thermalcovariates_monthly.reset_index(inplace=True)
        df_thermalcovariates_monthly.rename(columns={'index':'Fecha'}, inplace=True)

        return (df_thermalgen_cov, df_thermalcovariates_monthly)

    def preprocess_for_modeling(self, df, target_col='TERMICA'):
        """Prepare data for modeling by scaling and renaming columns."""
        # Store original scale info for inverse transform later
        y_mean = df[target_col].mean()
        y_std = df[target_col].std()

        # Rename columns for Prophet
        df = df.rename(columns={
            'Fecha': 'ds',
            target_col: 'y',
            'Volume in Energy': 'exog2',
            'ONI Anomaly': 'exog3',
            'AporEnergia': 'exog4',
            'PorcVoluUtilDiario': 'exog6',
        })

        # Scale features
        scaler = StandardScaler()
        numeric_cols = ['y'] + [f'exog{i}' for i in range(1, 8) if f'exog{i}' in df.columns]
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df, scaler, y_mean, y_std

    def add_lagged_regressors(self, df, regressor_col, lags):
        """Add lagged regressors to the DataFrame."""
        for exog in regressor_col:
          for lag in lags:
            df[f'{exog}_lag_t-{lag}'] = df[exog].shift(lag)
        return df

    def mod_df(self, df, delay, number_of_lags, final_lag):
        delay = 1
        number_of_lags = 3
        final_lag = delay + number_of_lags
        lags = np.arange(delay,final_lag)
        new_column = 'y_exog'
        df[new_column] = df['y']
        exog_lagged = ['exog2', 'exog3', 'exog4', 'exog6', 'y_exog']
        df = self.add_lagged_regressors(df, exog_lagged, lags)
        df.dropna(inplace=True)
        df_noexog = df.drop(columns=exog_lagged)
        return df_noexog