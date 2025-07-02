import pandas as pd
import datetime as dt
from pydataxm import *
from pydataxm.pydatasimem import ReadSIMEM

class DataExtractor:
    """Handles all data fetching operations"""
    
    def fetch_data(self, list_date, metric_Id_list, metric_Id_list_tar, entity_cov, entity_tar):
        """Fetch data from various sources including APIs and external datasets."""
        df_thermalgen = []

        # Fetch covariates data from API
        objetoAPI = pydataxm.ReadDB()
        for metric_id in metric_Id_list:
            df = objetoAPI.request_data(
                metric_id,
                entity_cov,
                dt.datetime.fromisoformat(list_date[0]),
                dt.datetime.fromisoformat(list_date[1]))
            df_thermalgen.append(df)

        # Fetch thermal generation data
        dataset_gen = 'E17D25'  # Real Generation from API
        simem_gen = ReadSIMEM(dataset_gen, list_date[0], list_date[1])
        df_tar = simem_gen.main()
        df_thermalgen.append(df_tar)

        # Fetch SST data for ONI index
        url = "https://psl.noaa.gov/data/correlation/nina3.anom.data"
        df_sst = self._process_sst_data(url, list_date)
        df_thermalgen.append(df_sst)

        return df_thermalgen

    def _process_sst_data(self, url, date_range):
        """Process SST data from NOAA"""
        df_sst = pd.read_csv(url, delim_whitespace=True, skiprows=1, header=None)
        df_sst_clean = df_sst.iloc[2:-3,:]
        df_sst_clean.columns = ['Year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

        # Reshape and clean data
        df_sst_clean_stack = df_sst_clean.set_index(['Year'])[df_sst_clean.columns[1:]].stack().reset_index()
        df_sst_clean_stack.columns = ['Year', 'Month', 'ONI Anomaly']
        df_sst_clean_stack['Date'] = pd.to_datetime(df_sst_clean_stack[['Year', 'Month']].assign(DAY=1))
        df_sst_clean_stack.drop(columns=['Year', 'Month'], inplace=True)
        df_sst_clean_stack.set_index('Date', inplace=True)
        df_sst_clean_stack['ONI Anomaly'] = pd.to_numeric(df_sst_clean_stack['ONI Anomaly'], errors='coerce')

        # Filter by date range and resample
        df_sst_clean_stack = df_sst_clean_stack[
            (df_sst_clean_stack.index >= date_range[0]) &
            (df_sst_clean_stack.index <= date_range[1])
        ]
        df_sst_clean_stack_daily = df_sst_clean_stack.resample('D').ffill()

        # Extend with 30 days of forward fill
        new_dates = pd.date_range(
            start=df_sst_clean_stack_daily.index[-1] + pd.Timedelta(days=1),
            periods=30,
            freq='D'
        )
        df_sst_clean_stack_daily = df_sst_clean_stack_daily.reindex(
            df_sst_clean_stack_daily.index.union(new_dates)
        ).ffill()

        return df_sst_clean_stack_daily
