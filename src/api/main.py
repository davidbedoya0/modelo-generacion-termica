from src.data.extraction import DataExtractor
from src.data.processing import DataProcessor
from src.models.management import ModelManager
from src.models.training import ModelTrainer
from src.models.prediction import ModelPredictor




# Initialize components
data_extractor = DataExtractor()
data_processor = DataProcessor()
model_manager = ModelManager()
model_trainer = ModelTrainer()

# Modified code with proper path handling

model_predictor = ModelPredictor(
    data_extractor=data_extractor,
    data_processor=data_processor,
    model_manager=model_manager,
    list_date=['2025-02-01', '2025-05-31'],
    metric_Id_list=['VoluUtilDiarEner','AporEner','PorcVoluUtilDiar'],
    metric_Id_list_tar=['Gene'],
    entity_cov='Sistema',
    entity_tar='Recurso'
)

# Make prediction
# date for prediction can be set here must be in the format 'YYYY-MM-DD' with the last day of the month
# example, datebacktesting = '2024-01-31'
#forecast_data, monthly_data, modeling_data, y_mean, y_std = model_predictor.make_prediction(start_date='2024-01-31')
forecast_data, monthly_data, modeling_data, y_mean, y_std = model_predictor.make_prediction()

# Rescale the forecast
rescaled_forecast = model_predictor.forecast_data_rescaled(
    forecast_data,
    columns=['yhat', 'yhat_lower', 'yhat_upper'],
    y_mean=y_mean,
    y_std=y_std
)
periods_to_analyze = [
    ('1 año', 1),
    ('2 años', 2),
    ('5 años', 5)
]
results, df_results = model_manager.generate_descriptive_stats(monthly_data, periods= periods_to_analyze)
model_manager.save_forecast_data(
    forecast_data=rescaled_forecast,
    columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'],
    output_file='forecast_data.csv')
model_manager.save_forecast_data(
    forecast_data=monthly_data,
    columns=monthly_data.columns.tolist(),
    output_file='exogenous_target.csv')
model_manager.save_forecast_data(
    forecast_data=df_results,
    columns=df_results.columns.tolist(),
    output_file='descriptive_statistics_5_years.csv')

#df_metrics = model_manager.load_json_metrics('data/metrics.json')

print("Forecast Data:")
print(forecast_data.head())
print("\nMonthly Data:")
print(monthly_data.head())
print("\nModeling Data:")
print(modeling_data.head())
print("\nScaling Parameters:")
print(y_mean, y_std)
print("\nRescaled Forecast:")
print(rescaled_forecast.head())
print(len(rescaled_forecast))
print("\nDescriptive Statistics Results:")
print(df_results)



#from fastapi import FastAPI

#app = FastAPI()

#@app.get("/")
#async def read_root():
#    return {"message": "Hello, World!"}
