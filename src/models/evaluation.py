import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""
    
    def evaluate_model(self, model, test_df):
        """Evaluate the model on test data."""
        regressors = [f'exog{i}' for i in range(1, 8) if f'exog{i}' in test_df.columns]

        # Prepare future dataframe with regressors
        future = test_df[['ds'] + regressors].copy()

        # Make predictions
        forecast = model.predict(future)

        # Get actual and predicted values
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values

        # Calculate metrics
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }

        return forecast, metrics

    def calculate_metrics(self, model, results, y_mean, y_std):
        """Calculate evaluation metrics for forecast results."""
        if results.empty:
            print("No predictions were made.")
            return None

        # Rescale values to original scale
        true_rescaled = results["true"] * y_std + y_mean
        pred_rescaled = results["predicted"] * y_std + y_mean

        # Calculate basic metrics
        metrics = {
            "rmse": np.sqrt(mean_squared_error(true_rescaled, pred_rescaled)),
            "mae": mean_absolute_error(true_rescaled, pred_rescaled),
            "mape": np.mean(np.abs((true_rescaled - pred_rescaled) / true_rescaled)) * 100
        }

        # Calculate information criteria if model is provided
        if model:
            residuals = results["true"] - results["predicted"]
            n = len(residuals)
            ssr = np.sum(residuals**2)
            sigma2 = ssr / n  # MLE Estimator Variance

            # Log-likelihood (normal distribution)
            llf = -n/2 * np.log(2*np.pi) - n/2 * np.log(sigma2) - ssr/(2*sigma2)

            # Count parameters
            k = (len(model.params['k']) +        # growth parameters
                 len(model.params['m']) +        # offset parameters
                 len(model.params['sigma_obs']) + # observation noise
                 (len(model.seasonalities) * 2) + # fourier terms for each seasonality
                 len(model.extra_regressors))

            metrics["aic"] = -2 * llf + 2 * k
            metrics["bic"] = -2 * llf + k * np.log(n)

        return metrics