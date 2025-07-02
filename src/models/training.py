from prophet import Prophet
from prophet.utilities import regressor_coefficients

class ModelTrainer:
    """Handles model training operations"""
    
    def train_model(self, train_df, seasonality_mode='additive',
                   weekly_seasonality=False, yearly_seasonality=True):
        """Train a Prophet model with optional regressors."""
        regressors = [f'exog{i}' for i in range(1, 8) if f'exog{i}' in train_df.columns]

        model = Prophet(
            seasonality_mode=seasonality_mode,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality
        )

        # Add regressors
        for reg in regressors:
            model.add_regressor(reg)

        model.fit(train_df)
        return model