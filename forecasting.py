"""
Forecasting Module
Forecasts future expenses using Prophet time series model.
"""

import pandas as pd
from prophet import Prophet


def prepare_forecast_data(df):
    """
    Prepare expense data for Prophet forecasting.
    
    Args:
        df: Expense DataFrame
        
    Returns:
        pd.DataFrame: Data formatted for Prophet (ds, y columns)
    """
    # Group by date and sum amounts
    daily_expenses = df.groupby('date')['amount'].sum().reset_index()
    daily_expenses.columns = ['ds', 'y']
    
    return daily_expenses


def train_forecast(df):
    """
    Train Prophet model on expense data.
    
    Args:
        df: Expense DataFrame with date and amount columns
        
    Returns:
        Prophet: Trained Prophet model
    """
    forecast_data = prepare_forecast_data(df)
    
    if len(forecast_data) < 2:
        return None
    
    model = Prophet()
    model.fit(forecast_data)
    
    return model


def forecast_next_month(df, periods=30):
    """
    Forecast future expenses using Prophet.
    
    Args:
        df: Expense DataFrame
        periods: Number of days to forecast (default 30 for next month)
        
    Returns:
        pd.DataFrame: Forecasted expenses with confidence intervals
    """
    model = train_forecast(df)
    
    if model is None:
        return None
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Forecast
    forecast = model.predict(future)
    
    # Return only future predictions
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    forecast_df.columns = ['date', 'forecasted_amount', 'lower_bound', 'upper_bound']
    
    return forecast_df


def forecast_expenses(df, periods=30):
    """
    Alias for forecast_next_month for backward compatibility.
    Forecast future expenses using Prophet.
    
    Args:
        df: Expense DataFrame
        periods: Number of days to forecast
        
    Returns:
        pd.DataFrame: Forecasted expenses with confidence intervals
    """
    return forecast_next_month(df, periods)


def forecast_by_category(df, category, periods=30):
    """
    Forecast expenses for a specific category.
    
    Args:
        df: Expense DataFrame
        category: Category to forecast
        periods: Number of days to forecast
        
    Returns:
        pd.DataFrame: Category-specific forecast
    """
    category_df = df[df['category'] == category].copy()
    
    if len(category_df) < 2:
        return None
    
    return forecast_next_month(category_df, periods)

