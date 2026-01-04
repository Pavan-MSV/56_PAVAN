"""
Anomaly Detection Module
Detects unusual transactions and spending patterns.
"""

import pandas as pd
import numpy as np


def detect_anomalies(df):
    """
    Detect anomalous expense transactions using statistical method.
    Mark transactions whose amount > mean + 2*std.
    
    Args:
        df: Expense DataFrame
        
    Returns:
        pd.DataFrame: Expenses with anomaly flags
    """
    df = df.copy()
    
    if len(df) == 0:
        df['is_anomaly'] = False
        return df
    
    # Calculate mean and standard deviation
    mean_amount = df['amount'].mean()
    std_amount = df['amount'].std()
    
    # Threshold: mean + 2*std
    threshold = mean_amount + 2 * std_amount
    
    # Mark anomalies
    df['is_anomaly'] = df['amount'] > threshold
    
    return df


def detect_spending_spikes(df, window=7):
    """
    Detect sudden spikes in spending.
    
    Args:
        df: Expense DataFrame
        window: Time window in days for spike detection
        
    Returns:
        pd.DataFrame: Dates and amounts of detected spikes
    """
    df = df.copy()
    df = df.sort_values('date')
    
    # Calculate rolling mean
    df['rolling_mean'] = df['amount'].rolling(window=window, center=True).mean()
    df['rolling_std'] = df['amount'].rolling(window=window, center=True).std()
    
    # Identify spikes (amount > rolling_mean + 2*rolling_std)
    df['is_spike'] = df['amount'] > (df['rolling_mean'] + 2 * df['rolling_std'].fillna(0))
    
    spikes = df[df['is_spike']][['date', 'amount', 'description', 'category']].copy()
    
    return spikes


def detect_category_anomalies(df):
    """
    Detect unusual spending in specific categories.
    
    Args:
        df: Expense DataFrame with categories
        
    Returns:
        pd.DataFrame: Anomalous category spending
    """
    anomalies_by_category = []
    
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        
        if len(category_df) < 3:
            continue
        
        mean_amount = category_df['amount'].mean()
        std_amount = category_df['amount'].std()
        threshold = mean_amount + 2 * std_amount
        
        category_anomalies = category_df[category_df['amount'] > threshold].copy()
        category_anomalies['anomaly_reason'] = f"Amount exceeds category average by 2 standard deviations"
        
        anomalies_by_category.append(category_anomalies)
    
    if anomalies_by_category:
        return pd.concat(anomalies_by_category, ignore_index=True)
    else:
        return pd.DataFrame()

