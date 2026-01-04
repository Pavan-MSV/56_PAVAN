"""
Data Cleaner Module
Handles cleaning and preprocessing of expense data.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def clean_expense_data(df):
    """
    Clean and preprocess expense data.
    
    Args:
        df: pd.DataFrame with raw expense data
        
    Returns:
        pd.DataFrame with cleaned data
    """
    df = df.copy()
    
    # Standardize date column
    df = _clean_dates(df)
    
    # Clean amount column
    df = _clean_amounts(df)
    
    # Clean description column
    df = _clean_descriptions(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['amount', 'date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def _clean_dates(df):
    """Clean and convert date column to datetime."""
    if 'date' not in df.columns:
        return df
    
    # Try to convert to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['date'])
    
    return df


def _clean_amounts(df):
    """Clean amount column - handle different formats."""
    if 'amount' not in df.columns:
        return df
    
    # Convert to string first to handle mixed types
    df['amount'] = df['amount'].astype(str)
    
    # Remove currency symbols and commas
    df['amount'] = df['amount'].str.replace('$', '', regex=False)
    df['amount'] = df['amount'].str.replace(',', '', regex=False)
    df['amount'] = df['amount'].str.replace('€', '', regex=False)
    df['amount'] = df['amount'].str.replace('£', '', regex=False)
    df['amount'] = df['amount'].str.replace('₹', '', regex=False)
    
    # Convert to float
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Remove negative signs (convert to positive for expenses)
    df['amount'] = df['amount'].abs()
    
    # Remove rows with invalid amounts
    df = df.dropna(subset=['amount'])
    df = df[df['amount'] > 0]  # Only positive amounts
    
    return df


def _clean_descriptions(df):
    """Clean description column."""
    if 'description' not in df.columns:
        # Create description column from other columns if possible
        desc_cols = [col for col in df.columns if col not in ['amount', 'date', 'category']]
        if desc_cols:
            df['description'] = df[desc_cols].fillna('').astype(str).agg(' '.join, axis=1)
        else:
            df['description'] = 'Expense'
    
    # Clean description text
    df['description'] = df['description'].astype(str)
    df['description'] = df['description'].str.strip()
    df['description'] = df['description'].str.lower()
    
    # Replace empty descriptions
    df['description'] = df['description'].replace('', 'uncategorized')
    
    return df


def extract_features(df):
    """
    Extract additional features from the cleaned data.
    
    Args:
        df: pd.DataFrame with cleaned expense data
        
    Returns:
        pd.DataFrame with additional features
    """
    df = df.copy()
    
    # Extract date features
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.day_name()
        df['month_name'] = df['date'].dt.strftime('%B')
        df['year_month'] = df['date'].dt.to_period('M').astype(str)
    
    # Create category column if it doesn't exist
    if 'category' not in df.columns:
        df['category'] = 'Uncategorized'
    
    return df


