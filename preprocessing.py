"""
Data Preprocessing Module
Handles loading, cleaning, and preparing expense data from CSV/Excel files.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_data(file_path):
    """
    Load expense data from CSV or Excel file.
    
    Args:
        file_path: Path to the expense file
        
    Returns:
        pd.DataFrame: Raw expense data
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def clean_data(df):
    """
    Clean and standardize expense data.
    - Normalize column names
    - Guess amount column if not found
    - Convert dates
    - Handle missing values
    - If category missing, keep "unknown"
    
    Args:
        df: Raw expense DataFrame
        
    Returns:
        pd.DataFrame: Cleaned expense data
    """
    df = df.copy()
    
    # Normalize column names (lowercase, strip spaces)
    df.columns = df.columns.str.lower().str.strip()
    
    # Map common column name variations to standard names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        
        # Map amount columns
        if any(term in col_lower for term in ['amount', 'price', 'cost', 'value', 'expense', 'spent']):
            column_mapping[col] = 'amount'
        # Map date columns
        elif any(term in col_lower for term in ['date', 'datetime', 'time', 'transaction_date', 'when']):
            column_mapping[col] = 'date'
        # Map description columns
        elif any(term in col_lower for term in ['description', 'desc', 'details', 'merchant', 'vendor', 'item', 'transaction', 'name', 'store']):
            column_mapping[col] = 'description'
        # Map category columns
        elif any(term in col_lower for term in ['category', 'cat', 'type', 'expense_type', 'tag']):
            column_mapping[col] = 'category'
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Guess amount column if not found
    if 'amount' not in df.columns:
        # Try to find numeric columns that could be amounts
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Use the first numeric column as amount
            df['amount'] = df[numeric_cols[0]]
        else:
            raise ValueError("Could not find or infer amount column")
    
    # Convert amount to numeric, handling currency symbols and commas
    if df['amount'].dtype == 'object':
        df['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False)
        df['amount'] = df['amount'].str.replace(',', '', regex=False)
        df['amount'] = df['amount'].str.replace('€', '', regex=False)
        df['amount'] = df['amount'].str.replace('£', '', regex=False)
        df['amount'] = df['amount'].str.replace('₹', '', regex=False)
        df['amount'] = df['amount'].str.replace(' ', '', regex=False)
    
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['amount'] = df['amount'].abs()  # Make all amounts positive
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # Try to infer date from other columns
        date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_candidates:
            df['date'] = pd.to_datetime(df[date_candidates[0]], errors='coerce')
        else:
            # Create a default date if none found
            df['date'] = pd.Timestamp.now()
    
    # Handle missing values in amount and date
    df = df.dropna(subset=['amount', 'date'])
    df = df[df['amount'] > 0]  # Remove zero or negative amounts
    
    # Handle description column
    if 'description' not in df.columns:
        # Try to create from other text columns
        text_cols = df.select_dtypes(include=['object']).columns
        text_cols = [col for col in text_cols if col not in ['category']]
        if text_cols:
            df['description'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
        else:
            df['description'] = 'Expense'
    
    # Clean description
    df['description'] = df['description'].astype(str).str.strip()
    df['description'] = df['description'].replace('', 'uncategorized')
    
    # Handle category column - keep "unknown" if missing
    if 'category' not in df.columns:
        df['category'] = 'unknown'
    else:
        df['category'] = df['category'].fillna('unknown').astype(str)
        df['category'] = df['category'].replace('', 'unknown')
        df['category'] = df['category'].str.strip()
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    return df


def prepare_features(df):
    """
    Extract and prepare features for ML models.
    - Date features (year, month, day of week)
    - Text features for categorization
    - Statistical features
    
    Args:
        df: Cleaned expense DataFrame
        
    Returns:
        pd.DataFrame: Feature-enhanced DataFrame
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
    
    return df

