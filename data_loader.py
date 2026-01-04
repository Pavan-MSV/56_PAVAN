"""
Data Loader Module
Handles loading and parsing of CSV and Excel expense files.
"""

import pandas as pd
import streamlit as st
from io import BytesIO


def load_expense_file(uploaded_file):
    """
    Load expense data from uploaded CSV or Excel file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        pd.DataFrame: Loaded expense data
    """
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Read file based on extension
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def validate_expense_data(df):
    """
    Validate that the loaded DataFrame has required columns.
    
    Args:
        df: pd.DataFrame to validate
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    # Check for common expense-related columns (case-insensitive)
    df_lower = df.columns.str.lower()
    
    # Required: amount or price
    has_amount = any(col in df_lower for col in ['amount', 'price', 'cost', 'value'])
    
    # Required: date or datetime
    has_date = any(col in df_lower for col in ['date', 'datetime', 'time'])
    
    if not has_amount:
        return False, "Missing required column: amount/price/cost/value"
    
    if not has_date:
        return False, "Missing required column: date/datetime/time"
    
    return True, "Data validation passed"


def standardize_column_names(df):
    """
    Standardize column names to a common format.
    
    Args:
        df: pd.DataFrame with original column names
        
    Returns:
        pd.DataFrame with standardized column names
    """
    df = df.copy()
    
    # Create mapping for common column name variations
    column_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Map amount columns
        if col_lower in ['amount', 'price', 'cost', 'value', 'expense']:
            column_mapping[col] = 'amount'
        
        # Map date columns
        elif col_lower in ['date', 'datetime', 'time', 'transaction_date']:
            column_mapping[col] = 'date'
        
        # Map description columns
        elif col_lower in ['description', 'desc', 'details', 'merchant', 'vendor', 'item', 'transaction']:
            column_mapping[col] = 'description'
        
        # Map category columns (if exists)
        elif col_lower in ['category', 'cat', 'type', 'expense_type']:
            column_mapping[col] = 'category'
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    return df


