"""
Model Training Module
Trains and manages the categorization model using XGBoost + TF-IDF.
"""

import pickle
import os
import pandas as pd
from core.categorization import ExpenseCategorizer


def train_categorization_model(df):
    """
    Train the expense categorization model on historical data.
    
    Args:
        df: Expense DataFrame with descriptions and categories
        
    Returns:
        ExpenseCategorizer: Trained model
    """
    # Filter out 'unknown' category for training
    train_df = df[df['category'] != 'unknown'].copy()
    
    if len(train_df) < 10:
        # Not enough data to train, return untrained model
        return ExpenseCategorizer()
    
    descriptions = train_df['description'].tolist()
    categories = train_df['category'].tolist()
    
    categorizer = ExpenseCategorizer()
    categorizer.train(descriptions, categories)
    
    return categorizer


def save_model(categorizer, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    classifier_path = os.path.join(model_dir, 'classifier.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(categorizer.vectorizer, f)

    with open(classifier_path, 'wb') as f:
        pickle.dump(categorizer.model, f)

    with open(encoder_path, 'wb') as f:
        pickle.dump(categorizer.label_encoder, f)

