import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class ExpenseCategorizer:
    """Categorizes expenses using TF-IDF and XGBoost."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = XGBClassifier(random_state=42, eval_metric='mlogloss')
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, descriptions, categories):
        if isinstance(descriptions, pd.Series):
            descriptions = descriptions.tolist()
        if isinstance(categories, pd.Series):
            categories = categories.tolist()

        # Fit vectorizer
        X = self.vectorizer.fit_transform(descriptions)

        # Encode labels
        y = self.label_encoder.fit_transform(categories)

        # Train classifier
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, descriptions):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if isinstance(descriptions, pd.Series):
            descriptions = descriptions.tolist()

        X = self.vectorizer.transform(descriptions)

        preds = self.model.predict(X)

        # Decode numeric labels back to text
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, descriptions):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if isinstance(descriptions, pd.Series):
            descriptions = descriptions.tolist()

        X = self.vectorizer.transform(descriptions)
        return self.model.predict_proba(X)


def load_model(model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    classifier_path = os.path.join(model_dir, 'classifier.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

    if not (os.path.exists(vectorizer_path) and os.path.exists(classifier_path) and os.path.exists(encoder_path)):
        return None

    categorizer = ExpenseCategorizer()

    with open(vectorizer_path, 'rb') as f:
        categorizer.vectorizer = pickle.load(f)

    with open(classifier_path, 'rb') as f:
        categorizer.model = pickle.load(f)

    with open(encoder_path, 'rb') as f:
        categorizer.label_encoder = pickle.load(f)

    categorizer.is_trained = True
    return categorizer
