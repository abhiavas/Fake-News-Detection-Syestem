"""
Fake News Detection System - Windows Compatible Version
This script trains multiple machine learning models to detect fake news.
"""

import pandas as pd
import numpy as np
import re
import string
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Define common English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
    "won't", 'wouldn', "wouldn't"
}

class FakeNewsDetector:
    """
    A comprehensive fake news detection system with multiple ML models.
    """
    
    def __init__(self, data_path):
        """Initialize the detector with data path."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        
    def load_data(self):
        """Load the dataset."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nDataset info:")
        print(self.df.info())
        print(f"\nLabel distribution:")
        print(self.df['label'].value_counts())
        return self.df
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text data.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Simple tokenization and stopword removal
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
        
        return ' '.join(tokens)
    
    def prepare_data(self):
        """Prepare and preprocess the data."""
        print("\nPreprocessing text data...")
        
        # Combine title and text
        self.df['content'] = self.df['title'] + ' ' + self.df['text']
        
        # Apply preprocessing
        self.df['cleaned_content'] = self.df['content'].apply(self.preprocess_text)
        
        # Split features and labels
        X = self.df['cleaned_content']
        y = self.df['label']
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
    def vectorize_text(self, max_features=5000):
        """Convert text to TF-IDF features."""
        print("\nVectorizing text using TF-IDF...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)
        
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Feature matrix shape: {self.X_train.shape}")
        
    def train_models(self):
        """Train multiple classification models."""
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            print(f"{name} trained successfully!")
            
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            print("-" * 40)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=['Real', 'Fake']))
            
        # Create results dataframe
        results_df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("SUMMARY OF ALL MODELS")
        print("="*50)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'],
                       yticklabels=['Real', 'Fake'],
                       ax=axes[idx])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        # Use current directory - works on Windows!
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrices saved to 'confusion_matrices.png'")
        plt.close()
        
    def plot_model_comparison(self, results_df):
        """Plot comparison of model performances."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(results_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, results_df[metric], width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        # Use current directory - works on Windows!
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("Model comparison chart saved to 'model_comparison.png'")
        plt.close()
        
    def save_best_model(self):
        """Save the best performing model."""
        # Find best model based on F1-score
        best_model_name = None
        best_f1 = 0
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
        
        print(f"\nBest model: {best_model_name} (F1-Score: {best_f1:.4f})")
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Created 'models' folder")
        
        # Save the best model and vectorizer - use current directory!
        joblib.dump(self.models[best_model_name], 'models/best_model.pkl')
        joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
        
        print("Best model and vectorizer saved successfully!")
        
    def predict_news(self, text):
        """Predict if a news article is fake or real."""
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Predict using best model
        prediction = list(self.models.values())[0].predict(text_vector)[0]
        probability = list(self.models.values())[0].predict_proba(text_vector)[0]
        
        result = "FAKE" if prediction == 1 else "REAL"
        confidence = probability[prediction] * 100
        
        return result, confidence


def main():
    """Main execution function."""
    print("="*50)
    print("FAKE NEWS DETECTION SYSTEM")
    print("="*50)
    
    # Initialize detector
    detector = FakeNewsDetector('data/news_dataset.csv')
    
    # Load and explore data
    detector.load_data()
    
    # Prepare data
    detector.prepare_data()
    
    # Vectorize text
    detector.vectorize_text()
    
    # Train models
    detector.train_models()
    
    # Evaluate models
    results_df = detector.evaluate_models()
    
    # Create visualizations
    detector.plot_confusion_matrices()
    detector.plot_model_comparison(results_df)
    
    # Save best model
    detector.save_best_model()
    
    print("\n" + "="*50)
    print("TESTING PREDICTION ON SAMPLE NEWS")
    print("="*50)
    
    # Test predictions
    test_samples = [
        "Scientists discover miracle cure that pharmaceutical companies don't want you to know about!",
        "The Federal Reserve announced today an interest rate increase of 0.25 percent to address inflation concerns.",
        "Shocking truth revealed: Moon landing was faked in Hollywood studio!"
    ]
    
    for sample in test_samples:
        result, confidence = detector.predict_news(sample)
        print(f"\nNews: {sample[:80]}...")
        print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nSaved files:")
    print("- models/best_model.pkl")
    print("- models/vectorizer.pkl")
    print("- confusion_matrices.png")
    print("- model_comparison.png")


if __name__ == "__main__":
    main()