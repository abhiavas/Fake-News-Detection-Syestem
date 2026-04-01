"""
Simple Prediction Script - Windows Compatible
Use this to test individual news articles for fake news detection.
"""

import joblib
import re
import string
import os
"""
Simple Prediction Script - Windows Compatible
Use this to test individual news articles for fake news detection.
"""

import joblib
import re
import string
import os

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


class NewsPredictor:
    """Simple predictor for fake news detection."""
    
    def __init__(self, model_path='models/best_model.pkl', 
                 vectorizer_path='models/vectorizer.pkl'):
        """Load the trained model and vectorizer."""
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
    def preprocess_text(self, text):
        """Clean and preprocess text."""
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
        
        # Tokenization and stopword removal
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
        
        return ' '.join(tokens)
    
    def predict(self, text):
        """Predict if news is fake or real."""
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]
        
        result = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        confidence = probability[prediction] * 100
        
        return result, confidence


def main():
    """Interactive prediction mode."""
    print("="*60)
    print("FAKE NEWS DETECTOR - Interactive Mode")
    print("="*60)
    print("\nLoading model...")
    
    try:
        predictor = NewsPredictor()
        print("Model loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nModel files not found. Please train the model first.")
        print("Run: python train_model_simple.py")
        print("\nPress Enter to exit...")
        input()
        return
    
    while True:
        print("\n" + "-"*60)
        print("Enter a news article to check (or 'quit' to exit):")
        print("-"*60)
        
        news_text = input("\n> ")
        
        if news_text.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using Fake News Detector!")
            break
        
        if len(news_text.strip()) < 10:
            print("Please enter a longer text (at least 10 characters).")
            continue
        
        # Make prediction
        result, confidence = predictor.predict(news_text)
        
        # Display result
        print("\n" + "="*60)
        print(f"PREDICTION: {result}")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print("="*60)
        
        if result == "FAKE NEWS":
            print("⚠️  This article shows characteristics of fake news.")
        else:
            print("✓ This article appears to be legitimate news.")


if __name__ == "__main__":
    main()