"""
Fake News Detection Web App using Streamlit - Windows Compatible
Run with: streamlit run app.py
"""

import streamlit as st
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


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer."""
    try:
        # Check if model files exist
        if not os.path.exists('models/best_model.pkl'):
            return None, None
        if not os.path.exists('models/vectorizer.pkl'):
            return None, None
            
        model = joblib.load('models/best_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def preprocess_text(text):
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


def predict_news(model, vectorizer, text):
    """Predict if news is fake or real."""
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    text_vector = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return prediction, probability


def main():
    """Main Streamlit app."""
    
    # Page configuration
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .fake-news {
            color: #ff4444;
            font-size: 28px;
            font-weight: bold;
        }
        .real-news {
            color: #44ff44;
            font-size: 28px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Header
    st.title("📰 Fake News Detection System")
    st.markdown("**Powered by Machine Learning | Detecting Misinformation**")
    st.markdown("---")
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please run train_model_simple.py first to train the model.")
        st.info("Run the following command in your terminal: `python train_model_simple.py`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This application uses Machine Learning to detect fake news articles.
        
        **How it works:**
        1. Enter a news article
        2. Click 'Analyze'
        3. Get instant prediction
        
        **Indicators of Fake News:**
        - Sensational headlines
        - Unnamed sources
        - Poor grammar
        - Emotional language
        - Unverified claims
        """)
        
        st.header("Model Info")
        st.success(f"✓ Model loaded successfully")
        
        st.header("Examples")
        if st.button("Load Fake News Example"):
            st.session_state.example_text = "Scientists discover miracle cure that pharmaceutical companies don't want you to know about! This amazing breakthrough uses only natural ingredients and can cure all diseases instantly."
        
        if st.button("Load Real News Example"):
            st.session_state.example_text = "The Federal Reserve announced today a 0.25% increase in interest rates as part of its ongoing effort to combat inflation. This marks the third rate hike this year."
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter News Article")
        
        # Get example text if available
        default_text = st.session_state.get('example_text', '')
        
        news_text = st.text_area(
            "Paste the news article here:",
            value=default_text,
            height=200,
            placeholder="Enter the news article title and content..."
        )
        
        analyze_button = st.button("🔍 Analyze Article", type="primary")
        
        if analyze_button and news_text:
            if len(news_text.strip()) < 10:
                st.warning("Please enter a longer text (at least 10 characters).")
            else:
                with st.spinner("Analyzing..."):
                    # Make prediction
                    prediction, probability = predict_news(model, vectorizer, news_text)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    if prediction == 1:  # Fake news
                        st.markdown('<p class="fake-news">⚠️ FAKE NEWS DETECTED</p>', 
                                  unsafe_allow_html=True)
                        confidence = probability[1] * 100
                        st.error(f"This article shows strong indicators of fake news.")
                    else:  # Real news
                        st.markdown('<p class="real-news">✓ APPEARS TO BE REAL NEWS</p>', 
                                  unsafe_allow_html=True)
                        confidence = probability[0] * 100
                        st.success(f"This article appears to be legitimate news.")
                    
                    # Confidence meter
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                    st.progress(confidence / 100)
                    
                    # Additional info
                    st.info("""
                    **Note:** This is an automated prediction and should not be the sole basis 
                    for determining the credibility of news. Always verify information from 
                    multiple reliable sources.
                    """)
    
    with col2:
        st.subheader("Quick Tips")
        st.markdown("""
        **Verify News by:**
        - ✓ Checking the source
        - ✓ Reading beyond headlines
        - ✓ Checking the date
        - ✓ Looking for author info
        - ✓ Verifying with fact-checkers
        - ✓ Checking other sources
        
        **Red Flags:**
        - ⚠️ Sensational headlines
        - ⚠️ No author listed
        - ⚠️ No sources cited
        - ⚠️ Poor grammar/spelling
        - ⚠️ Asks you to share immediately
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Fake News Detection System | Built with Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()