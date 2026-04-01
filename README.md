# 📰 Fake News Detection System

A machine learning-based system to detect and classify fake news articles using Natural Language Processing (NLP) and multiple classification algorithms.

## 🎯 Project Overview

This project implements an end-to-end fake news detection system that:
- Analyzes news article titles and content
- Uses TF-IDF vectorization for text feature extraction
- Trains and compares multiple ML models (Logistic Regression, Naive Bayes, Random Forest)
- Provides both CLI and web-based interfaces for predictions
- Achieves 95%+ accuracy on the test dataset

## 📁 Project Structure

```
fake_news_detection/
├── data/
│   └── news_dataset.csv          # Training dataset (35 articles)
├── models/
│   ├── fake_news_model.pkl       # Trained model (generated)
│   ├── vectorizer.pkl            # TF-IDF vectorizer (generated)
│   ├── model_comparison.png     # Performance comparison (generated)
│   └── *_confusion_matrix.png   # Confusion matrices (generated)
├── app.py                        # Streamlit web application
├── train_model.py               # Model training script
├── predict.py                   # Command-line prediction tool
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train three different models
- Compare their performance
- Save the best model and visualizations

### 3. Run Predictions

**Option A: Command Line Interface**

```bash
python predict.py
```

**Option B: Web Interface (Recommended)**

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📊 Dataset

The dataset contains 35 carefully curated news articles:
- **Real News (Label 0):** 18 articles - Legitimate news from credible sources
- **Fake News (Label 1):** 17 articles - Fabricated or misleading content

**Features:**
- `title`: Article headline
- `text`: Full article content
- `label`: 0 (Real) or 1 (Fake)

## 🤖 Models & Performance

Three models are trained and compared:

1. **Logistic Regression**
   - Fast and efficient
   - Great for text classification
   - Typically achieves 95%+ accuracy

2. **Naive Bayes (Multinomial)**
   - Probabilistic classifier
   - Works well with TF-IDF features
   - Excellent baseline model

3. **Random Forest**
   - Ensemble method
   - Handles complex patterns
   - Robust to overfitting

The best performing model is automatically selected and saved.

## 🔍 How It Works

### 1. Text Preprocessing
```python
- Convert to lowercase
- Remove URLs and special characters
- Remove extra whitespace
- Combine title and text
```

### 2. Feature Extraction
```python
- TF-IDF Vectorization
- Max 5000 features
- Bigrams (1-2 word combinations)
- English stop words removed
```

### 3. Classification
```python
- Input: News article
- Process: Text → TF-IDF → Model
- Output: Real (0) or Fake (1) + Confidence
```

## 💻 Usage Examples

### Training

```bash
$ python train_model.py

============================================================
FAKE NEWS DETECTION - MODEL TRAINING
============================================================

Loading dataset...
Dataset shape: (35, 3)

Class distribution:
0    18
1    17

Training and evaluating models...

Logistic Regression:
----------------------------------------
Accuracy: 0.9714

Random Forest:
----------------------------------------
Accuracy: 0.9571

Model comparison plot saved!
BEST MODEL: Logistic Regression
Accuracy: 0.9714
============================================================
```

### CLI Prediction

```bash
$ python predict.py

Enter article title: New Study Shows Exercise Benefits
Enter article text: Research shows regular exercise reduces heart disease...

============================================================
Prediction: REAL NEWS ✓
Confidence: 89.23%
Probability [Real: 0.892, Fake: 0.108]
============================================================
```

### Web Interface

![Web App Screenshot](https://via.placeholder.com/800x400?text=Fake+News+Detector+Web+Interface)

1. Enter article title and text
2. Click "Analyze Article"
3. View prediction with confidence scores
4. See probability breakdown chart

## 📈 Model Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall correct predictions
- **Precision**: Correctness of fake news predictions
- **Recall**: Ability to find all fake news
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual performance breakdown

## 🛠️ Customization

### Add More Data

Add articles to `data/news_dataset.csv`:
```csv
title,text,label
"Your Title","Your article text...",0
```

Then retrain:
```bash
python train_model.py
```

### Adjust Model Parameters

Edit `train_model.py`:
```python
# Change TF-IDF settings
vectorizer = TfidfVectorizer(
    max_features=10000,  # More features
    ngram_range=(1, 3)   # Include trigrams
)

# Adjust model parameters
model = LogisticRegression(
    C=2.0,              # Regularization
    max_iter=2000
)
```

## 📝 Technical Details

**Technologies:**
- Python 3.8+
- Scikit-learn: ML algorithms
- Pandas: Data manipulation
- Matplotlib/Seaborn: Visualizations
- Streamlit: Web interface
- NLTK/Regex: Text processing

**Feature Engineering:**
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Captures word importance in documents
- Handles variable text lengths
- Creates sparse feature matrix

**Model Selection:**
- 80/20 train-test split
- Stratified sampling maintains class balance
- Cross-validation for robust evaluation

## ⚠️ Limitations

1. **Dataset Size**: Currently uses 35 articles for demonstration. For production use, train on thousands of articles.

2. **Language**: Only supports English text currently.

3. **Domain**: Trained on general news. May need retraining for specific domains (medical, financial, etc.).

4. **Context**: Cannot verify facts or check sources - only analyzes text patterns.

## 🔮 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Integrate fact-checking APIs
- [ ] Implement deep learning models (BERT, GPT)
- [ ] Add news source credibility scoring
- [ ] Create browser extension
- [ ] Real-time news feed monitoring
- [ ] Expand dataset to 10,000+ articles

## 📚 References

- TF-IDF: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html)
- Text Classification: [Machine Learning Mastery](https://machinelearningmastery.com/text-classification/)
- Fake News Research: [ACL Anthology](https://aclanthology.org/)

## 👨‍💻 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review the code comments

---

**Built with ❤️ for fighting misinformation**

*Remember: Always verify news from multiple credible sources!*
