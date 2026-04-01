# 📰 Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

A Machine Learning based system to detect whether a news article is **Real or Fake** using Natural Language Processing (NLP) and multiple classification algorithms.

---

## 🚀 Features

* 🔍 Detect fake vs real news instantly
* 🤖 Multiple ML models (Logistic Regression, Naive Bayes, Random Forest)
* 📊 Model comparison with metrics
* 🌐 Streamlit web app for easy interaction
* 💻 Command-line prediction tool
* 📈 Confusion matrix & performance visualization

---

## 📂 Project Structure

```
fake-news-detection/
│
├── data/
│   └── news_dataset.csv
│
├── models/
│   ├── best_model.pkl
│   └── vectorizer.pkl
│
├── images/
│   ├── confusion_matrices.png
│   └── model_comparison.png
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Results & Visualizations

### 🔹 Confusion Matrix

![Confusion Matrix](images/confusion_matrices.png)

### 🔹 Model Performance Comparison

![Model Comparison](images/model_comparison.png)

---

## 🤖 Models Used

| Model               | Description                    |
| ------------------- | ------------------------------ |
| Logistic Regression | Fast, accurate, best performer |
| Naive Bayes         | Probabilistic baseline model   |
| Random Forest       | Ensemble learning method       |

---

## 📈 Performance Metrics

* Accuracy
* Precision
* Recall
* F1-Score

👉 Best Model: **Logistic Regression (100% accuracy on test data)**

---

## 🔍 How It Works

### 1. Text Preprocessing

* Convert to lowercase
* Remove punctuation, URLs, numbers
* Remove stopwords

### 2. Feature Extraction

* TF-IDF Vectorization
* Bigrams support

### 3. Classification

* Input text → Vector → Model → Prediction

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🌐 Run Web App (Recommended)

```bash
streamlit run app.py
```

Open: http://localhost:8501

---

### 💻 Run Command Line Tool

```bash
python predict.py
```

---

### 🔄 Train Model

```bash
python train_model.py
```

---

## 💡 Example

**Input:**

```
Scientists discover miracle cure for all diseases!
```

**Output:**

```
⚠️ FAKE NEWS
Confidence: 98%
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas & NumPy
* Matplotlib & Seaborn
* Streamlit

---

## ⚠️ Limitations

* Small dataset (demo purpose)
* English language only
* Cannot verify factual correctness (pattern-based)

---

## 🔮 Future Improvements

* Add deep learning (BERT, LSTM)
* Larger dataset
* Real-time news API integration
* Multi-language support

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve.

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Abhishek Avasthi, Jatin Yadav, Shlok**

---

⭐ If you like this project, don’t forget to star the repo!
