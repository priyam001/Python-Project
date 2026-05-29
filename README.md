# 📰 News Category Classification

A machine learning project that classifies news articles into categories based on their **headlines** and **short descriptions**, using Natural Language Processing (NLP) and multiple classification algorithms.

## 📋 Problem Statement

Can you categorize news articles based on their headlines and short descriptions? Do news articles from different categories have different writing styles?

A classifier trained on this dataset can be used on free text to identify the type of news it represents — enabling automated content categorization, recommendation systems, and media trend analysis.

## 📊 Dataset

The dataset contains **999 news articles** from [HuffPost](https://www.huffingtonpost.com/) spanning **19 categories**, with the following features:

| Column | Description |
|--------|-------------|
| `category` | News category label (target variable) |
| `headline` | Article headline |
| `authors` | Author name(s) |
| `link` | URL to the article |
| `short_description` | Brief summary of the article |
| `date` | Publication date |

### Category Distribution (Top 5)
- **STYLE & BEAUTY**: 279 articles
- **WELLNESS**: 218 articles
- **PARENTING**: 87 articles
- **TRAVEL**: 75 articles
- **WEDDINGS / HOME & LIVING**: 72 articles each

> ⚠️ The dataset is **imbalanced** — some categories like CRIME (3) and CULTURE & ARTS (4) have very few samples.

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/priyam001/Python-Project.git
cd Python-Project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (first run only)
The script will automatically download the required NLTK data (stopwords, wordnet) on the first run.

## 💻 Usage

### Run the classifier
```bash
python news_classifier.py
```

This will:
1. Load and explore the dataset
2. Preprocess and clean the text data
3. Vectorize text using TF-IDF
4. Train 4 different ML models
5. Evaluate and compare model performance
6. Save the best model to disk
7. Run interactive predictions

### Project Output
- Model comparison metrics (Accuracy, Precision, Recall, F1)
- Classification reports per model
- Confusion matrix visualizations (saved as PNG)
- Best model saved as `.joblib` file

## 📁 Project Structure

```
Python-Project/
├── .deepsource.toml            # DeepSource static analysis config
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── problem_statement.txt       # Detailed problem statement
├── news_category_dataset.csv   # HuffPost news dataset
├── news_classifier.py          # Main ML pipeline script
└── Capstone Project.pptx       # Project presentation
```

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear model, fast and interpretable |
| **Multinomial Naive Bayes** | Probabilistic, excellent for text classification |
| **Random Forest** | Ensemble of decision trees |
| **Linear SVM** | Support Vector Machine with linear kernel |

## 🛠️ Technologies

- **Python 3.x**
- **pandas** — Data manipulation
- **NumPy** — Numerical operations
- **scikit-learn** — ML models, TF-IDF, evaluation metrics
- **NLTK** — Text preprocessing (stopwords, lemmatization)
- **matplotlib / seaborn** — Data visualization

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset sourced from [HuffPost News](https://www.huffingtonpost.com/) via Kaggle
- Built as a Capstone Project (2020)
