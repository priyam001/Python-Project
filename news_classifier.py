"""
News Category Classification
=============================
A complete ML pipeline for classifying news articles into categories
based on their headlines and short descriptions.

Models used:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest
    - Linear SVM (Support Vector Machine)

Author: Priyam
Project: Capstone Project 2020 (Updated 2026)
"""

import os
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# NLTK setup — download required data quietly
# ---------------------------------------------------------------------------
import nltk


def _ensure_nltk_data():
    """Download required NLTK data if not already present."""
    packages = ["stopwords", "wordnet"]
    for pkg in packages:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except Exception:
            # Not found or corrupted — try to download
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass  # Ignore download errors (e.g., file locks)


_ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ===========================================================================
# 1. DATA LOADING & EXPLORATION
# ===========================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """Load and return the news category dataset."""
    print("=" * 70)
    print("STEP 1: DATA LOADING & EXPLORATION")
    print("=" * 70)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please ensure the CSV file is in the project directory."
        )

    df = pd.read_csv(filepath)

    print(f"\nDataset loaded successfully!")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nNumber of unique categories: {df['category'].nunique()}")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())

    return df


# ===========================================================================
# 2. DATA PREPROCESSING
# ===========================================================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataframe."""
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Handle missing values in 'authors' column
    missing_authors = df["authors"].isnull().sum()
    print(f"\n  Missing 'authors' values: {missing_authors}")
    df["authors"] = df["authors"].fillna("Unknown")

    # Combine headline and short_description into a single text feature
    df["text"] = df["headline"].astype(str) + " " + df["short_description"].astype(str)
    print(f"  Created 'text' feature (headline + short_description)")

    # Drop columns not needed for classification
    cols_to_drop = ["headline", "short_description", "authors", "link", "date"]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped columns: {cols_to_drop}")
    print(f"  Remaining columns: {list(df.columns)}")
    print(f"  Final shape: {df.shape}")

    return df


# ===========================================================================
# 3. TEXT CLEANING
# ===========================================================================
def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """Clean a single text string: lowercase, remove special chars, lemmatize."""
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove special characters, numbers, and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the entire dataframe."""
    print("\n" + "=" * 70)
    print("STEP 3: TEXT CLEANING")
    print("=" * 70)

    df = df.copy()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    print("  Applying text cleaning (lowercase, remove punctuation,")
    print("  remove stopwords, lemmatize)...")

    df["text_clean"] = df["text"].apply(
        lambda x: clean_text(x, lemmatizer, stop_words)
    )

    # Show a sample
    print(f"\n  Sample original text:")
    print(f"    '{df['text'].iloc[0][:100]}...'")
    print(f"  Sample cleaned text:")
    print(f"    '{df['text_clean'].iloc[0][:100]}...'")

    # Remove any rows where cleaned text is empty
    empty_count = (df["text_clean"].str.len() == 0).sum()
    if empty_count > 0:
        print(f"  Removed {empty_count} rows with empty cleaned text")
        df = df[df["text_clean"].str.len() > 0]

    return df


# ===========================================================================
# 4. FEATURE ENGINEERING (TF-IDF)
# ===========================================================================
def vectorize_text(X_train, X_test, max_features=5000):
    """Apply TF-IDF vectorization to the text data."""
    print("\n" + "=" * 70)
    print("STEP 4: TF-IDF VECTORIZATION")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"  Max features: {max_features}")
    print(f"  N-gram range: (1, 2)")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Train matrix shape: {X_train_tfidf.shape}")
    print(f"  Test matrix shape:  {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ===========================================================================
# 5. TRAIN / TEST SPLIT
# ===========================================================================
def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Split data into training and testing sets with stratification."""
    print("\n" + "=" * 70)
    print("STEP 5: TRAIN / TEST SPLIT")
    print("=" * 70)

    X = df["text_clean"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Test size: {test_size * 100:.0f}%")
    print(f"  Random state: {random_state}")
    print(f"  Stratified: Yes")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ===========================================================================
# 6. MODEL TRAINING & EVALUATION
# ===========================================================================
def get_models() -> dict:
    """Return a dictionary of models to train and evaluate."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, class_weight="balanced"
        ),
        "Multinomial NB": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Linear SVM": LinearSVC(
            max_iter=2000, C=1.0, random_state=42, class_weight="balanced"
        ),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train each model, evaluate, and return results."""
    print("\n" + "=" * 70)
    print("STEP 6: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    results = {}

    for name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"  Training: {name}")
        print(f"{'-' * 50}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "y_pred": y_pred,
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    return results


# ===========================================================================
# 7. VISUALIZATION
# ===========================================================================
def plot_category_distribution(df: pd.DataFrame, save_path="category_distribution.png"):
    """Plot the category distribution as a bar chart."""
    plt.figure(figsize=(14, 6))
    counts = df["category"].value_counts()
    colors = sns.color_palette("viridis", len(counts))

    ax = counts.plot(kind="bar", color=colors, edgecolor="black", linewidth=0.5)
    plt.title("News Category Distribution", fontsize=18, fontweight="bold")
    plt.xlabel("Category", fontsize=14)
    plt.ylabel("Number of Articles", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()

    # Add value labels on bars
    for i, (val, name) in enumerate(zip(counts.values, counts.index)):
        ax.text(i, val + 2, str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results: dict, save_path="model_comparison.png"):
    """Plot a comparison of all models' performance metrics."""
    print("\n" + "=" * 70)
    print("STEP 7: VISUALIZATION")
    print("=" * 70)

    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    data = {label: [results[m][metric] for m in model_names] for metric, label in zip(metrics, metric_labels)}

    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (label, values) in enumerate(data.items()):
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i], edgecolor="black", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("Models", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Model Performance Comparison", fontsize=18, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(y_test, y_pred, model_name, labels, save_path=None):
    """Plot a confusion matrix for a single model."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted", fontsize=13)
    plt.ylabel("Actual", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ===========================================================================
# 8. SAVE BEST MODEL
# ===========================================================================
def save_best_model(results, vectorizer, save_dir="."):
    """Save the best performing model and the vectorizer."""
    print("\n" + "=" * 70)
    print("STEP 8: SAVING BEST MODEL")
    print("=" * 70)

    # Find best model by F1 score
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_result = results[best_name]

    print(f"\n  Best model: {best_name}")
    print(f"  F1 Score:   {best_result['f1_score']:.4f}")
    print(f"  Accuracy:   {best_result['accuracy']:.4f}")

    model_path = os.path.join(save_dir, "best_model.joblib")
    vectorizer_path = os.path.join(save_dir, "tfidf_vectorizer.joblib")

    joblib.dump(best_result["model"], model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\n  Model saved to:      {model_path}")
    print(f"  Vectorizer saved to: {vectorizer_path}")

    return best_name, best_result


# ===========================================================================
# 9. PREDICTION FUNCTION
# ===========================================================================
def predict_category(text: str, model, vectorizer, lemmatizer, stop_words):
    """Predict the category of a given text."""
    cleaned = clean_text(text, lemmatizer, stop_words)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]


def interactive_predictions(best_model, vectorizer):
    """Run interactive predictions on user-provided headlines."""
    print("\n" + "=" * 70)
    print("STEP 9: INTERACTIVE PREDICTIONS")
    print("=" * 70)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Demo predictions with sample headlines
    sample_headlines = [
        "10 Easy Recipes for a Healthy Dinner Tonight",
        "How to Plan the Perfect Beach Vacation in 2024",
        "Celebrity Couple Announces Divorce After 10 Years",
        "New Study Reveals Benefits of Daily Meditation",
        "Best Makeup Trends for Spring Fashion Week",
        "Tips for Getting Your Toddler to Sleep Through the Night",
    ]

    print("\n  Sample Predictions:")
    print("  " + "-" * 60)
    for headline in sample_headlines:
        category = predict_category(
            headline, best_model, vectorizer, lemmatizer, stop_words
        )
        print(f"  [NEWS] \"{headline}\"")
        print(f"     -> Predicted: {category}")
        print()


# ===========================================================================
# 10. RESULTS SUMMARY
# ===========================================================================
def print_results_summary(results):
    """Print a formatted summary table of all model results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print("  " + "-" * 65)

    for name, res in results.items():
        print(
            f"  {name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
            f"{res['recall']:>10.4f} {res['f1_score']:>10.4f}"
        )

    best_name = max(results, key=lambda k: results[k]["f1_score"])
    print("  " + "-" * 65)
    print(f"  [BEST] Best Model: {best_name} (F1: {results[best_name]['f1_score']:.4f})")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    """Execute the complete news classification pipeline."""
    print("\n" + "+" + "=" * 68 + "+")
    print("|" + " NEWS CATEGORY CLASSIFICATION PIPELINE ".center(68) + "|")
    print("+" + "=" * 68 + "+")

    # Determine dataset path
    dataset_file = "news_category_dataset.csv"
    if not os.path.exists(dataset_file):
        # Fallback to original filename
        dataset_file = "news_catagery_dataset.csv"

    # Step 1: Load data
    df = load_data(dataset_file)

    # Step 2: Preprocess data
    df = preprocess_data(df)

    # Step 3: Clean text
    df = apply_text_cleaning(df)

    # Visualize category distribution
    plot_category_distribution(df, save_path="category_distribution.png")

    # Step 4 & 5: Split data, then vectorize
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)

    # Step 6: Train and evaluate models
    models = get_models()
    results = train_and_evaluate(models, X_train_tfidf, X_test_tfidf, y_train, y_test)

    # Step 7: Visualizations
    plot_model_comparison(results, save_path="model_comparison.png")

    # Confusion matrices for each model
    labels = sorted(df["category"].unique())
    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_test,
            res["y_pred"],
            name,
            labels,
            save_path=f"confusion_matrix_{safe_name}.png",
        )

    # Step 8: Save best model
    best_name, best_result = save_best_model(results, vectorizer)

    # Step 9: Interactive predictions
    interactive_predictions(best_result["model"], vectorizer)

    # Step 10: Results summary
    print_results_summary(results)

    print("\n" + "+" + "=" * 68 + "+")
    print("|" + " PIPELINE COMPLETE ".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print()


if __name__ == "__main__":
    main()
