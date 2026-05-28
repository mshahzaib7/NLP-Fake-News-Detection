

import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()



def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_series(series):
    processed = []
    for t in tqdm(series.astype(str), desc="Preprocessing"):
        t = clean_text(t)
        tokens = nltk.word_tokenize(t)
        tokens = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
        processed.append(" ".join(tokens))
    return pd.Series(processed, index=series.index)



def generate_wordclouds(df, text_col, label_col, out_dir="wordclouds"):
    os.makedirs(out_dir, exist_ok=True)
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        text = " ".join(subset[text_col])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        wc.to_file(os.path.join(out_dir, f"{label}_wordcloud.png"))
    print(f"Wordclouds saved in {out_dir}/")


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC()
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, pos_label=1)
        print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        print(classification_report(y_test, preds))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Fake News Detection Script")
    parser.add_argument("--data_path", required=True, help="Path to Fake/Real news CSV dataset")
    parser.add_argument("--title_col", default="title", help="Column name for title")
    parser.add_argument("--text_col", default="text", help="Column name for content/body")
    parser.add_argument("--label_col", default="label", help="Column name for label (Fake/Real)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.data_path)

    # Basic column checks
    for col in [args.title_col, args.text_col, args.label_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Combine title + text
    df["combined_text"] = (df[args.title_col].fillna("") + " " + df[args.text_col].fillna("")).str.strip()

    # Normalize labels
    df[args.label_col] = df[args.label_col].str.lower().map({"fake": 0, "real": 1})
    if df[args.label_col].isnull().any():
        raise ValueError("Label column must contain only 'Fake' or 'Real' values.")

    # Preprocess text
    df["clean_text"] = preprocess_series(df["combined_text"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df[args.label_col],
        test_size=args.test_size, random_state=args.random_state, stratify=df[args.label_col]
    )

    # TF-IDF
    print("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train + Evaluate
    train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)

    # Wordclouds
    print("\nGenerating word clouds...")
    generate_wordclouds(df, "clean_text", args.label_col)


if __name__ == "__main__":
    main()
