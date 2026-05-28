
import argparse
import os
import re
import string
from collections import Counter, defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm

# NLP and Neural network
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from wordcloud import WordCloud

# Ensure resources
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


def plot_top_words(df, text_col, label_col, n=20):
    categories = df[label_col].unique()
    fig, axes = plt.subplots(len(categories), 1, figsize=(10, 5 * len(categories)))

    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        text = " ".join(df[df[label_col] == cat][text_col])
        tokens = text.split()
        counter = Counter(tokens)
        common = counter.most_common(n)
        words, counts = zip(*common)
        ax.barh(words[::-1], counts[::-1])
        ax.set_title(f"Top {n} words in '{cat}'")
        ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig("top_words_per_category.png")
    plt.show()


def generate_wordclouds(df, text_col, label_col, out_dir="wordclouds"):
    os.makedirs(out_dir, exist_ok=True)
    for cat in df[label_col].unique():
        text = " ".join(df[df[label_col] == cat][text_col])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        wc.to_file(os.path.join(out_dir, f"{cat}_wordcloud.png"))
    print(f"Wordclouds saved in {out_dir}/")



def train_ml_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Linear SVM": LinearSVC()
    }
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        results[name] = acc
    return results


def train_neural_network(X_train, X_test, y_train, y_test, n_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Neural Network Accuracy: {acc:.4f}")
    return model, acc



def main():
    parser = argparse.ArgumentParser(description="News Category Classification Script")
    parser.add_argument("--data_path", required=True, help="Path to AG News CSV file")
    parser.add_argument("--text_col", default="description", help="Text column name")
    parser.add_argument("--label_col", default="label", help="Label/category column name")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.data_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise KeyError(f"Dataset must have '{args.text_col}' and '{args.label_col}' columns.")

    df[args.text_col] = preprocess_series(df[args.text_col])

    # Encode labels
    labels = df[args.label_col].astype(str)
    label2id = {l: i for i, l in enumerate(sorted(labels.unique()))}
    id2label = {i: l for l, i in label2id.items()}
    df["label_id"] = labels.map(label2id)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_col], df["label_id"],
        test_size=args.test_size, random_state=args.random_state, stratify=df["label_id"]
    )

    # Vectorize
    print("Vectorizing using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train ML models
    results = train_ml_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

    # Visualize frequent words
    plot_top_words(df, args.text_col, args.label_col)
    generate_wordclouds(df, args.text_col, args.label_col)

    # Neural network (optional)
    print("\nTraining simple feedforward neural network...")
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    model, acc = train_neural_network(X_train_tfidf.toarray(), X_test_tfidf.toarray(), y_train_cat, y_test_cat, n_classes=len(label2id))

    print("\nSummary:")
    for name, val in results.items():
        print(f"{name}: {val:.4f}")
    print(f"Neural Network: {acc:.4f}")


if __name__ == "__main__":
    main()
