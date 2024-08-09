import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

# Step 1: Load the Data and Preprocess
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

def load_data(file_path):
    text_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            text_data.append(preprocess_text(line.strip()))
            if (i + 1) % 100000 == 0:
                print(f'{i + 1} lines processed')
    print(f'Total {i + 1} lines processed')
    return text_data

# Step 2: Calculate Word Frequencies
def calculate_word_frequencies(text_data):
    all_words = ' '.join(text_data).split()
    word_freq = Counter(all_words)
    # Filter words with frequency >= 10
    word_freq = {word: freq for word, freq in word_freq.items() if freq >= 10}
    return word_freq

# Step 3: Build Co-occurrence Matrix
def build_cooccurrence_matrix(text_data, word_freq):
    vectorizer = CountVectorizer(vocabulary=word_freq.keys(), binary=True)
    X = vectorizer.fit_transform(text_data)
    Xc = (X.T * X)  # Co-occurrence matrix
    Xc.setdiag(0)  # Set diagonal to zero (a word co-occurring with itself is not meaningful)
    return Xc.toarray(), vectorizer.get_feature_names_out()

# Step 4: Perform Clustering
def perform_clustering(cooccurrence_matrix, n_clusters):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    # Print progress during clustering
    step = max(1, cooccurrence_matrix.shape[0] // 10)
    for i in range(0, cooccurrence_matrix.shape[0], step):
        clustering_model.fit(cooccurrence_matrix[:i + step])
        print(f'Clustering progress: {min(i + step, cooccurrence_matrix.shape[0])} words clustered')
    clustering_model.fit(cooccurrence_matrix)  # Final fit for the entire matrix
    return clustering_model.labels_

# Step 5: Save the Results
def save_results(word_freq, clustering_labels, feature_names):
    # Save word frequencies
    with open('word_frequencies.txt', 'w', encoding='utf-8') as f:
        for word, freq in word_freq.items():
            f.write(f'{word}: {freq}\n')

    # Save clustered themes
    clusters = {}
    for word, label in zip(feature_names, clustering_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(word)

    with open('word_clusters.txt', 'w', encoding='utf-8') as f:
        for label, words in clusters.items():
            f.write(f'Cluster {label}:\n')
            f.write(', '.join(words) + '\n\n')

def main():
    # Load and preprocess the data
    file_path = 'stackoverflow_questions.txt'
    text_data = load_data(file_path)

    # Calculate word frequencies
    word_freq = calculate_word_frequencies(text_data)

    # Build the co-occurrence matrix
    cooccurrence_matrix, feature_names = build_cooccurrence_matrix(text_data, word_freq)

    # Perform clustering into 20-25 themes
    n_clusters = 25
    clustering_labels = perform_clustering(cooccurrence_matrix, n_clusters)

    # Save the results
    save_results(word_freq, clustering_labels, feature_names)

    print("Word frequency and clustering completed. Results saved to 'word_frequencies.txt' and 'word_clusters.txt'.")

if __name__ == '__main__':
    main()
