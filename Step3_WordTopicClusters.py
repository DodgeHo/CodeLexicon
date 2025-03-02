from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
import nltk
import json
import string  # 导入 string 模块
import subprocess
from collections import Counter

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def count_lines_with_wc(filename):
    result = subprocess.run(['wc', '-l', filename], stdout=subprocess.PIPE, text=True)
    line_count = int(result.stdout.split()[0])
    return line_count

def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

def batch_process(lines, batch_size):
    for i in range(0, len(lines), batch_size):
        yield lines[i:i + batch_size]

def dbscan_cluster_words(tokens, eps=0.5, min_samples=5):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tokens)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = model.fit_predict(X)
    
    clusters = {}
    for i, label in enumerate(labels):
        cluster = clusters.get(str(label), [])  # 将键转换为字符串类型
        cluster.append(tokens[i])
        clusters[str(label)] = cluster
    
    return clusters

def main():
    print("Counting lines in file...")
    total_lines = count_lines_with_wc('stackoverflow_questions.txt')
    print(f"Total lines in file: {total_lines}")
    
    print("Starting clustering of words...")
    
    all_lines = []
    with open('stackoverflow_questions.txt', 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            tokens = clean_and_tokenize(line)
            all_lines.extend(tokens)
            
            if line_number % 100000 == 0:
                print(f"Processed {line_number} of {total_lines} lines ({(line_number / total_lines) * 100:.2f}%)")
    
    print("Tokenization complete. Filtering low frequency words...")
    word_counter = Counter(all_lines)
    filtered_words = [word for word in all_lines if word_counter[word] >= 10]
    print(f"Words after filtering: {len(filtered_words)}")
    
    print("Starting DBSCAN clustering for all words...")
    clusters = dbscan_cluster_words(filtered_words, eps=0.3, min_samples=5)
    
    print("Removing duplicate words from clusters...")
    unique_clusters = {}
    seen_words = set()
    for cluster_id, words in clusters.items():
        unique_words = []
        for word in words:
            if word not in seen_words:
                unique_words.append(word)
                seen_words.add(word)
        unique_clusters[cluster_id] = unique_words
    
    print("Clustering complete. Printing cluster sizes...")
    for cluster_id, words in unique_clusters.items():
        print(f"Cluster {cluster_id}: {len(words)} words")
    
    print("Saving results to file...")
    with open('word_clusters.json', 'w', encoding='utf-8') as file:
        json.dump(unique_clusters, file, ensure_ascii=False, indent=4)
    
    print("Results saved to word_clusters.json")

if __name__ == '__main__':
    main()
