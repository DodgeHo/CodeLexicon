from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import nltk
import json
import string  # 导入 string 模块
import subprocess

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

def cluster_words(tokens, num_clusters=10):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tokens)
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X)
    
    clusters = {}
    for i, label in enumerate(model.labels_):
        cluster = clusters.get(int(label), [])  # 将键转换为原生 Python 的 int 类型
        cluster.append(tokens[i])
        clusters[int(label)] = cluster  # 将键转换为原生 Python 的 int 类型
    
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
            
            # 打印提示信息
            if line_number % 100000 == 0:
                print(f"Processed {line_number} of {total_lines} lines ({(line_number / total_lines) * 100:.2f}%)")
    
    print("Tokenization complete. Starting batch processing for clustering...")
    batch_size = 5000
    clusters = {}
    for batch_number, batch in enumerate(batch_process(all_lines, batch_size), start=1):
        batch_clusters = cluster_words(batch)
        for cluster_id, words in batch_clusters.items():
            if cluster_id in clusters:
                clusters[cluster_id].extend(words)
            else:
                clusters[cluster_id] = words
        
        print(f"Processed batch {batch_number}, total clusters: {len(clusters)}")
    
    print("Clustering complete. Saving results to file...")
    with open('word_clusters.json', 'w', encoding='utf-8') as file:
        json.dump(clusters, file, ensure_ascii=False, indent=4)
    
    print("Results saved to word_clusters.json")

if __name__ == '__main__':
    main()