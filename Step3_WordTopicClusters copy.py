from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

def cluster_words(tokens, num_clusters=25, max_iter=300, n_init=10):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tokens)
    model = KMeans(n_clusters=num_clusters, max_iter=max_iter, n_init=n_init, random_state=42)
    model.fit(X)
    
    clusters = {}
    for i, label in enumerate(model.labels_):
        cluster = clusters.get(int(label), [])
        cluster.append(tokens[i])
        clusters[int(label)] = cluster
    
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
    
    print("Tokenization complete. Filtering low frequency words...")
    word_counter = Counter(all_lines)
    high_freq_words = {word for word, count in word_counter.items() if count >= 3000}
    filtered_words = [word for word in all_lines if word_counter[word] >= 10 and word not in high_freq_words]
    print(f"Words after filtering: {len(filtered_words)}")
    print(f"High frequency words: {len(high_freq_words)}")
    
    # 打印高频词汇
    print("High frequency words:")
    for word in high_freq_words:
        print(word)

    # 保存高频词汇到文件
    with open('high_freq_words.txt', 'w', encoding='utf-8') as file:
        for word in high_freq_words:
            file.write(word + '\n')
    
    print("Starting batch processing for clustering high frequency words...")
    high_freq_clusters = cluster_words(list(high_freq_words), num_clusters=15, max_iter=1000, n_init=50)

    print("Starting batch processing for clustering remaining words...")
    batch_size = 5000
    clusters = {}
    for batch_number, batch in enumerate(batch_process(filtered_words, batch_size), start=1):
        batch_clusters = cluster_words(batch)
        for cluster_id, words in batch_clusters.items():
            if cluster_id in clusters:
                clusters[cluster_id].extend(words)
            else:
                clusters[cluster_id] = words
        
        print(f"Processed batch {batch_number}, total clusters: {len(clusters)}")
    
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
    
    print("Clustering complete. Printing cluster sizes for high frequency words...")
    for cluster_id, words in high_freq_clusters.items():
        print(f"High frequency cluster {cluster_id}: {len(words)} words")
    
    print("Clustering complete. Printing cluster sizes for remaining words...")
    for cluster_id, words in unique_clusters.items():
        print(f"Cluster {cluster_id}: {len(words)} words")
    
    print("Saving results to file...")
    with open('high_freq_word_clusters.json', 'w', encoding='utf-8') as file:
        json.dump(high_freq_clusters, file, ensure_ascii=False, indent=4)
    
    with open('word_clusters.json', 'w', encoding='utf-8') as file:  # 'w' 模式会覆盖文件
        json.dump(unique_clusters, file, ensure_ascii=False, indent=4)
    
    print("Results saved to high_freq_word_clusters.json and word_clusters.json")

if __name__ == '__main__':
    main()
