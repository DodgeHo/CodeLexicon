import nltk
from nltk.corpus import stopwords
from collections import Counter
import string

# 下载所需的 NLTK 数据包
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

def main():
    print("Starting word frequency analysis...")
    word_freq = Counter()
    
    with open('stackoverflow_questions.txt', 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            tokens = clean_and_tokenize(line)
            word_freq.update(tokens)
            
            # 打印提示信息
            if line_number % 10000 == 0:
                print(f"Processed {line_number} lines.")
    
    print("Word frequency analysis complete. Saving results to file...")
    
    # 将高频词保存到文件
    with open('word_frequencies.txt', 'w', encoding='utf-8') as f:
        for word, freq in word_freq.most_common():
            f.write(f"{word}: {freq}\n")
    
    print("Word frequencies saved to word_frequencies.txt")

if __name__ == '__main__':
    main()
