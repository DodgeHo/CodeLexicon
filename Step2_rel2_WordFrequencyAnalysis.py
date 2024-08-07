import nltk
from collections import Counter
import string

# 下载所需的 NLTK 数据包
nltk.download('punkt')

def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return tokens

def main():
    print("Starting word frequency analysis...")
    word_freq = Counter()
    
    with open('stackoverflow_questions.txt', 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            tokens = clean_and_tokenize(line)
            word_freq.update(tokens)
            
            # 打印提示信息
            if line_number % 100000 == 0:
                print(f"Processed {line_number} lines.")
    
    print("Word frequency analysis complete. Saving results to file...")
    
    # 将高频词保存到文件
    with open('word_frequencies_l.txt', 'w', encoding='utf-8') as f:
        for word, freq in word_freq.items():
            if freq > 20:
                f.write(f"{word} \n")
    
    print("Word frequencies saved to word_frequencies.txt")

if __name__ == '__main__':
    main()
