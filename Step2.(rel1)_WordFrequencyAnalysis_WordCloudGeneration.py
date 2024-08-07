import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

def generate_wordcloud(text):
    font_path = os.path.join(os.path.dirname(__file__), 'arial.ttf')  # 确保字体文件路径正确
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def main():
    print("Starting word frequency analysis...")
    word_freq = Counter()
    
    with open('stackoverflow_questions.txt', 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            tokens = clean_and_tokenize(line)
            word_freq.update(tokens)
            
            # 打印提示信息
            if line_number % 1000 == 0:
                print(f"Processed {line_number} lines.")
    
    print("Word frequency analysis complete. Generating word cloud...")
    # 生成词云
    generate_wordcloud(' '.join(word_freq.elements()))
    print("Word cloud generated and displayed.")

if __name__ == '__main__':
    main()
