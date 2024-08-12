import requests
from bs4 import BeautifulSoup
import time
import hashlib

def fetch_stackoverflow_data(page_num):
    url = f'https://stackoverflow.com/questions?sort=votes&page={page_num}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    questions = soup.find_all('a', class_='question-hyperlink')
    
    question_texts = []
    for question in questions:
        question_texts.append(question.get_text())
    
    return question_texts, hashlib.md5(response.text.encode('utf-8')).hexdigest()

def main():
    previous_hash = None
    
    with open('stackoverflow_questions.txt', 'a', encoding='utf-8') as file:
        for i in range(1, 200000):
            try:
                questions, page_hash = fetch_stackoverflow_data(i)
                
                # 判断页面是否重复
                if page_hash == previous_hash:
                    print(f"Page {i} is duplicate, skipping.")
                    continue
                previous_hash = page_hash

                for question in questions:
                    file.write(question + '\n')
                
                print(f"Page {i} fetched, {len(questions)} questions added. Total pages processed: {i}")
                
            except Exception as e:
                print(f"Error fetching page {i}: {e}")
            
            time.sleep(1)  # 避免过于频繁的请求
    
    print("Data fetching complete. All questions saved to stackoverflow_questions.txt")

if __name__ == '__main__':
    main()
