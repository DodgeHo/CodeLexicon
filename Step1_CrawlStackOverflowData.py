import requests
from bs4 import BeautifulSoup
import time

def fetch_stackoverflow_data(page_num):
    url = f'https://stackoverflow.com/questions?sort=votes&page={page_num}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    questions = soup.find_all('a', class_='question-hyperlink')
    
    question_texts = []
    for question in questions:
        question_texts.append(question.get_text())
    
    return question_texts

def main():
    with open('stackoverflow_questions.txt', 'w', encoding='utf-8') as file:
        for i in range(1, 20001):
            questions = fetch_stackoverflow_data(i)
            for question in questions:
                file.write(question + '\n')
            
            # 打印提示信息
            print(f"Page {i} fetched, {len(questions)} questions added. Total pages processed: {i}")
            
            time.sleep(1)  # 避免过于频繁的请求
    
    print("Data fetching complete. All questions saved to stackoverflow_questions.txt")

if __name__ == '__main__':
    main()
