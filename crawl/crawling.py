import pandas as pd
import numpy as np
import time
import re

from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import requests

search_content = input("검색할 키워드를 입력해주세요: ")
max_news = int(input("\n몇 개의 뉴스를 크롤링할지 입력해주세요. ex) 1000(숫자만입력): "))

# 크롤링할 기간 설정
startday = ["2024.08.14"]
endday = ["2024.11.29"]

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}


def url_crawling(search_content, start_season, end_season, max_news):
    # 집합 형태로 저장해 중복 url 제거
    url_set = set()
    for start_day, end_day in zip(start_season, end_season):
        for page in tqdm(range(1, 2000, 10)):
            response = requests.get(
                f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search_content}&start={page}&pd=3&ds={start_day}&de={end_day}",
                headers=headers)
            # page를 넘기다 page가 없으면 종료
            # 200은 HTTP 상태코드중 하나로 OK의 의미를 가짐. 요청이 성공적으로 처리되었음을 나타냄. 200이 아니라는것은 페이지가 없어 페이지를 넘길 수 없다는 의미
            if response.status_code != 200:
                print(f"페이지 {page // 10}가 없습니다. Exiting.")
                break
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            ul = soup.select_one("div.group_news > ul.list_news")

            if ul is None:
                break
            li_list = ul.find_all('li')
            for li in li_list:
                a_list = li.select('div.news_area > div.news_info > div.info_group > a.info') # div.news_contents > a.news_tit
                for a_tag in a_list:
                    href = a_tag.get('href')
                    # href 속성값이 "n.news.naver.com"(네이버 뉴스)을 포함하는지 확인한다.
                    if "n.news.naver.com" in href:
                        try:
                            # request.head()로 추출한 url이 rediret되는지 확인한다. redirect 되지않은 url만 저장한다.
                            response = requests.head(href, allow_redirects=True)
                            if response.status_code == 200:
                                url_set.add(href)
                                # 원하는 개수의 기사가 모두 크롤링 되었으면 크롤링 종료
                                if len(url_set) >= max_news:
                                    return list(url_set)
                        except Exception as e:
                            print(f"An error occurred: {e}")
            time.sleep(0.4)

    return list(url_set)


url = url_crawling(search_content, startday, endday, max_news)
print(len(url))

news_url = url

# 신문사, 제목, 본문 추출
news_company = []
news_title = []
news_content = []

for url in tqdm(news_url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    company = soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_top > a > img[alt]")
    news_company.append(company['alt'] if company else 'None')
    title = soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
    news_title.append(title.text if title else 'None')
    content = soup.select_one("article#dic_area")
    news_content.append(content.text if content else 'None')

# 데이터프레임 생성
columns = ["company", "url", "title", "content"]

data = {
    "company": news_company,
    "url": news_url,
    "title": news_title,
    "content": news_content
}

df_news = pd.DataFrame(data, columns=columns)
print(df_news.head())
df_news.to_csv("crawled_news.csv", index=False, encoding='utf-8-sig')
