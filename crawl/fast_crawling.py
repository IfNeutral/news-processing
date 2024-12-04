import pandas as pd
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
import time

# 사용자 입력
search_content = input("검색할 키워드를 입력해주세요: ")
max_news = int(input("\n몇 개의 뉴스를 크롤링할지 입력해주세요. ex) 1000(숫자만입력): "))

# 크롤링할 기간 설정
startday = ["2024.08.14"]
endday = ["2024.11.29"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

# 비동기 URL 크롤링
async def fetch_urls(session, search_content, start_day, end_day, page, retry=0):
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        async with session.get(
            f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search_content}&start={page}&pd=3&ds={start_day}&de={end_day}",
            headers=headers,
            timeout=10,
        ) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                ul = soup.select_one("div.group_news > ul.list_news")
                if not ul:
                    return []
                urls = [
                    a_tag["href"]
                    for li in ul.find_all("li")
                    for a_tag in li.select('div.news_area > div.news_info > div.info_group > a.info')
                    if "n.news.naver.com" in a_tag["href"]
                ]
                return urls
    except Exception as e:
        if retry < 3:
            print(f"Retrying page {page}, attempt {retry + 1}: {e}")
            await asyncio.sleep(1)
            return await fetch_urls(session, search_content, start_day, end_day, page, retry + 1)
        else:
            print(f"Failed to fetch page {page} after 3 retries: {e}")
    return []

async def crawl_urls(search_content, startday, endday, max_news, batch_size=10):
    url_set = set()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10)) as session:
        for start_day, end_day in zip(startday, endday):
            for page in tqdm(range(1, 2000, batch_size * 10), desc="Fetching URL batches"):
                tasks = [
                    fetch_urls(session, search_content, start_day, end_day, page + i * 10)
                    for i in range(batch_size)
                ]
                results = await asyncio.gather(*tasks)
                for urls in results:
                    url_set.update(urls)
                    if len(url_set) >= max_news:
                        return list(url_set)[:max_news]
                await asyncio.sleep(0.3)  # 서버 부하 방지
            if len(url_set) >= max_news:
                break
    return list(url_set)[:max_news]

# 뉴스 콘텐츠 크롤링
async def fetch_news_content(session, url, retry=0):
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                company = soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_top > a > img[alt]")
                title = soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
                content = soup.select_one("article#dic_area")
                return {
                    "company": company["alt"] if company else "None",
                    "url": url,
                    "title": title.text if title else "None",
                    "content": content.text.strip() if content else "None",
                }
    except Exception as e:
        if retry < 3:
            print(f"Retrying content from {url}, attempt {retry + 1}: {e}")
            await asyncio.sleep(1)
            return await fetch_news_content(session, url, retry + 1)
        else:
            print(f"Failed to fetch content from {url} after 3 retries: {e}")
    return {"company": "None", "url": url, "title": "None", "content": "None"}

async def crawl_news(news_url_list, batch_size=10):
    news_data = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10)) as session:
        for i in tqdm(range(0, len(news_url_list), batch_size), desc="Fetching News Content"):
            batch_urls = news_url_list[i:i + batch_size]
            tasks = [fetch_news_content(session, url) for url in batch_urls]
            results = await asyncio.gather(*tasks)
            news_data.extend(results)
            await asyncio.sleep(0.3)  # 서버 부하 방지
    return news_data

# 실행 함수
async def main():
    # URL 크롤링
    news_url = await crawl_urls(search_content, startday, endday, max_news)
    print(f"총 {len(news_url)}개의 URL을 수집했습니다.")

    # 뉴스 데이터 크롤링
    news_data = await crawl_news(news_url)

    # 데이터프레임 생성 및 저장
    df_news = pd.DataFrame(news_data)
    df_news.to_csv("crawled_news.csv", index=False, encoding="utf-8-sig")


# 비동기 실행
asyncio.run(main())