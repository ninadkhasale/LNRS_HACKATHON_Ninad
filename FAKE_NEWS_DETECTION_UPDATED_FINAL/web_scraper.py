import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)

def search_bbc_via_google(query):
    """Searches for BBC articles using Google."""
    url = f"https://www.google.com/search?q=site:bbc.com+{query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []

        for result in soup.select('div.g'):
            title_tag = result.find('h3')
            if title_tag:
                title = title_tag.get_text()
                link = result.find('a')['href']
                articles.append({"title": title, "url": link})

        logging.info(f"Found {len(articles)} articles from Google search.")
        return articles
    else:
        logging.error(f"Error fetching search results: {response.status_code}")
        return []

def scrape_fact_checks():
    """Scrapes recent fact-check articles from PolitiFact."""
    url = "https://www.politifact.com/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        fact_checks = []

        for article in soup.select('section.o-listicle__item'):
            title = article.find('a').get_text(strip=True)
            link = f"https://www.politifact.com{article.find('a')['href']}"
            fact_checks.append({"title": title, "url": link})

        logging.info(f"Found {len(fact_checks)} fact-checks from PolitiFact.")
        return fact_checks
    else:
        logging.error(f"Error scraping PolitiFact: {response.status_code}")
        return []
