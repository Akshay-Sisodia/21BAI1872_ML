import asyncio
import aiohttp
from bs4 import BeautifulSoup
import csv
import time
import re
import unidecode  # Import the unidecode package to normalize text
from newspaper import Article  # Import the Article class from newspaper3k

class NewsScraperIndia:
    def __init__(self):
        self.sources = {
            "Times of India": {
                "url": "https://timesofindia.indiatimes.com/briefs",
                "article_selector": ".brief_box h2 a",
            },
            "NDTV": {
                "url": "https://www.ndtv.com/latest",
                "article_selector": ".news_Itm-cont h2 a",
            },
            "The Hindu": {
                "url": "https://www.thehindu.com/latest-news/",
                "article_selector": ".title a",
            },
        }
        # Regex to detect Hindi text (Devanagari script)
        self.hindi_regex = re.compile(r'[\u0900-\u097F]')

    async def fetch(self, session, url):
        async with session.get(url) as response:
            return await response.text()

    def clean_content(self, content):
        """Format content by removing irregular characters, extra spaces, newlines, and truncating if necessary."""
        # Remove irregular characters using unidecode to normalize text
        content = unidecode.unidecode(content)
        
        # Remove extra spaces and newlines
        content = re.sub(r'\s+', ' ', content.strip())

        # Truncate content to 500 characters
        return content

    def is_hindi(self, text):
        """Check if the text contains Hindi (Devanagari script) characters."""
        return bool(self.hindi_regex.search(text))

    def extract_with_newspaper(self, url):
        """Extract content from the article using the newspaper3k library."""
        article = Article(url)
        article.download()
        article.parse()
        return self.clean_content(article.text)

    async def scrape_article(self, session, source_name, article_url):
        try:
            if source_name == "Times of India":
                article_url = "https://timesofindia.indiatimes.com/" + article_url

            # Extract and clean content using newspaper3k
            content = self.extract_with_newspaper(article_url)

            # Skip if content is in Hindi
            if self.is_hindi(content):
                print(f"Skipping Hindi article: {article_url}")
                return None

            return {
                "source": source_name,
                "url": article_url,
                "content": content,  # Truncated and cleaned content
            }
        except Exception as e:
            print(f"Error scraping article {article_url}: {str(e)}")
            return None

    async def scrape_source(self, session, name, info):
        try:
            html = await self.fetch(session, info["url"])
            soup = BeautifulSoup(html, "html.parser")
            articles = soup.select(info["article_selector"])

            tasks = []
            for article in articles:  # Limit to 20 articles per source
                article_url = article["href"]
                if not article_url.startswith("http"):
                    article_url = (
                        "https:" + article_url
                        if article_url.startswith("//")
                        else "https://" + re.sub(r"^/+", "", article_url)
                    )

                # Skip if the article title is in Hindi
                if self.is_hindi(article.text):
                    print(f"Skipping Hindi article: {article_url}")
                    continue

                task = self.scrape_article(session, name, article_url)
                tasks.append(task)

            return await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error scraping {name}: {str(e)}")
            return []

    async def scrape_all_sources(self):
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.scrape_source(session, name, info)
                for name, info in self.sources.items()
            ]
            results = await asyncio.gather(*tasks)
            return [item for sublist in results for item in sublist if item]

    def save_to_csv(self, data, filename="indian_news.csv"):
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["source", "url", "content"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in data:
                writer.writerow(article)

    def run(self):
        start_time = time.time()
        results = asyncio.run(self.scrape_all_sources())
        end_time = time.time()

        self.save_to_csv(results)

        print(f"Scraped {len(results)} articles.")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        print("Results saved to indian_news.csv")


if __name__ == "__main__":
    scraper = NewsScraperIndia()
    scraper.run()
