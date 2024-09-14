import requests
from bs4 import BeautifulSoup
import csv
import time
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib
import textwrap
from unidecode import unidecode


class NewsScraperIndia:
    def extract_article_content(self, url):
        """Extract article content from a given URL."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            content = article.text
            # Use unidecode to handle character encoding issues
            decoded_content = unidecode(content)
            # Format content with line breaks
            formatted_content = "\n".join(textwrap.wrap(decoded_content, width=500))
            return {"url": url, "content": formatted_content}
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return {"url": url, "content": None}

    def scrape_term(self, term):
        url = f"https://www.google.com/search?q={term}&tbm=nws"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.select('a[data-ved^="2ahUKE"]')
        links_ex = [
            link.get("href").removeprefix("/url?q=").split("&", 1)[0] for link in links
        ]
        print(links_ex)
        return links_ex

    def scrape(self, term):
        term = urllib.parse.quote(term)
        links = self.scrape_term(term)
        data = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.extract_article_content, link) for link in links
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    data.append(result)

        return data

    def save_to_csv(self, data, filename="indian_news.csv"):
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["url", "content"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in data:
                if article and article["content"]:  # Skip None values and empty content
                    writer.writerow(article)

    def run(self, term="latest"):
        start_time = time.time()

        results = self.scrape(term)

        end_time = time.time()

        self.save_to_csv(results)

        print(f"Scraped {len(results)} articles.")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        print("Results saved to indian_news.csv")


if __name__ == "__main__":
    scraper = NewsScraperIndia()
    scraper.run("donald trump")
