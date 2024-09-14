import requests
from bs4 import BeautifulSoup
import time
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib
import textwrap
from unidecode import unidecode
import json

class NewsScraperIndia:
    def extract_article_content(self, url):
        """
        Extract article content from a given URL using the newspaper3k library.
        
        Args:
            url (str): URL of the news article.
        
        Returns:
            dict: A dictionary containing the URL and the extracted content.
        """
        try:
            # Initialize the Article object
            article = Article(url)
            article.download()
            article.parse()
            content = article.text
            
            # Use unidecode to handle character encoding issues
            decoded_content = unidecode(content)
            
            # Remove any line breaks for single-line content
            single_line_content = " ".join(decoded_content.split())
            
            return {"url": url, "content": single_line_content}
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return {"url": url, "content": None}

    def scrape_term(self, term):
        """
        Scrape Google News search results for a given search term.
        
        Args:
            term (str): The search term for Google News.
        
        Returns:
            list: A list of URLs extracted from the search results.
        """
        url = f"https://www.google.com/search?q={term}&tbm=nws"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Select article links from the search results
        links = soup.select('a[data-ved^="2ahUKE"]')
        links_ex = [
            link.get("href").removeprefix("/url?q=").split("&", 1)[0] for link in links
        ]
        print(links_ex)
        return links_ex

    def scrape(self, term):
        """
        Perform the scraping of news articles based on the search term.
        
        Args:
            term (str): The search term for scraping news articles.
        
        Returns:
            list: A list of dictionaries containing the URL and content of each article.
        """
        term = urllib.parse.quote(term)
        links = self.scrape_term(term)
        data = []

        # Use ThreadPoolExecutor to parallelize the content extraction
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.extract_article_content, link) for link in links
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    data.append(result)

        return data

    def run(self, term="latest"):
        """
        Run the news scraper with the specified search term.
        
        Args:
            term (str): The search term for scraping news articles.
        
        Returns:
            tuple: A tuple containing the results as a JSON object and the runtime as a string.
        """
        start_time = time.time()

        results = self.scrape(term)
        
        # Convert results to JSON format
        results_json = json.dumps(results, ensure_ascii=False, indent=4)

        end_time = time.time()
        
        runtime = f"{end_time - start_time:.2f} seconds"
        print(f"Scraped {len(results)} articles.")
        print(f"Total time taken: {runtime}")
        
        return results_json, runtime