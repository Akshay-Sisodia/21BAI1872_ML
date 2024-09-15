import time
import logging
import httpx
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize SentenceTransformer model for request similarity checking
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for storing previous requests and responses
request_cache = []
MAX_CACHE_SIZE = 100

# Dictionary to track user request counts
user_request_counts = defaultdict(int)
MAX_REQUESTS_PER_USER = 5

# Client for making synchronous HTTP requests
http_client = httpx.Client(timeout=15)

class SearchQuery(BaseModel):
    search_term: str
    top_k: int = Query(5, gt=0)
    threshold: float = Query(0.5, ge=0.0, le=1.0)
    user_id: str

def start_servers():
    # Start scraper server
    subprocess.Popen(["python", "scraper_server.py"])
    logger.info("Scraper server started")

    # Start db server
    subprocess.Popen(["python", "db_server.py"])
    logger.info("DB server started")

    # Wait for servers to initialize
    time.sleep(5)

@app.on_event("startup")
def startup_event():
    start_servers()

@app.on_event("shutdown")
def shutdown_event():
    http_client.close()

def check_user_limit(user_id: str):
    if user_request_counts[user_id] >= MAX_REQUESTS_PER_USER:
        raise HTTPException(status_code=429, detail="Too many requests")
    user_request_counts[user_id] += 1

def format_context(articles):
    context = ""
    for article in articles:
        context += f"URL: {article['entity']['url']}\n\n"
        context += f"Content: {article['entity']['content']}\n\n"
        context += "---\n\n"
    return context.strip()

def scrape_and_insert(search_term: str):
    # Scrape news articles
    scraper_response = http_client.get(f"http://localhost:8000/search?term={search_term}")
    scraper_data = scraper_response.json()
    
    # Insert scraped data into the database
    insert_response = http_client.post("http://localhost:8001/insert/", json=json.loads(scraper_data['results']))
    if insert_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to insert data into the database")

def search_database(query: SearchQuery):
    # Search the database for relevant articles
    query_payload = {
        "search_term": query.search_term,
        "top_k": query.top_k,
        "threshold": query.threshold,
    }
    search_response = http_client.post("http://localhost:8001/search/", json=query_payload)
    if search_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to search the database")
    return search_response.json()['results']

def check_request_similarity(new_request, threshold=0.95):
    if not request_cache:
        return None
    
    new_embedding = model.encode([new_request], show_progress_bar=False)
    cache_embeddings = model.encode([req['request'] for req in request_cache], show_progress_bar=False)
    
    similarities = cosine_similarity(new_embedding, cache_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    
    if similarities[most_similar_idx] > threshold:
        return request_cache[most_similar_idx]['response']
    return None

def update_cache(request, response):
    request_cache.append({'request': request, 'response': response})
    if len(request_cache) > MAX_CACHE_SIZE:
        request_cache.pop(0)

@app.post("/get_context")
def get_context(query: SearchQuery):
    start_time = time.time()
    
    # Check user request limit
    check_user_limit(query.user_id)
    
    # Check cache for similar requests
    cached_response = check_request_similarity(query.search_term)
    if cached_response:
        logger.info(f"Cache hit for query: {query.search_term}")
        return {"context": cached_response, "source": "cache"}
    
    # Scrape and insert new data
    scrape_and_insert(query.search_term)
    
    # Search the database
    search_results = search_database(query)
    
    # Format the context
    context = format_context(search_results)
    
    # Update cache
    update_cache(query.search_term, context)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    logger.info(f"Request processed. User: {query.user_id}, Query: {query.search_term}, Inference time: {inference_time:.2f}s")
    
    return {"context": context, "source": "database", "inference_time": inference_time}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)