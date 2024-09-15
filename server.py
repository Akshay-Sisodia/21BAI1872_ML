import time
import logging
import httpx
import json
import numpy as np
import redis
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize SentenceTransformer model for request similarity checking
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Redis client
MAX_CACHE_SIZE = 100
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

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
    cached_response = redis_client.get(new_request)
    if cached_response:
        return cached_response
    
    if not redis_client.exists("embeddings"):
        return None

    new_embedding = model.encode([new_request], show_progress_bar=False)
    cache_embeddings = json.loads(redis_client.get("embeddings"))

    cache_embeddings = np.array(cache_embeddings)
    similarities = cosine_similarity(new_embedding, cache_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    
    if similarities[most_similar_idx] > threshold:
        return redis_client.get(f"response:{most_similar_idx}")
    return None

def update_cache(request, response):
    redis_client.set(request, response)
    cache_keys = redis_client.keys("response:*")
    if len(cache_keys) >= MAX_CACHE_SIZE:
        redis_client.delete(cache_keys[0])
    redis_client.set(f"response:{len(cache_keys)}", response)
    redis_client.rpush("requests", request)

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