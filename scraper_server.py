from fastapi import FastAPI, Query, HTTPException
from scraper import NewsScraper  # Adjust import as necessary

app = FastAPI()
scraper = NewsScraper()


@app.get("/status")
async def status():
    """
    Status endpoint to check if the service is running.
    """
    return {"status": "Service is up and running"}


@app.get("/search")
async def search(
    term: str = Query("latest", description="Search term for news articles"),
):
    """
    Search endpoint to scrape news articles based on a search term.
    Expects a query parameter 'term' for the search term.
    """
    if not term:
        raise HTTPException(status_code=400, detail="Missing 'term' query parameter")

    results_json, runtime = scraper.run(term)
    return {"results": results_json, "runtime": runtime}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
