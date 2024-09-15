from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from db_helper import MilvusHelper  # Import your MilvusHelper class here
import uvicorn

app = FastAPI()
milvus_helper = MilvusHelper()

class Article(BaseModel):
    url: str
    content: str

class InsertData(BaseModel):
    search_term: str
    articles: List[Article]

class SearchQuery(BaseModel):
    search_term: str
    top_k: int = Query(5, gt=0)
    threshold: float = Query(0.5, ge=0.0, le=1.0)

@app.on_event("startup")
async def startup_event():
    milvus_helper.setup_database()

@app.on_event("shutdown")
async def shutdown_event():
    milvus_helper.close()

@app.post("/insert/")
async def insert_data(data: InsertData):
    try:
        milvus_helper.insert_data(data.dict())
        return {"message": "Data inserted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/")
async def search_data(query: SearchQuery):
    try:
        results = milvus_helper.search(query.search_term, top_k=query.top_k, threshold=query.threshold)
        return {"results": [result.to_dict() for result in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)