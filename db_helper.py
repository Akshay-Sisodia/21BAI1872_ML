import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusHelper:
    def __init__(self, host="localhost", port="19530", collection_name="news_articles"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info("Connected to Milvus server.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="search_term", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(fields, description="News articles collection")
        
        try:
            self.collection = Collection(self.collection_name, schema)
            logger.info(f"Collection '{self.collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def create_index(self):
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
        try:
            self.collection.create_index("embedding", index_params)
            logger.info("Index created successfully.")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    import json  # Add this import at the top

    def insert_data(self, data):
        try:
            if isinstance(data, str):
                data = json.loads(data)

            entities = []
            search_term = data.get('search_term')
            articles = data.get('articles')

            if search_term and isinstance(articles, list):
                for article in articles:
                    url = article.get('url')
                    content = article.get('content')
                    if content:
                        embedding = self.model.encode(content).tolist()
                        entities.append([search_term, url, content, embedding])
                    else:
                        logger.warning(f"Article with URL {url} has no content.")
            else:
                raise ValueError("Unsupported data format. Please provide a valid 'search_term' and a list of 'articles'.")

            try:
                insert_data = [
                    [e[0] for e in entities],  # search_term
                    [e[1] for e in entities],  # url
                    [e[2] for e in entities],  # content
                    [e[3] for e in entities]   # embedding
                ]
                self.collection.insert(insert_data)
                self.collection.flush()  # Ensure data is flushed to storage
                logger.info(f"Inserted {len(entities)} articles with search term '{search_term}'.")
            except Exception as e:
                logger.error(f"Failed to insert data: {e}")
                raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse data as JSON: {e}")
            raise




    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query).tolist()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["search_term", "url", "content"]
            )
            logger.info(f"Search completed. Found {len(results[0])} results.")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def setup_database(self):
        self.connect()
        if not utility.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' does not exist. Creating...")
            self.create_collection()
            self.create_index()
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")
            self.collection = Collection(self.collection_name)
        
        self.collection.load()
        logger.info("Database setup completed.")


    def close(self):
        if self.collection:
            self.collection.release()
        connections.disconnect("default")
        logger.info("Milvus connection closed.")