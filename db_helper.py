import cudf
from cuml.neighbors import NearestNeighbors
import cupy as cp
import numpy as np
from cuml.metrics.pairwise_distances import pairwise_distances
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from sentence_transformers import SentenceTransformer
import logging
import json
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSimilarityRanker:
    def __init__(self, metric='cosine'):
        self.metric = metric

    def rank(self, embeddings):
        # Ensure we're working with a numpy array
        if isinstance(embeddings, cudf.DataFrame):
            embeddings = embeddings.to_numpy()
        elif isinstance(embeddings, cp.ndarray):
            embeddings = cp.asnumpy(embeddings)
        
        # Compute similarity matrix using cuML's pairwise_distances
        similarity_matrix = pairwise_distances(embeddings, metric=self.metric)
        
        # Compute ranking scores (lower distance means higher similarity)
        ranking_scores = np.sum(similarity_matrix, axis=1)
        
        # Sort embeddings based on ranking scores (ascending order for distances)
        sorted_indices = np.argsort(ranking_scores)
        
        return sorted_indices

class MilvusHelper:
    def __init__(self, host="localhost", port="19530", collection_name="news_articles"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.ranker = VectorSimilarityRanker()
    
class MilvusHelper:
    def __init__(self, host="localhost", port="19530", collection_name="news_articles"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.ranker = VectorSimilarityRanker()

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
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
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
            "params": {"nlist": 1024},
        }
        try:
            self.collection.create_index("embedding", index_params)
            logger.info("Index created successfully.")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def insert_data(self, data):
        try:
            if isinstance(data, str):
                data = json.loads(data)

            entities = []
            search_term = data.get("search_term")
            articles = data.get("articles")

            if search_term and isinstance(articles, list):
                embeddings = []
                for article in articles:
                    url = article.get("url")
                    content = article.get("content")
                    if content:
                        embedding = self.model.encode(
                            content, show_progress_bar=False
                        ).tolist()
                        embeddings.append(embedding)
                        entities.append([search_term, url, content, embedding])
                    else:
                        logger.warning(f"Article with URL {url} has no content.")

                # Rank embeddings
                ranked_indices = self.ranker.rank(np.array(embeddings))
                ranked_entities = [entities[i] for i in ranked_indices]

                # Prepare data for insertion
                insert_data = [
                    [e[0] for e in ranked_entities],  # search_term
                    [e[1] for e in ranked_entities],  # url
                    [e[2] for e in ranked_entities],  # content
                    [e[3] for e in ranked_entities],  # embedding
                ]
                
                self.collection.insert(insert_data)
                self.collection.flush()  # Ensure data is flushed to storage
                logger.info(
                    f"Inserted {len(ranked_entities)} ranked articles with search term '{search_term}'."
                )
            else:
                raise ValueError(
                    "Unsupported data format. Please provide a valid 'search_term' and a list of 'articles'."
                )
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            raise

    def search(self, query, top_k=5, threshold=0.5):
        query_embedding = self.model.encode(query, show_progress_bar=False).tolist()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 2,  # Retrieve more results to allow for filtering
                output_fields=["search_term", "url", "content"],
            )

            # Normalize distances to [0, 1] range
            distances = np.array([hit.distance for hit in results[0]])
            max_distance = np.max(distances)
            min_distance = np.min(distances)
            normalized_distances = (distances - min_distance) / (max_distance - min_distance)

            # Filter results based on the normalized threshold
            filtered_results = []
            for i, result in enumerate(results[0]):
                if normalized_distances[i] <= threshold:
                    filtered_results.append(result)
                if len(filtered_results) == top_k:
                    break

            logger.info(f"Search completed. Found {len(filtered_results)} results within the threshold.")
            return filtered_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def setup_database(self):
        self.connect()
        if not utility.has_collection(self.collection_name):
            logger.info(
                f"Collection '{self.collection_name}' does not exist. Creating..."
            )
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