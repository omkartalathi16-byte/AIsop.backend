from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import logging
import time


class MilvusService:
    def __init__(self, host="127.0.0.1", port="19530", collection_name="sop_collection", dim=384):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self._connect()
        self._set_up_collection()

    def _connect(self):
        retries = 10
        for i in range(retries):
            try:
                connections.connect("default", host=self.host, port=self.port)
                logging.info(f"Connected to Milvus at {self.host}:{self.port}")
                return
            except Exception as e:
                logging.warning(f"Failed to connect to Milvus (attempt {i+1}/{retries}): {e}")
                time.sleep(5)
        raise Exception(f"Could not connect to Milvus after {retries} attempts")

    def _set_up_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logging.info(f"Using existing collection: {self.collection_name}")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description="SOP storage with link")
            self.collection = Collection(self.collection_name, schema)

            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logging.info(f"Created new collection and index: {self.collection_name}")

    def insert_sop(self, title: str, chunks: list[str], embeddings: list[list[float]], link: str = ""):
        data = [
            [title] * len(chunks),
            chunks,
            [link] * len(chunks),
            embeddings
        ]
        self.collection.insert(data)
        self.collection.flush()

    def search_sops(self, query_embedding: list[float], top_k: int = 5):
        self.collection.load()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "content", "link"]
        )

        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "title": hit.entity.get("title"),
                    "content": hit.entity.get("content"),
                    "link": hit.entity.get("link"),
                    "score": hit.score
                })
        return formatted_results
