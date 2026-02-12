from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import logging

class MilvusService:
    def __init__(self, host="localhost", port="19530", collection_name="sop_collection", dim=384):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self._connect()
        self._set_up_collection()

    def _connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            logging.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            raise e

    def _set_up_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logging.info(f"Using existing collection: {self.collection_name}")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description="SOP storage")
            self.collection = Collection(self.collection_name, schema)
            
            # Create index for search
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logging.info(f"Created new collection and index: {self.collection_name}")

    def insert_sop(self, title: str, content: str, embedding: list[float]):
        data = [
            [title],
            [content],
            [embedding]
        ]
        self.collection.insert(data)
        self.collection.flush()

    def search_sops(self, query_embedding: list[float], top_k: int = 5):
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "content"]
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "title": hit.entity.get("title"),
                    "content": hit.entity.get("content"),
                    "score": hit.score
                })
        return formatted_results
