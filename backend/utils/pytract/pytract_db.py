import redis
import json
import heapq
import nltk
import requests
from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.dataclasses.byte_stream import ByteStream
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter, RecursiveDocumentSplitter
from scipy.spatial.distance import cosine

class pytract_db:
    
    def __init__(self, host='34.31.232.10', port=6379, chunking_strategy='sentence-5'):
        nltk.download('punkt')
        self.cs=chunking_strategy
        self.host = host
        self.port =port
        self.db = {'sentence-5':0, 'word-400-overlap-40':1, 'char-1200-overlap-120':2}.get(self.cs)
        self.redis_client = redis.StrictRedis(host=host, port=port, db=self.db, decode_responses=True)

    def _cosine_similarity(self, vec1, vec2):
        return 1 - cosine(vec1, vec2)

    def _get_embeddings(self, key):
        return json.loads(self.redis_client.get(key)) | {}
    
    def _set_embeddings(self, key, embeddings):
        self.redis_client.set(key, json.dumps(embeddings))
        
    def run_indexing_pipeline(self, url):
        markdown_bytes = requests.get(url).content
        markdown_stream = ByteStream(data=markdown_bytes, mime_type="text/markdown")
        converter = MarkdownToDocument()
        cleaner = DocumentCleaner()
        if self.cs=='sentence-5':
            splitter=DocumentSplitter(split_by="sentence", split_length=5)
        elif self.cs=='word-400-overlap-40':
            splitter=RecursiveDocumentSplitter(
                split_length=400,
                split_overlap=40,
                split_unit="word",
                separators=["\n\n", "\n", "sentence", " "])
        else:
            splitter=RecursiveDocumentSplitter(
                    split_length=1200,
                    split_overlap=120,
                    split_unit="char",
                    separators=["\n\n", "\n", "sentence", " "])
        embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small", dimensions=1536)
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("converter", converter)
        indexing_pipeline.add_component("cleaner", cleaner)
        indexing_pipeline.add_component("splitter", splitter)
        indexing_pipeline.add_component("embedder", embedder)
        indexing_pipeline.connect("converter.documents", "cleaner.documents")
        indexing_pipeline.connect("cleaner.documents", "splitter.documents")
        indexing_pipeline.connect("splitter.documents", "embedder.documents")
        data=indexing_pipeline.run(data={"sources": [markdown_stream]})
        embeddings = {doc.content : doc.embedding for  index, doc in enumerate(data['embedder']['documents'])}
        redis_key= f"{url}:{self.cs}"
        self._set_embeddings(redis_key, embeddings)
    
    def get_top_n_chunks(self, key, query, n=5):
        query_embedding = OpenAITextEmbedder(model="text-embedding-3-small", dimensions=1536).run(query)['embedding']
        embeddings = self._get_embeddings(key)
        unsorted_embeddings = { k: self._cosine_similarity(v, query_embedding) for k,v in embeddings.items()}
        top_n = heapq.nlargest(n, unsorted_embeddings.items(), key=lambda x: x[1])
        return ", ".join(pair[0] for pair in top_n)