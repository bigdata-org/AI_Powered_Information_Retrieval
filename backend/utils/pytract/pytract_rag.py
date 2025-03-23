from haystack import Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.builders import PromptBuilder, ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter, RecursiveDocumentSplitter
from haystack.dataclasses.byte_stream import ByteStream
from haystack.components.writers import DocumentWriter
from utils.litellm.llm import llm
import requests
import nltk

class pytract_rag:
    def __init__(self, mode, db='pinecone',chunking_strategy='sentence-5'): #mode either nvidia or custom
        nltk.download('punkt')
        self.mode = mode
        self.db=db.lower()
        self.cs=chunking_strategy
        self.namespace = {'sentence-5':'nvidia_cs_1', 'word-400-overlap-40':'nvidia_cs_2', 'char-1200-overlap-120':'nvidia_cs_3'}.get(chunking_strategy) \
            if self.mode=='nvidia' else {'sentence-5':'custom_cs_1', 'word-400-overlap-40':'custom_cs_2', 'char-1200-overlap-120':'custom_cs_3'}.get(chunking_strategy)
        self.document_store = PineconeDocumentStore(index="nvidia-vectors", namespace=self.namespace, dimension=1536) \
                        if self.db=='pinecone' else \
                     ChromaDocumentStore(host="34.31.232.10", port="8000", collection_name=self.namespace)
        
    def run_nvidia_text_generation_pipeline(self, search_params, query, model='gpt-4o-mini-2024-07-18'):
        master_documents=[]
        for param in search_params:
            year, qtr = param['year'], param['qtr']
            filters = { "operator": "AND",
                        "conditions": [
                            {"field": "meta.year", "operator": "==", "value": year},
                            {"field": "meta.qtr", "operator": "==", "value": qtr},
                                ]
                        }
            text_embedder = OpenAITextEmbedder(model="text-embedding-3-small", dimensions=1536)
            retriever = PineconeEmbeddingRetriever(document_store=self.document_store, filters=filters) if self.db=='pinecone' \
                else ChromaEmbeddingRetriever(document_store=self.document_store, filters=filters)
            rag_pipeline = Pipeline()
            rag_pipeline.add_component("text_embedder", text_embedder)
            rag_pipeline.add_component("retriever", retriever)
            rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
            result = rag_pipeline.run(data={"text_embedder": {"text": query}})
            master_documents.extend(result['retriever']['documents'])
            prompt_template = [
                    ChatMessage.from_user(
                        """
                        Given the following documents, answer the question in **Markdown format**.
                        
                        ## Documents:
                        {% for doc in documents %}
                        - **Document Year:{{ doc.meta['year'] }}, Quarter: {{ doc.meta['qtr'] }}**  
                        {{ doc.content }}
                        
                        {% endfor %}
                        
                        ## Question:
                        **{{query}}**
                        
                        ---
                        
                        ## Answer:
                        Format the response in the markdown format using:
                        - Headings (`##`, `###`)
                        - Bullet points (`-`, `*`)
                        - Tables (if necessary)
                        - Properly formatted code blocks for technical content (` ``` `)
                        
                        Rememeber the following instructions: 
                        - The above context can have one or more documents belonging to different year and quarters of nvidia 10k report
                        - Each document is indicated as Document Year: <Year>, Quarter: <Quarter>
                        - Analyze all document contents and answer the question
                        """
                    )
                ]
        prompt_builder = ChatPromptBuilder(template=prompt_template)
        prompt = prompt_builder.run(documents=master_documents, query=query)['prompt'][0]._content[0].text
        response = llm(model, prompt)
        return response
    
    def run_custom_text_generation_pipeline(self, url, query, model='gpt-4o-mini-2024-07-18'):
        filters = {"field": "meta.src", "operator": "==", "value": url}
                
        text_embedder = OpenAITextEmbedder()
        retriever = PineconeEmbeddingRetriever(document_store=self.document_store, filters=filters) if self.db=='pinecone' \
                else ChromaEmbeddingRetriever(document_store=self.document_store, filters=filters)        
        prompt_template = [
        ChatMessage.from_user(
            """
            Given the following documents, answer the question in **Markdown format**.
            
            ## Documents:
            {% for doc in documents %}
            - **Document {{ loop.index }}:**  
            {{ doc.content }}
            
            {% endfor %}
            
            ## Question:
            **{{query}}**
            
            ---
            
            ## Answer:
            Format the response in the markdown format using:
            - Headings (`##`, `###`)
            - Bullet points (`-`, `*`)
            - Tables (if necessary)
            - Properly formatted code blocks for technical content (` ``` `)
            """
        )
    ]
        prompt_builder = ChatPromptBuilder(template=prompt_template)
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)

        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        result = rag_pipeline.run(data={"prompt_builder": {"query":query}, "text_embedder": {"text": query}})
        prompt = result['prompt_builder']['prompt'][0]._content[0].text
        response = llm(model,prompt)
        return response 
    
    def run_custom_indexing_pipeline(self, url):
        markdown_bytes = requests.get(url).content
        markdown_stream = ByteStream(data=markdown_bytes, mime_type="text/markdown", meta={"src":url})
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
        embedder = OpenAIDocumentEmbedder( model="text-embedding-3-small",  dimensions=1536)

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("converter", converter)
        indexing_pipeline.add_component("cleaner", cleaner)
        indexing_pipeline.add_component("splitter", splitter)
        indexing_pipeline.add_component("embedder", embedder)

        indexing_pipeline.connect("converter.documents", "cleaner.documents")
        indexing_pipeline.connect("cleaner.documents", "splitter.documents")
        indexing_pipeline.connect("splitter.documents", "embedder.documents")
        
        data=indexing_pipeline.run(data={"sources": [markdown_stream]})
        
        docs=[]
        for doc in data['embedder']['documents']:
            if '_split_overlap' in doc.meta:
                doc.meta.pop('_split_overlap')
            if doc.embedding:
                docs.append(doc)
        self.document_store.write_documents(docs)
    
