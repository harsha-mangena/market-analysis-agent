from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
import os

def init_vector_store():
    """
    Initializes the Pinecone vector store with the required embedding.
    
    Returns:
        PineconeVectorStore: Configured vector store instance
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
    
    index = pc.Index(index_name)
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    return PineconeVectorStore(index=index, embedding=embeddings)