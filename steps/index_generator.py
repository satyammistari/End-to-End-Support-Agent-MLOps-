import os
from typing import List

from langchain_core.vectorstores import VectorStore 
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from materializers.faiss_materializer import FAISSMaterializer
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step

@step(output_materializers={"vector_store": FAISSMaterializer})
def index_generator(
    documents: List[Document],
) -> Annotated[VectorStore, "vector_store"]:
    """Enterprise step to generate a searchable FAISS index using local models."""
    
    # 1. Initialize Local Embeddings (Free, runs on your CPU)
    # This replaces the need for an OpenAI API key and quota
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Split documents into smaller chunks for better AI retrieval
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    compiled_texts = text_splitter.split_documents(documents)

    # 3. Log Metadata (Enterprise best practice for tracking)
    log_artifact_metadata(
        artifact_name="vector_store",
        metadata={
            "embedding_type": "HuggingFace (all-MiniLM-L6-v2)",
            "vector_store_type": "FAISS",
            "chunk_size": 1000,
            "document_count": len(documents)
        },
    )

    # 4. Create and return the Vector Store
    return FAISS.from_documents(compiled_texts, embeddings)