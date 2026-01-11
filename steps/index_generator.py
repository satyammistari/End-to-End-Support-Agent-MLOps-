import os
from typing import List

from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from materializers.faiss_materializer import FAISSMaterializer
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step
from zenml.client import Client

@step(output_materializers={"vector_store": FAISSMaterializer})
def index_generator(
    documents: List[Document],
) -> Annotated[VectorStore, "vector_store"]:
    """Enterprise step to generate a searchable FAISS index."""
    
    # 1. Attempt to get the OpenAI key from ZenML Secret Store
    # We changed the name here to match the secret we created earlier
    try:
        secret = Client().get_secret("llm_complete")
        api_key = secret.secret_values["openai_api_key"]
    except Exception:
        # Fallback to environment variable for local testing
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OpenAI API Key not found in ZenML secrets or Environment!")

    # 2. Initialize Embeddings (This turns text into math vectors)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # 3. Split documents into smaller chunks for better AI retrieval
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    compiled_texts = text_splitter.split_documents(documents)

    # 4. Log Metadata (Enterprise best practice for tracking)
    log_artifact_metadata(
        artifact_name="vector_store",
        metadata={
            "embedding_type": "OpenAIEmbeddings",
            "vector_store_type": "FAISS",
            "chunk_size": 1000,
            "document_count": len(documents)
        },
    )

    # 5. Create and return the Vector Store
    return FAISS.from_documents(compiled_texts, embeddings)