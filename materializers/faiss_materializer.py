import os
from typing import Type, Any

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStore
# Use the same local embeddings we used in the index_generator
from langchain_huggingface import HuggingFaceEmbeddings 
from zenml.materializers.base_materializer import BaseMaterializer

class FAISSMaterializer(BaseMaterializer):
    """Enterprise materializer for FAISS vector stores using local embeddings."""

    ASSOCIATED_TYPES = (FAISS, VectorStore)

    def save(self, data: FAISS) -> None:
        """Saves the FAISS index to the ZenML artifact store."""
        # data.save_local writes the index.faiss and index.pkl files
        data.save_local(self.uri)

    def load(self, data_type: Type[Any]) -> FAISS:
        """Loads the FAISS index using local HuggingFace embeddings."""
        
        # 1. Initialize the SAME embedding model used during creation
        # This is critical so the vectors "align" correctly
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. Load from disk with 'dangerous' deserialization enabled
        # This is required because LangChain uses pickle for FAISS metadata
        return FAISS.load_local(
            self.uri,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )