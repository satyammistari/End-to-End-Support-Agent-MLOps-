from langchain_core.vectorstores import VectorStore
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType

@step()
def agent_creator(
    vector_store: VectorStore, 
) -> Annotated[
    VectorStore, 
    ArtifactConfig(name="zenml_knowledge_base", artifact_type=ArtifactType.DATA),
]:
    """
    Passes the vector store through to the dashboard.
    Building tools here causes serialization errors; we build them in chat.py instead.
    """
    # Simply return the vector store as the 'agent' artifact
    return vector_store