import os
from typing import Dict, List, Tuple

from agent.prompt import PREFIX, SUFFIX
# Use the classic library for legacy agent types to ensure stability
from langchain_classic.agents import AgentExecutor, ConversationalChatAgent
from langchain_core.vectorstores import VectorStore
from langchain_core.tools import BaseTool
from langchain_community.tools.vectorstore.tool import VectorStoreQATool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing_extensions import Annotated
from zenml import ArtifactConfig, log_metadata, step
from zenml.client import Client
from zenml.enums import ArtifactType

# Enterprise Config: Pull secrets at runtime to avoid hardcoding
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            secret = Client().get_secret("llm_complete")
            api_key = secret.secret_values["openai_api_key"]
        except Exception:
            api_key = None
    return api_key

class AgentParameters(BaseModel):
    """Configuration for the agent's LLM reasoning engine."""
    llm: Dict = {
        "temperature": 0,
        "max_tokens": 1000,
        "model_name": "gpt-3.5-turbo",
    }

    class Config:
        extra = "ignore"

@step()
def agent_creator(
    vector_store: VectorStore, 
    config: AgentParameters = AgentParameters()
) -> Annotated[
    Tuple[ConversationalChatAgent, List[BaseTool]],
    ArtifactConfig(name="agent", artifact_type=ArtifactType.DATA),
]:
    """Builds a conversational agent with document retrieval tools."""
    
    api_key = get_api_key()
    if not api_key:
        raise ValueError("OpenAI API Key missing! Check ZenML secrets or .env")

    # 1. Initialize the LLM with the secure key
    llm = ChatOpenAI(**config.llm, api_key=api_key)

    # 2. Define the 'Tools' (The Agent's Hands)
    # We use VectorStoreQATool to allow the agent to search the docs we scraped
    tools = [
        VectorStoreQATool(
            name="zenml-qa-tool",
            vectorstore=vector_store,
            llm=llm, # Pass the initialized LLM directly
            description="Use this tool to answer questions about ZenML documentation, "
                        "debugging errors, and conceptual abstractions."
        ),
    ]

    # 3. Assemble the Agent (The Agent's Brain)
    system_prompt = PREFIX.format(character="technical assistant")
    
    my_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=system_prompt,
        human_message=SUFFIX,
    )

    # 4. Professional Metadata Logging (Enterprise Observability)
    # We use the modern log_metadata instead of the deprecated log_artifact_metadata
    log_metadata(
        metadata={
            "agent_type": "ConversationalChatAgent",
            "model": config.llm["model_name"],
            "tool_count": len(tools),
            "tools_list": [t.name for t in tools],
        },
        infer_artifact=True
    )

    return my_agent, tools