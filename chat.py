import os
from zenml.client import Client
from langchain_classic.agents import AgentExecutor, ConversationalChatAgent
from langchain_ollama import ChatOllama 
from langchain_community.tools.vectorstore.tool import VectorStoreQATool
from agent.prompt import PREFIX, SUFFIX

def chat_with_agent():
    print("---  Initializing LOCAL ZenML Support Agent ---")
    client = Client()
    
    # 1. Fetch the Vector Store from ZenML
    run = client.get_pipeline("zenml_agent_creation_pipeline").last_successful_run
    vector_store = run.steps["agent_creator"].output.load()
    print(" Knowledge base loaded.")
    
    # 2. Setup Local LLM (DeepSeek)
    # Ensure Ollama is running in the background!
    llm = ChatOllama(model="deepseek-r1:7b", temperature=0)
    
    # 3. Build Tools locally
    tools = [
        VectorStoreQATool(
            name="zenml-qa-tool",
            vectorstore=vector_store,
            llm=llm,
            description="Use this tool to answer questions about ZenML documentation."
        ),
    ]
    
    # 4. Assemble Agent Logic
    system_prompt = PREFIX.format(character="technical assistant")
    my_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=system_prompt,
        human_message=SUFFIX,
    )

    executor = AgentExecutor.from_agent_and_tools(
        agent=my_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print("\n---  Local Agent is Online (No API Key Required) ---")
    while True:
        query = input("\nAsk ZenML a question (or type 'exit'): ")
        if query.lower() == 'exit': 
            break
        try:
            response = executor.invoke({
                "input": query,
                "chat_history": []
            })
            print(f"\nAgent: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_agent()