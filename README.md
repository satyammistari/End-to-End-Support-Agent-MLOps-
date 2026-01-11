# ZenML Support Agent

A production-ready agent that can help you with your ZenML questions.

<div align="center">
  <img src="./assets/llm-agent/image.jpg" alt="ZenML Support Agent Architecture" width="80%">
</div>

## 🤖 What is the ZenML Support Agent?

The ZenML Support Agent is an intelligent assistant powered by Large Language Models (LLMs) that can answer questions about ZenML, its features, documentation, and best practices. It leverages the power of Retrieval Augmented Generation (RAG) to provide accurate, up-to-date information by accessing ZenML's documentation, examples, and other knowledge sources.

### Key Features

- **Intelligent Question Answering**: Get instant, accurate responses to your ZenML-related questions
- **Up-to-date Knowledge**: Automatically updates its knowledge base as ZenML documentation evolves
- **MLOps-powered Architecture**: Built with ZenML pipelines for reproducibility, versioning, and production readiness
- **Customizable**: Adapt the agent to your own data sources and knowledge base
- **Production-ready**: Designed to be deployed and maintained in production environments

## 🚀 Why Use the ZenML Support Agent?

Traditional documentation search can be time-consuming and may not always yield the most relevant results. The ZenML Support Agent:

- Understands natural language questions and provides concise, relevant answers
- Saves time by quickly retrieving information from across multiple knowledge sources
- Provides context-aware responses that consider the broader ZenML ecosystem
- Continuously improves as new information is added to the knowledge base

## 😯 Challenges with Productionizing LLM Agents

In principle, agents with question-answering capabilities work on the Retrieval Augmented Generation concept (RAG), which is popularly implemented in many frameworks and projects.
Upon a closer look and while using the agents in production, however, we need to address some challenges:

- Data that powers the agent's answering ability is constantly changing, new information comes in every day.
- Not feasible to manually generate vector stores based on the changing data. There's a need for automation.
- Tracking code changes to the vector store creation logic is important as it allows you to tie your outputs to the embeddings model and the other settings that you use.
- Tracking and versioning the agents that are created helps you manage what agents are used in production and allows easy switching.
- Important metadata like the agent's personality, the model parameters used for the LLM powering the agent, the type of vector store used, are essential while performing experiments to find the best performing combination.

## 🤝 ZenML + LLM Frameworks

There are various terms being tried out to describe this new paradigm — from LLMOps to Big Model Ops. We wanted to experience how users of ZenML might go about using our framework to integrate with these tools and models. We had seen lots of demos showcasing useful and striking use cases, but none addressed some of the complexities around deploying these in production.

What we showcase in this project is a way to build a pipeline around the vector store, and consequently the agent creation. This allows us to automate all the manual steps, among other benefits like making use of caching, tracking artifacts and more!

## 🛠️ The Agent Creation Pipeline

The ZenML Support Agent is built using a ZenML pipeline that handles data ingestion, constructs the vector store, and creates an intelligent agent. The pipeline consists of the following components:

### 📚 Data Sources

We carefully selected data sources that would provide the most value to users:

- **ZenML Documentation**: Comprehensive guides, tutorials, and reference materials
- **Example READMEs**: Starting points for ZenML users with practical implementation examples
- **Release Notes**: Latest features, improvements, and bug fixes

LangChain provides functions to help obtain content from various sources, but we implemented a custom scraper for maximum flexibility in retrieving web content from our documentation and examples.

### 🧠 Vector Store

We use [FAISS](https://faiss.ai) (Facebook AI Similarity Search) as our vector store, which provides:

- Efficient similarity search and clustering of high-dimensional vectors
- Fast, open-source implementation with excellent documentation
- Seamless integration with LangChain and other LLM frameworks

Documents are split into 1000-character chunks, combined with embeddings, and fed into the vector store, creating a queryable knowledge base.

### 🤖 Agent

The agent coordinates communication between the LLM and its tools. We use LangChain's `ConversationalChatAgent` class, which we customize with prompts that influence the style and personality of the responses.
