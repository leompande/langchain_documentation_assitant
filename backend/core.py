import os
from typing import Any, Dict
from dotenv import load_dotenv
from httpx2 import query
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Initialize embedding models
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b",
)

# Initialize vector store
vector_store = Chroma(
    collection_name="document_assistant_db",
    embedding_function=embeddings,
    persist_directory="./chroma_document_assistant_db",
)

# Initialize chat model
model = init_chat_model("qwen3.8:27b",model_provider="ollama")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant information to answer user query about LangChain."""
    retrieved_docs = vector_store.as_retriever().invoke(query,k=4)

    # Serialized documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
        Run the RAG pipeline to answer a query using retrieved documentation.
        Args:
             query : The user's question
        Returns:
            Dictionary containing:
                 - answer: The generated answer
                 - context: List of retrieved documents
    """

    system_prompt = (
        "You are a helpful AI assistant the answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation."
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role":"user","content":query}]

    #Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extra t the context documents from the ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is ToolMessage  with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of documents objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    response = run_llm(query="What is langchain?")
    print(response)