import os
import ssl
from importlib.metadata import metadata
from typing import Any, Dict,List
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from ollama_embedding_progress_wrapper import ProgressOllamaEmbeddings

from logger import (Colors, log_error, log_header,log_info,log_success,log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

embeddings = ProgressOllamaEmbeddings(
    model="qwen3-embedding:8b",
)

vector_store = Chroma(
    collection_name="document_assistant_db",
    embedding_function=embeddings,
    persist_directory="./chroma_document_assistant_db",
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f" VectorStore Indexing: Preparing to add {len(documents)} documents to a vector store.",
        Colors.DARKCYAN
    )

    # Create batches
    batches = [
        documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f" VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
             )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.add_documents(batch)
            log_success(f"Vector Store Indexing : Successfully added batch {batch_num/len(batch)} ({len(batch)}) documents to a vector store.")
        except Exception as e:
            log_error(f"Vector Store Indexing : Failed to add batch{batch_num} - {e}")
            return False
        return True

    #Process batches concurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks,return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(f"Vector Store Indexing : Processed {successful/len(batches)} batches successfully.")
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches failed."
        )

async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENT INGESTION PIPELINE")

    log_info(" TavilyCrawl: Starting to Crawl documentations from https://python.langchain.com/",
             Colors.PURPLE)

    # Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,
        "extra_depth": "advanced",
    })
    all_docs = [
        Document(
            page_content=result.get('raw_content') or "",
            metadata={"source": result.get('url', '')}
        ) for result in res['results']
    ]
    log_success(
        f" Tavily Crawl: Successfully crawled {len(all_docs)} URLs from documentation site",
    )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f" Text Splitter : processing {len(all_docs)} documents with 4000 chunk size and 200 overlap", Colors.YELLOW)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks form {len(all_docs)} documents",
    )

    # Process document asynchronously
    await index_documents_async(splitted_docs, 500)

if __name__ == "__main__":
    asyncio.run(main())
