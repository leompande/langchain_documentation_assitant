import os
import ssl
from importlib.metadata import metadata
from typing import Any, Dict, List
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from ollama_embedding_progress_wrapper import ProgressOllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(
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
    """Process documents in batches concurrently using worker threads."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f" VectorStore Indexing: Preparing to add {len(documents)} documents to a vector store.",
        Colors.DARKCYAN
    )

    # Create batches
    batches = [
        documents[i:i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    total_batches = len(batches)
    log_info(f" VectorStore Indexing: Split into {total_batches} batches of {batch_size} documents each")

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            # Offload synchronous Chroma I/O to a worker thread to prevent event loop blocking
            await asyncio.to_thread(vector_store.add_documents, batch)
            log_success(
                f"Vector Store Indexing : Successfully added batch {batch_num}/{total_batches} ({len(batch)} documents).")
            return True
        except Exception as e:
            log_error(f"Vector Store Indexing : Failed to add batch {batch_num} - {e}")
            return False

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == total_batches:
        log_success(f"Vector Store Indexing : Processed all {successful}/{total_batches} batches successfully.")
    else:
        log_warning(
            f"VectorStore Indexing: Completed {successful}/{total_batches} batches. {total_batches - successful} failed."
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
        ) for result in res.get('results', [])
    ]
    log_success(
        f" Tavily Crawl: Successfully crawled {len(all_docs)} URLs from documentation site",
    )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(" Text Splitter : processing documents with 1000 chunk size and 100 overlap", Colors.YELLOW)

    # Optimized chunk size for embedding model text limits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents",
    )

    # Process documents concurrently in batches of 500
    await index_documents_async(splitted_docs, 500)

    log_header("PIPELINE COMPLETE")
    log_success(" Document Ingestion pipepline finished successfully.!")
    log_info("Summary", colors.BOLD)
    log_info(f"  .URLs mapped: {len(splitted_map['results'])}")
    log_info(f"  .Documents extracted: {len(splitted_docs)}")
    log_info(f"  .Chunks created: {len(splitted_docs)}")



if __name__ == "__main__":
    asyncio.run(main())
